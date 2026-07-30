"""ATOF event emission for SkyRL rollouts.

Streams rollout trajectories (LLM turns, env steps, rewards) through the
NeMo Relay runtime to the Fleet event pipeline (MSK -> ClickPipes ->
ClickHouse). Enabled by default; set NEMO_RELAY_ENABLED=0 to opt out. Init and
every emit fail open so training behavior is never affected.

Images never go inline: they upload to S3 keyed by content hash and events
carry the object URL. A payload that would exceed the Kafka broker cap
(only possible from malformed env output) is truncated and flagged.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

MSK_PLUGIN_KIND = "theseus-msk-atof-export"
DEFAULT_TOPIC = "atof.received"
DEFAULT_CLIENT_ID = "skyrl-nemo-relay"
DEFAULT_IMAGE_BUCKET = "fleet-trajectory-artifacts"
# Production MSK endpoint defaults keep every SkyRL path aligned. The env vars
# remain as overrides.
DEFAULT_MSK_BROKERS = (
    "b-1-public.tracesmskprod.v2hopy.c14.kafka.us-east-1.amazonaws.com:9198,"
    "b-2-public.tracesmskprod.v2hopy.c14.kafka.us-east-1.amazonaws.com:9198,"
    "b-3-public.tracesmskprod.v2hopy.c14.kafka.us-east-1.amazonaws.com:9198"
)
DEFAULT_TENANT_ID = "skyrl"
IMAGE_KEY_PREFIX = "skyrl"
# Broker cap is 20MB; leave headroom for the event envelope and metadata.
MAX_PAYLOAD_BYTES = 19_000_000
# Bound on plugin.initialize(): a hung init must disable ATOF, not stall
# training startup (drain_atof is bounded the same way).
ATOF_INIT_TIMEOUT_S = 10.0

_nemo_module: Any = None


def init_atof(
    *,
    entrypoint: str,
    run_name: str,
    model: str,
    agent_kind: Optional[str] = None,
) -> Optional["AtofEmitter"]:
    """Start the NeMo runtime and return an emitter, or None (disabled/failed).

    Enabled unless NEMO_RELAY_ENABLED=0. Uses either the MSK env vars
    (THESEUS_ATOF_MSK_BROKERS, THESEUS_ATOF_TENANT_ID) or SKYRL_ATOF_FILE_DIR
    for local file output. Any failure logs one warning and returns None.
    """
    global _nemo_module

    if os.environ.get("NEMO_RELAY_ENABLED", "1").strip().lower() in {"0", "false", "f", "no", "n", "off"}:
        return None

    component = _component_config()
    if component is None:
        return None
    os.environ["SKYRL_ATOF_RUN_NAME"] = run_name

    if component["kind"] == MSK_PLUGIN_KIND:
        config = component["config"]
        os.environ.setdefault("THESEUS_ATOF_MSK_BROKERS", config["brokers"])
        os.environ.setdefault("THESEUS_ATOF_TENANT_ID", config["tenant_id"])
        os.environ.setdefault("THESEUS_ATOF_MSK_TOPIC", config["topic"])
        os.environ.setdefault("THESEUS_ATOF_MSK_CLIENT_ID", config["client_id"])
        os.environ.setdefault("AWS_REGION", config["region"])
        try:
            from nemo_relay_runtime import get_nemo_runtime

            if get_nemo_runtime() is None:
                return None
            import nemo_relay
        except ImportError as exc:
            logger.warning(f"ATOF disabled: NeMo wheels not installed ({exc})")
            return None
        except Exception as exc:
            logger.warning(f"ATOF disabled: shared runtime initialization failed ({type(exc).__name__}: {exc})")
            return None
    else:
        try:
            import nemo_relay

            report = _run_sync(nemo_relay.plugin.initialize({"version": 1, "components": [component]}))
        except Exception as exc:
            logger.warning(f"ATOF disabled: plugin initialization failed ({type(exc).__name__}: {exc})")
            return None

        errors = _initialization_errors(report)
        if errors:
            logger.warning(f"ATOF disabled: plugin configuration errors {errors}")
            return None

    _nemo_module = nemo_relay
    logger.info(f"ATOF enabled: exporter={component['kind']} run={run_name}")
    return AtofEmitter(
        nemo_relay,
        entrypoint=entrypoint,
        run_name=run_name,
        model=model,
        agent_kind=agent_kind,
    )


def drain_atof(timeout: float = 5.0) -> None:
    """Flush buffered events before process exit. No-op when disabled."""
    if _nemo_module is None:
        return
    try:
        _nemo_module.plugin.drain(timeout)
    except Exception as exc:
        logger.warning(f"ATOF drain failed ({type(exc).__name__}: {exc})")


def install_atof(
    generator: Any,
    *,
    entrypoint: str,
    run_name: str,
    model: str,
    agent_kind: Optional[str] = None,
) -> Optional["AtofEmitter"]:
    """Init the runtime and install the emitter on a SkyRLGymGenerator.

    Must be called with the inner generator, before any wrapper like
    FleetTraceWrappedGenerator: the wrapper only delegates attribute reads,
    so setting the attribute on it would install nothing.
    """
    emitter = init_atof(
        entrypoint=entrypoint,
        run_name=run_name,
        model=model,
        agent_kind=agent_kind,
    )
    if emitter is not None:
        generator.atof_emitter = emitter
    return emitter


@dataclass
class RolloutTrace:
    handle: Any
    metadata: Dict[str, Any]
    image_urls: List[Dict[str, Any]] = field(default_factory=list)
    counters: Dict[str, int] = field(
        default_factory=lambda: {"truncated": 0, "image_upload_failures": 0, "emit_errors": 0}
    )


class AtofEmitter:
    """Emits one ATOF trace per rollout. Every method fails open."""

    def __init__(
        self,
        nemo: Any,
        *,
        entrypoint: str,
        run_name: str,
        model: str,
        agent_kind: Optional[str] = None,
    ):
        self._nemo = nemo
        self._entrypoint = entrypoint
        self._run_name = run_name
        self._model = model
        self._agent_kind = agent_kind
        self._image_bucket = os.environ.get("SKYRL_ATOF_IMAGE_BUCKET", DEFAULT_IMAGE_BUCKET)
        self._image_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="atof-image")
        self._uploaded_shas: set = set()
        self._s3 = None
        self._warned: set = set()

    def rollout_start(
        self,
        *,
        task_key: str,
        env_class: str,
        global_step: Optional[int],
        phase: Optional[str],
        sample_idx: Optional[int],
    ) -> Optional[RolloutTrace]:
        try:
            session_key = "\x1f".join(
                (
                    self._run_name,
                    self._entrypoint,
                    str(task_key or ""),
                    str(phase or ""),
                    "" if global_step is None else str(global_step),
                )
            )
            metadata = _drop_none(
                {
                    "producer_session_id": uuid.uuid5(uuid.NAMESPACE_URL, session_key).hex,
                    "trace_id": uuid.uuid4().hex,
                    "run_name": self._run_name,
                    "entrypoint": self._entrypoint,
                    "model": self._model,
                    "task_key": task_key,
                    "env_class": env_class,
                    "global_step": global_step,
                    "phase": phase,
                    "sample_idx": sample_idx,
                    "agent_kind": self._agent_kind or self._model,
                }
            )
            handle = self._nemo.scope.push(f"rollout:{task_key}", self._nemo.ScopeType.Agent, metadata=metadata)
            return RolloutTrace(handle=handle, metadata=metadata)
        except Exception as exc:
            self._warn_once("rollout_start", exc)
            return None

    def llm_call_metadata(self, **metadata: Any) -> Dict[str, Any]:
        """Build metadata for an LLM call that has no rollout scope."""
        try:
            trace_id = uuid.uuid4().hex
            values = {
                "producer_session_id": trace_id,
                "trace_id": trace_id,
                "run_name": self._run_name,
                "entrypoint": self._entrypoint,
                "model": self._model,
                **metadata,
            }
            if self._agent_kind is not None:
                values["agent_kind"] = self._agent_kind
            elif values.get("agent_kind") is None:
                values["agent_kind"] = self._model
            return _drop_none(values)
        except Exception as exc:
            self._warn_once("llm_call_metadata", exc)
            return {}

    def llm_request(self, trace: Optional[RolloutTrace], *, new_messages: List[dict]) -> Dict[str, Any]:
        """Build the bounded request recorded around a real provider call."""
        if trace is None:
            return {"messages": new_messages}
        try:
            return self._guard(trace, {"messages": self._offload_images(trace, new_messages)})
        except Exception as exc:
            trace.counters["emit_errors"] += 1
            self._warn_once("llm_request", exc)
            return {"messages": "[ATOF request capture failed]"}

    def env_step(
        self, trace: Optional[RolloutTrace], *, action: str, observations: List[dict], reward: float, done: bool
    ) -> None:
        if trace is None:
            return
        try:
            args = self._guard(trace, {"action": action})
            result = self._guard(
                trace,
                {"observations": self._offload_images(trace, observations), "reward": reward, "done": done},
            )
            handle = self._nemo.tools.call(
                "env_step",
                args,
                handle=trace.handle,
                metadata=trace.metadata,
            )
            self._nemo.tools.call_end(handle, result, metadata=trace.metadata)
        except Exception as exc:
            trace.counters["emit_errors"] += 1
            self._warn_once("env_step", exc)

    def rollout_end(
        self, trace: Optional[RolloutTrace], *, reward: float, stop_reason: Optional[str], num_turns: int
    ) -> None:
        if trace is None:
            return
        try:
            data = _drop_none(
                {
                    "reward": reward,
                    "stop_reason": stop_reason,
                    "num_turns": num_turns,
                    "image_urls": trace.image_urls or None,
                    "counters": trace.counters,
                }
            )
            self._nemo.scope.event("rollout_end", handle=trace.handle, data=data, metadata=trace.metadata)
            self._nemo.scope.pop(trace.handle, output={"reward": reward})
        except Exception as exc:
            self._warn_once("rollout_end", exc)

    def _offload_images(self, trace: RolloutTrace, messages: List[dict]) -> List[dict]:
        """Replace base64 image data URLs with content-addressed S3 URLs.

        Returns a copy when replacements happen; the caller's messages (the
        training chat history) are never mutated.
        """
        if not any(_message_has_data_image(message) for message in messages):
            return messages

        messages = copy.deepcopy(messages)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not (isinstance(item, dict) and item.get("type") == "image_url"):
                    continue
                url = (item.get("image_url") or {}).get("url") or ""
                if not url.startswith("data:image"):
                    continue
                image_bytes = base64.b64decode(url.split(",", 1)[1] if "," in url else url)
                sha = hashlib.sha256(image_bytes).hexdigest()
                key = f"{IMAGE_KEY_PREFIX}/{self._run_name}/{sha}"
                s3_url = f"s3://{self._image_bucket}/{key}"
                if sha not in self._uploaded_shas:
                    self._uploaded_shas.add(sha)
                    self._image_pool.submit(self._upload_image, trace, key, image_bytes)
                item["image_url"]["url"] = s3_url
                trace.image_urls.append({"url": s3_url, "sha256": sha, "bytes": len(image_bytes)})
        return messages

    def _upload_image(self, trace: RolloutTrace, key: str, image_bytes: bytes) -> None:
        try:
            if self._s3 is None:
                self._s3 = _make_s3_client()
            self._s3.put_object(Bucket=self._image_bucket, Key=key, Body=image_bytes)
        except Exception as exc:
            trace.counters["image_upload_failures"] += 1
            self._warn_once("image_upload", exc)

    def _guard(self, trace: RolloutTrace, payload: Dict[str, Any]) -> Dict[str, Any]:
        serialized = json.dumps(payload, default=str)
        if len(serialized) <= MAX_PAYLOAD_BYTES:
            return payload
        trace.counters["truncated"] += 1
        logger.error(
            f"ATOF payload of {len(serialized)} bytes exceeds the broker cap; truncating "
            f"(trace_id={trace.metadata.get('trace_id')}). An env is emitting oversized output."
        )
        return {
            "truncated": True,
            "original_bytes": len(serialized),
            "content": serialized[: MAX_PAYLOAD_BYTES - 1024],
        }

    def _warn_once(self, site: str, exc: Exception) -> None:
        if site in self._warned:
            return
        self._warned.add(site)
        logger.warning(f"ATOF emit failed at {site}; further failures logged once ({type(exc).__name__}: {exc})")


def _component_config() -> Optional[Dict[str, Any]]:
    file_dir = os.environ.get("SKYRL_ATOF_FILE_DIR")
    if file_dir:
        return {
            "kind": "observability",
            "enabled": True,
            "config": {
                "version": 2,
                "atof": {
                    "enabled": True,
                    "sinks": [
                        {
                            "type": "file",
                            "output_directory": file_dir,
                            "filename": "events.jsonl",
                            "mode": "append",
                        }
                    ],
                },
            },
        }

    brokers = os.environ.get("THESEUS_ATOF_MSK_BROKERS", DEFAULT_MSK_BROKERS).strip()
    tenant_id = os.environ.get("THESEUS_ATOF_TENANT_ID", DEFAULT_TENANT_ID).strip()
    if not brokers or not tenant_id:
        logger.warning(
            "ATOF disabled: NEMO_RELAY_ENABLED=1 but THESEUS_ATOF_MSK_BROKERS/THESEUS_ATOF_TENANT_ID are "
            "explicitly empty"
        )
        return None
    return {
        "kind": MSK_PLUGIN_KIND,
        "enabled": True,
        "config": {
            "brokers": brokers,
            "tenant_id": tenant_id,
            "topic": os.environ.get("THESEUS_ATOF_MSK_TOPIC", DEFAULT_TOPIC),
            "client_id": os.environ.get("THESEUS_ATOF_MSK_CLIENT_ID", DEFAULT_CLIENT_ID),
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "fail_open": True,
        },
    }


def _initialization_errors(report: Any) -> Optional[List[Any]]:
    """Error diagnostics from an initialize() report; malformed reports are errors."""
    if not isinstance(report, dict):
        return [{"level": "error", "message": f"invalid report type {type(report).__name__}"}]
    diagnostics = report.get("diagnostics") or []
    if not isinstance(diagnostics, list):
        return [{"level": "error", "message": "invalid diagnostics"}]
    return [d for d in diagnostics if isinstance(d, dict) and d.get("level") == "error"] or None


def _run_sync(value: Any, timeout: Optional[float] = None) -> Any:
    """Resolve plugin.initialize()'s awaitable from sync or async contexts, bounded.

    Always resolves on a daemon thread: wait_for bounds a cancellable hang,
    but a hang inside a blocking FFI call never reaches an await point, so
    the join timeout is the real backstop (and the daemon flag keeps an
    abandoned thread from blocking process exit). A timeout raises into
    init_atof's except, which warns once and disables ATOF.
    """
    if not inspect.isawaitable(value):
        return value
    if timeout is None:
        timeout = ATOF_INIT_TIMEOUT_S
    result: Dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(asyncio.wait_for(value, timeout))
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, name="atof-init", daemon=True)
    thread.start()
    thread.join(timeout + 1.0)
    if "error" in result:
        raise result["error"]
    if thread.is_alive():
        raise TimeoutError(f"nemo_relay plugin.initialize did not return within {timeout}s")
    return result.get("value")


def _make_s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def _message_has_data_image(message: dict) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") == "image_url"
        and ((item.get("image_url") or {}).get("url") or "").startswith("data:image")
        for item in content
    )


def _drop_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
