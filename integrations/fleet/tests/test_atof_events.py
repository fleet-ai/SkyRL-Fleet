from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import types
import uuid

import pytest

from integrations.fleet import atof_events
from integrations.fleet.atof_events import (
    MAX_PAYLOAD_BYTES,
    AtofEmitter,
    drain_atof,
    init_atof,
    producer_session_id,
)


class FakeNemo(types.ModuleType):
    """Records every runtime call the emitter makes."""

    def __init__(self, diagnostics=None):
        super().__init__("nemo_relay")
        self.calls = []
        fake = self

        class ScopeType:
            Agent = "agent"

        class LLMRequest:
            def __init__(self, headers, content):
                self.headers = headers
                self.content = content

        class plugin:
            @staticmethod
            def initialize(config):
                fake.calls.append(("initialize", config))
                return {"diagnostics": diagnostics or []}

            @staticmethod
            def drain(timeout):
                fake.calls.append(("drain", timeout))
                return True

        class scope:
            @staticmethod
            def push(name, scope_type, **kwargs):
                fake.calls.append(("push", name, scope_type, kwargs))
                return "handle-1"

            @staticmethod
            def pop(handle, **kwargs):
                fake.calls.append(("pop", handle, kwargs))

            @staticmethod
            def event(name, **kwargs):
                fake.calls.append(("event", name, kwargs))

        class llm:
            @staticmethod
            def execute(name, request, func, **kwargs):
                fake.calls.append(("llm", name, request.content, func(request), kwargs))

        class tools:
            @staticmethod
            def call(name, args, **kwargs):
                handle = f"tool-handle-{len(fake.named('tool_start')) + 1}"
                fake.calls.append(("tool_start", name, args, kwargs, handle))
                return handle

            @staticmethod
            def call_end(handle, result, **kwargs):
                fake.calls.append(("tool_end", handle, result, kwargs))

        self.ScopeType = ScopeType
        self.LLMRequest = LLMRequest
        self.plugin = plugin
        self.scope = scope
        self.llm = llm
        self.tools = tools

    def named(self, kind):
        return [call for call in self.calls if call[0] == kind]


class FakeS3:
    def __init__(self):
        self.uploads = []
        self.fail = False

    def put_object(self, *, Bucket, Key, Body):
        if self.fail:
            raise RuntimeError("s3 down")
        self.uploads.append((Bucket, Key, Body))


def install_fake_runtime(monkeypatch, runtime=object()):
    module = types.ModuleType("nemo_relay_runtime")
    module.get_nemo_runtime = lambda: runtime
    monkeypatch.setitem(sys.modules, "nemo_relay_runtime", module)


@pytest.fixture
def fake_nemo(monkeypatch):
    fake = FakeNemo()
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    install_fake_runtime(monkeypatch)
    monkeypatch.setattr(atof_events, "_nemo_module", None)
    return fake


@pytest.fixture
def msk_env(monkeypatch):
    monkeypatch.setenv("NEMO_RELAY_ENABLED", "1")
    monkeypatch.setenv("THESEUS_ATOF_MSK_BROKERS", "b-1:9198")
    monkeypatch.setenv("THESEUS_ATOF_TENANT_ID", "skyrl")


def make_emitter(fake_nemo, *, agent_kind=None):
    return AtofEmitter(
        fake_nemo,
        entrypoint="main_fleet",
        run_name="run-1",
        model="Qwen/Qwen3.5-9B",
        agent_kind=agent_kind,
    )


def data_image_message(image_bytes=b"png-bytes"):
    url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}


class TestInit:
    def test_enabled_without_flag(self, fake_nemo, msk_env, monkeypatch):
        monkeypatch.delenv("NEMO_RELAY_ENABLED")
        assert init_atof(entrypoint="e", run_name="r", model="m") is not None
        assert os.environ["SKYRL_ATOF_RUN_NAME"] == "r"

    @pytest.mark.parametrize("value", ["0", "false", "off"])
    def test_disabled_with_falsey_value(self, fake_nemo, msk_env, monkeypatch, value):
        monkeypatch.setenv("NEMO_RELAY_ENABLED", value)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None
        assert fake_nemo.calls == []

    def test_defaults_apply_without_msk_vars(self, fake_nemo, monkeypatch):
        monkeypatch.delenv("NEMO_RELAY_ENABLED", raising=False)
        monkeypatch.delenv("THESEUS_ATOF_MSK_BROKERS", raising=False)
        monkeypatch.delenv("THESEUS_ATOF_TENANT_ID", raising=False)
        assert init_atof(entrypoint="e", run_name="r", model="m") is not None
        assert os.environ["THESEUS_ATOF_MSK_BROKERS"] == atof_events.DEFAULT_MSK_BROKERS
        assert os.environ["THESEUS_ATOF_TENANT_ID"] == atof_events.DEFAULT_TENANT_ID

    def test_disabled_when_msk_vars_explicitly_empty(self, fake_nemo, monkeypatch):
        monkeypatch.setenv("NEMO_RELAY_ENABLED", "1")
        monkeypatch.setenv("THESEUS_ATOF_MSK_BROKERS", "")
        assert init_atof(entrypoint="e", run_name="r", model="m") is None
        assert fake_nemo.calls == []

    def test_disabled_when_wheel_missing(self, msk_env, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemo_relay", None)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None

    def test_disabled_when_shared_runtime_init_fails(self, fake_nemo, msk_env, monkeypatch):
        install_fake_runtime(monkeypatch, runtime=None)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None

    def test_msk_runtime_config(self, fake_nemo, msk_env):
        emitter = init_atof(entrypoint="main_fleet", run_name="run-1", model="m")
        assert isinstance(emitter, AtofEmitter)
        assert os.environ["THESEUS_ATOF_MSK_BROKERS"] == "b-1:9198"
        assert os.environ["THESEUS_ATOF_TENANT_ID"] == "skyrl"
        assert os.environ["THESEUS_ATOF_MSK_TOPIC"] == "atof.received"
        assert os.environ["NEMO_RELAY_ENABLED"] == "1"

    def test_file_exporter_mode(self, fake_nemo, monkeypatch, tmp_path):
        monkeypatch.setenv("NEMO_RELAY_ENABLED", "1")
        monkeypatch.setenv("SKYRL_ATOF_FILE_DIR", str(tmp_path))
        assert init_atof(entrypoint="e", run_name="r", model="m") is not None
        ((_, config),) = fake_nemo.named("initialize")
        (component,) = config["components"]
        assert component["kind"] == "observability"
        assert component["config"] == {
            "version": 2,
            "atof": {
                "enabled": True,
                "sinks": [
                    {
                        "type": "file",
                        "output_directory": str(tmp_path),
                        "filename": "events.jsonl",
                        "mode": "append",
                    }
                ],
            },
        }


class TestEmit:
    def test_rollout_start_stamps_metadata(self, fake_nemo):
        trace = make_emitter(fake_nemo).rollout_start(
            task_key="task-9", env_class="fleet_task", global_step=7, phase="train_step_7", sample_idx=2
        )
        ((_, name, scope_type, kwargs),) = fake_nemo.named("push")
        assert name == "rollout:task-9"
        assert scope_type == "agent"
        metadata = kwargs["metadata"]
        assert metadata["producer_session_id"] != "run-1"
        assert str(uuid.UUID(metadata["producer_session_id"])) == metadata["producer_session_id"]
        assert metadata["run_name"] == "run-1"
        assert metadata["global_step"] == 7
        assert metadata["phase"] == "train_step_7"
        assert metadata["entrypoint"] == "main_fleet"
        assert metadata["model"] == "Qwen/Qwen3.5-9B"
        assert metadata["agent_kind"] == "Qwen/Qwen3.5-9B"
        assert len(metadata["trace_id"]) == 32
        assert trace.metadata is metadata

    def test_producer_session_id_is_stable_and_uses_uuid_text(self):
        values = {
            "run_name": "run-1",
            "entrypoint": "main_fleet",
            "task_key": "task-9",
            "global_step": 7,
            "phase": "train_step_7",
            "job_id": "job-1",
        }

        session_id = producer_session_id(**values)

        assert producer_session_id(**values) == session_id
        assert str(uuid.UUID(session_id)) == session_id
        assert "-" in session_id
        assert producer_session_id(**{**values, "job_id": "job-2"}) != session_id

    def test_rollouts_group_samples_by_task_session(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        traces = [
            emitter.rollout_start(
                task_key=f"task-{task_index}",
                env_class="fleet_task",
                global_step=7,
                phase="train_step_7",
                sample_idx=sample_idx,
            )
            for task_index in range(4)
            for sample_idx in range(2)
        ]

        sessions_by_task = {}
        trace_ids = set()
        for trace in traces:
            metadata = trace.metadata
            sessions_by_task.setdefault(metadata["task_key"], set()).add(metadata["producer_session_id"])
            trace_ids.add(metadata["trace_id"])

        assert len(sessions_by_task) == 4
        assert all(len(session_ids) == 1 for session_ids in sessions_by_task.values())
        assert len({next(iter(session_ids)) for session_ids in sessions_by_task.values()}) == 4
        assert len(trace_ids) == 8

    def test_standalone_llm_call_uses_its_own_session(self, fake_nemo):
        metadata = make_emitter(fake_nemo).llm_call_metadata(call_site="judge", agent_kind="caller-value")

        assert metadata["producer_session_id"] == metadata["trace_id"]
        assert metadata["producer_session_id"] != "run-1"
        assert metadata["run_name"] == "run-1"
        assert metadata["agent_kind"] == "caller-value"

    def test_llm_request_payload(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        messages = [{"role": "user", "content": "solve it"}]
        assert emitter.llm_request(trace, new_messages=messages) == {"messages": messages}

    def test_env_step_payloads(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        obs = [{"role": "user", "content": "tool output"}]
        emitter.env_step(trace, action="<tool>ls</tool>", observations=obs, reward=0.5, done=True)
        ((_, name, args, kwargs, tool_handle),) = fake_nemo.named("tool_start")
        ((_, end_handle, result, end_kwargs),) = fake_nemo.named("tool_end")
        assert name == "env_step"
        assert args == {"action": "<tool>ls</tool>"}
        assert tool_handle == end_handle
        assert result == {"observations": obs, "reward": 0.5, "done": True}
        assert kwargs["handle"] == "handle-1"
        assert kwargs["metadata"]["agent_kind"] == "Qwen/Qwen3.5-9B"
        assert end_kwargs["metadata"]["agent_kind"] == "Qwen/Qwen3.5-9B"

    def test_env_step_finishes_before_rollout_end(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)

        emitter.env_step(trace, action="act", observations=[], reward=0.0, done=False)
        emitter.rollout_end(trace, reward=0.0, stop_reason="done", num_turns=1)

        ordered_calls = [call[0] for call in fake_nemo.calls if call[0] in {"tool_start", "tool_end", "event", "pop"}]
        assert ordered_calls == ["tool_start", "tool_end", "event", "pop"]

    def test_known_agent_kind_is_used_for_rollout_and_standalone_calls(self, fake_nemo):
        agent_kind = "skyrl_agent.agents.react.ReActAgent"
        emitter = make_emitter(fake_nemo, agent_kind=agent_kind)

        trace = emitter.rollout_start(
            task_key="t",
            env_class="fleet_task",
            global_step=1,
            phase="p",
            sample_idx=0,
        )
        emitter.env_step(trace, action="<tool>ls</tool>", observations=[], reward=0.0, done=False)

        assert trace.metadata["agent_kind"] == agent_kind
        ((_, _, _, tool_kwargs, _),) = fake_nemo.named("tool_start")
        assert tool_kwargs["metadata"]["agent_kind"] == agent_kind
        assert emitter.llm_call_metadata()["agent_kind"] == agent_kind

    def test_model_is_used_when_standalone_call_has_no_harness(self, fake_nemo):
        metadata = make_emitter(fake_nemo).llm_call_metadata(call_site="judge")

        assert metadata["agent_kind"] == "Qwen/Qwen3.5-9B"

    def test_rollout_end_marks_and_pops(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        emitter.rollout_end(trace, reward=1.0, stop_reason="stop", num_turns=3)
        (rollout_end,) = fake_nemo.named("event")
        assert rollout_end[1] == "rollout_end"
        assert rollout_end[2]["data"]["reward"] == 1.0
        assert rollout_end[2]["data"]["metadata"] == {"verifier_score": 1.0}
        assert rollout_end[2]["data"]["num_turns"] == 3
        assert rollout_end[2]["data"]["counters"] == {
            "truncated": 0,
            "image_upload_failures": 0,
            "emit_errors": 0,
        }
        ((_, handle, pop_kwargs),) = fake_nemo.named("pop")
        assert handle == "handle-1"
        assert pop_kwargs["output"] == {"reward": 1.0}

    @pytest.mark.parametrize(
        ("score", "expected_metadata"),
        [(1.0, {"verifier_score": 1.0}), (0.0, {"verifier_score": 0.0}), (None, None)],
    )
    def test_session_completed_marks_group(self, fake_nemo, score, expected_metadata):
        emitter = make_emitter(fake_nemo)

        emitter.session_completed(session_id="group-1", score=score)

        ((_, event_name, kwargs),) = fake_nemo.named("event")
        assert event_name == "session.completed"
        assert kwargs["metadata"] == {"session_id": "group-1"}
        assert kwargs["data"]["status"] == "completed"
        assert "ended_at" in kwargs["data"]
        if expected_metadata is None:
            assert "metadata" not in kwargs["data"]
        else:
            assert kwargs["data"]["metadata"] == expected_metadata

    def test_none_trace_is_noop(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        emitter.env_step(None, action="", observations=[], reward=0.0, done=False)
        emitter.rollout_end(None, reward=0.0, stop_reason=None, num_turns=0)
        assert fake_nemo.calls == []

    def test_llm_request_failure_is_swallowed_and_counted(self, fake_nemo, monkeypatch):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        monkeypatch.setattr(emitter, "_offload_images", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert emitter.llm_request(trace, new_messages=[]) == {"messages": "[ATOF request capture failed]"}
        assert trace.counters["emit_errors"] == 1


class TestImages:
    def test_data_url_replaced_with_s3_url_and_uploaded_once(self, fake_nemo, monkeypatch):
        s3 = FakeS3()
        monkeypatch.setattr(atof_events, "_make_s3_client", lambda: s3)
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        image_bytes = b"png-bytes"
        sha = hashlib.sha256(image_bytes).hexdigest()
        messages = [data_image_message(image_bytes), data_image_message(image_bytes)]

        request = emitter.llm_request(trace, new_messages=messages)
        emitter._image_pool.shutdown(wait=True)

        expected_url = f"s3://fleet-trajectory-artifacts/skyrl/run-1/{sha}"
        for message in request["messages"]:
            assert message["content"][0]["image_url"]["url"] == expected_url
        assert s3.uploads == [("fleet-trajectory-artifacts", f"skyrl/run-1/{sha}", image_bytes)]
        assert trace.image_urls == [
            {"url": expected_url, "sha256": sha, "bytes": len(image_bytes)},
            {"url": expected_url, "sha256": sha, "bytes": len(image_bytes)},
        ]
        # The caller's messages are untouched.
        assert messages[0]["content"][0]["image_url"]["url"].startswith("data:image")

    def test_event_emitted_even_when_upload_fails(self, fake_nemo, monkeypatch):
        s3 = FakeS3()
        s3.fail = True
        monkeypatch.setattr(atof_events, "_make_s3_client", lambda: s3)
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)

        request = emitter.llm_request(trace, new_messages=[data_image_message()])
        emitter._image_pool.shutdown(wait=True)

        assert request["messages"][0]["content"][0]["image_url"]["url"].startswith("s3://")
        assert trace.counters["image_upload_failures"] == 1

    def test_plain_text_messages_not_copied(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        messages = [{"role": "user", "content": "plain"}]
        assert emitter._offload_images(trace, messages) is messages


class TestGuard:
    def test_normal_payload_untouched(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        payload = {"action": "small"}
        assert emitter._guard(trace, payload) is payload
        assert trace.counters["truncated"] == 0

    def test_oversized_payload_truncated_and_flagged(self, fake_nemo):
        emitter = make_emitter(fake_nemo)
        trace = emitter.rollout_start(task_key="t", env_class="fleet_task", global_step=1, phase="p", sample_idx=0)
        payload = {"action": "x" * (MAX_PAYLOAD_BYTES + 100)}
        guarded = emitter._guard(trace, payload)
        assert guarded["truncated"] is True
        assert guarded["original_bytes"] > MAX_PAYLOAD_BYTES
        assert len(json.dumps(guarded)) <= MAX_PAYLOAD_BYTES
        assert trace.counters["truncated"] == 1


class TestDrain:
    def test_drain_delegates(self, fake_nemo, msk_env):
        init_atof(entrypoint="e", run_name="r", model="m")
        drain_atof(timeout=2.5)
        assert ("drain", 2.5) in fake_nemo.calls

    def test_drain_noop_when_disabled(self, fake_nemo):
        drain_atof()
        assert fake_nemo.named("drain") == []


class TestRunSyncTimeout:
    """The local file exporter must not stall startup."""

    def test_non_awaitable_passes_through(self):
        report = {"diagnostics": []}
        assert atof_events._run_sync(report) is report

    def test_awaitable_resolves(self):
        async def initialize():
            return {"diagnostics": []}

        assert atof_events._run_sync(initialize(), timeout=1.0) == {"diagnostics": []}

    def test_cancellable_hang_raises_timeout(self):
        async def hang_at_await_point():
            await asyncio.Event().wait()

        with pytest.raises(asyncio.TimeoutError):
            atof_events._run_sync(hang_at_await_point(), timeout=0.1)

    def test_blocking_hang_raises_timeout(self):
        # A hang inside a blocking call never reaches an await point, so
        # wait_for can't cancel it; only the join bound catches it.
        async def hang_without_await_point():
            time.sleep(3)

        with pytest.raises(TimeoutError):
            atof_events._run_sync(hang_without_await_point(), timeout=0.1)

    def test_hung_file_exporter_init_disables_atof(self, fake_nemo, monkeypatch, tmp_path):
        async def hang(config):
            await asyncio.Event().wait()

        monkeypatch.setenv("SKYRL_ATOF_FILE_DIR", str(tmp_path))
        monkeypatch.setattr(fake_nemo.plugin, "initialize", hang)
        monkeypatch.setattr(atof_events, "ATOF_INIT_TIMEOUT_S", 0.1)
        assert init_atof(entrypoint="e", run_name="r", model="m") is None
