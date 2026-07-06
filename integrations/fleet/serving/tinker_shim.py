"""OpenAI-compatible HTTP shim in front of a Tinker LoRA checkpoint.

Boot env:
    FLEET_TINKER_CHECKPOINT   = "tinker://<run-id>/sampler_weights/step_final"
    FLEET_TINKER_BASE_MODEL   = "moonshotai/Kimi-K2.6" (or whatever base)
    TINKER_API_KEY            = required by tinker SDK
    TINKER_SHIM_HOST          = default "0.0.0.0"
    TINKER_SHIM_PORT          = default 8000

Endpoints:
    GET  /health                  -> 200 once sampler+tokenizer ready
    GET  /v1/models               -> list one served model
    POST /v1/chat/completions     -> OpenAI chat completion (non-stream only)

Design notes:
- One sampling_client per process, bound at startup. Restart to switch checkpoints.
- Tokenizer loaded once with trust_remote_code=True. Kimi-K2.6 cold load is ~2 min.
- Tool-call parsing and assistant-message construction reuse the SkyRL trainer's
  family adapters (skyrl_gym.envs.fleet_task.families) and parser
  (skyrl_gym.envs.fleet_task.tool_call_parser). The shim therefore renders
  history to the chat template in exactly the format the model was trained on
  (Kimi: `id="functions.<NAME>:<turn>"`), avoiding the parser-drop bug seen
  with arbitrary `call_<uuid>` ids.
- No streaming (stream=true -> 400). No multi-checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Iterable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from skyrl_gym.envs.fleet_task.families import Family, family_for_model, get_family
from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

# ─── Globals (bound in startup) ──────────────────────────────────────────


_service_client: Any = None  # tinker.ServiceClient
_sampling_client: Any = None  # tinker.SamplingClient
_tokenizer: Any = None  # transformers.AutoTokenizer
_family: Optional[Family] = None  # per-base-model family adapter
_ready: bool = False
_model_id: str = "tinker"  # served-model name surfaced in /v1/models
_token_logs: list[tuple[int, int]] = []  # (prompt_tokens, completion_tokens) running log


# ─── Request / response shapes (OpenAI v1 chat completion) ───────────────


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Default 4096: 512 truncates Kimi-K2.6 mid-tool-call. The Kimi grammar
    # is <|tool_call_begin|>name<|tool_call_argument_begin|>{json}<|tool_call_end|>;
    # a single Bash python heredoc args block routinely runs 1-3K tokens
    # (especially in Chinese, which is denser per char). When truncated,
    # the partial JSON fails to parse and the shim's tool_call_parser falls
    # back to arguments={} -> tool dispatched with no payload -> sandbox
    # 422s -> model loops on the same broken call until max_turns.
    # Measured at 100% of 2,479 dispatches across the claweval 26-06-30
    # run before this bump.
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = None
    stop: Optional[list[str] | str] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False


# ─── FastAPI app ─────────────────────────────────────────────────────────


app = FastAPI(title="fleet tinker shim", version="0.2.0")


@app.get("/health")
async def health():
    return {"ok": _ready, "model": _model_id, "family": _family.name if _family else None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fleet-tinker",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not _ready:
        raise HTTPException(503, "shim not ready")

    # Coerce incoming OpenAI-style ids to the trainer's canonical format BEFORE
    # rendering through apply_chat_template. Kimi's chat template emits
    # tool_calls[].id verbatim — if we let `call_<uuid>` ids reach the template,
    # the model echoes them on subsequent turns and the parser drops the call.
    msgs = [_msg_to_dict(m) for m in req.messages]
    msgs = _normalize_history_ids(msgs, _family) if _family is not None else msgs

    try:
        input_ids = _tokenizer.apply_chat_template(
            msgs,
            tools=req.tools,
            add_generation_prompt=True,
            tokenize=True,
        )
    except TypeError:
        # Older tokenizers don't accept `tools=...`; retry without it.
        input_ids = _tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )
    input_ids = _normalize_token_list(input_ids)

    # Sample.
    from tinker import types

    sampling_params = types.SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p if req.top_p is not None else 1.0,
        stop=_normalize_stop(req.stop),
    )

    try:
        result = await _sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(tokens=input_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
    except Exception as e:
        raise HTTPException(502, f"tinker sample error: {type(e).__name__}: {e}")

    output_tokens = list(result.sequences[0].tokens)
    text = _tokenizer.decode(output_tokens, skip_special_tokens=True)

    asst_turn = _count_assistant_turns(msgs)
    message = _build_response_message(text, asst_turn)
    finish_reason = "tool_calls" if message.get("tool_calls") else _finish_reason_from_result(result)

    completion_tokens = len(output_tokens)
    prompt_tokens = len(input_ids)
    _token_logs.append((prompt_tokens, completion_tokens))
    if len(_token_logs) % 10 == 0:
        # Periodic cost surface for the operator.
        pp = sum(p for p, _ in _token_logs)
        cc = sum(c for _, c in _token_logs)
        print(
            f"[tinker_shim] {len(_token_logs)} calls — " f"prompt={pp:,} completion={cc:,} tokens",
            file=sys.stderr,
            flush=True,
        )

    response_id = f"tinker-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        # Tinker SDK returns one-shot completions; emit it as a single SSE
        # chunk + a final delta with finish_reason + [DONE]. Strictly
        # OpenAI-compatible SSE format so clients like claw-eval that send
        # stream=true get back a valid stream instead of a 400 error and
        # fail every task with the API error in their results.
        delta = {"role": "assistant"}
        if message.get("content") is not None:
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]

        async def event_stream():
            content_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": _model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": _model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": _model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


# ─── Tool-call: build response from sampled text via family adapter ──────


def _build_response_message(text: str, asst_turn: int) -> dict:
    """Parse sampled text into an OpenAI assistant message using the trainer's
    family adapter. Falls back to a content-only message if no family is bound
    (unknown base model)."""
    if _family is None:
        return {"role": "assistant", "content": _nonempty(text)}

    parsed = parse_tool_call(text)
    msg = _family.build_assistant_message(text, parsed, asst_turn)

    # Wire-format boundary: the family adapter may carry `arguments` as a
    # dict (Qwen3.6's chat template iterates arguments|items, so the TRAINER
    # needs the dict), but the OpenAI response schema requires a JSON string —
    # strict clients (openai SDK pydantic models) reject a dict and the whole
    # completion fails client-side (claweval jobs 7f6568a0/f487d06d: every
    # request dropped, 0 turns). Serialize here, at the API edge only.
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if isinstance(fn.get("arguments"), (dict, list)):
            fn["arguments"] = json.dumps(fn["arguments"])

    # OpenAI clients (tau2-bench, BFCL) expect `content` to be either non-empty
    # or null. The family adapter may leave `content` empty when the model
    # emitted only a tool call. Promote empty content to a single space so
    # the response is OpenAI-valid AND non-null.
    if not msg.get("tool_calls") and not (msg.get("content") or "").strip():
        msg["content"] = _nonempty(text)
    return msg


def _nonempty(s: Optional[str]) -> str:
    return s if (s and s.strip()) else " "


# ─── History normalization: align incoming ids with trainer format ───────


def _normalize_history_ids(msgs: list[dict], family: Family) -> list[dict]:
    """Rewrite incoming OpenAI-style `call_<uuid>` tool_call ids to the
    trainer's canonical format BEFORE apply_chat_template. The Kimi chat
    template renders tool_calls[].id verbatim; the format the model sees in
    history must match training (`functions.<NAME>:<turn>`) or the model
    emits the wrong format on subsequent turns and the parser drops every
    follow-up call.

    Uses `family.build_assistant_message` as the SINGLE source of truth for
    the id format — if the trainer's format ever changes, the shim follows
    by import.

    Idempotent: ids already produced by the family adapter pass through
    unchanged. Tool-result messages have their `tool_call_id` remapped via
    the same lookup so the OpenAI tool/result pairing stays consistent.
    """
    out: list[dict] = []
    asst_turn = 0
    remap: dict[str, str] = {}
    for raw in msgs:
        m = dict(raw)
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                new_calls = []
                for tc in calls:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    name = fn.get("name")
                    # Inbound history carries OpenAI wire-format arguments (a
                    # JSON string); each family's template needs its own type
                    # (Qwen3.6: dict, Kimi: string). Coerce BEFORE
                    # apply_chat_template or Qwen's `arguments|items` raises
                    # on every multi-turn request.
                    if "arguments" in fn and hasattr(family, "coerce_tool_call_arguments"):
                        fn["arguments"] = family.coerce_tool_call_arguments(fn["arguments"])
                    tc["function"] = fn
                    old_id = tc.get("id", "") or ""
                    if name:
                        canonical_id = _canonical_id(family, name, asst_turn)
                        if old_id != canonical_id:
                            if old_id:
                                remap[old_id] = canonical_id
                            tc["id"] = canonical_id
                    new_calls.append(tc)
                m["tool_calls"] = new_calls
            asst_turn += 1
        elif role == "tool":
            old_id = m.get("tool_call_id")
            if old_id and old_id in remap:
                m["tool_call_id"] = remap[old_id]
        out.append(m)
    return out


def _canonical_id(family: Family, name: str, turn: int) -> str:
    """Synthesize the trainer's canonical tool_call id for (name, turn) by
    asking the family adapter to build a dummy assistant message and
    extracting the id it generates. This keeps the id-format definition in
    exactly ONE place (families.py)."""
    synth = family.build_assistant_message("", {"name": name, "arguments": {}}, turn)
    calls = synth.get("tool_calls") or []
    if not calls:
        # Adapter didn't emit a tool_calls section — leave id unchanged.
        return ""
    return calls[0]["id"]


def _count_assistant_turns(msgs: list[dict]) -> int:
    return sum(1 for m in msgs if m.get("role") == "assistant")


# ─── Helpers ─────────────────────────────────────────────────────────────


def _msg_to_dict(m: ChatMessage) -> dict:
    d: dict = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    if m.tool_calls is not None:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        d["name"] = m.name
    return d


def _normalize_stop(stop: Optional[list[str] | str]) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _normalize_token_list(x: Iterable[int] | Any) -> list[int]:
    if hasattr(x, "input_ids"):
        return list(x.input_ids)
    if isinstance(x, dict) and "input_ids" in x:
        return list(x["input_ids"])
    return list(x)


def _finish_reason_from_result(result: Any) -> str:
    # Tinker SDK doesn't expose a strict finish reason; default to "stop".
    return "stop"


# ─── Startup: bind sampler + tokenizer ───────────────────────────────────


@app.on_event("startup")
async def _on_startup() -> None:
    global _service_client, _sampling_client, _tokenizer, _family, _ready, _model_id

    checkpoint = os.environ.get("FLEET_TINKER_CHECKPOINT", "").strip()
    base_model = os.environ.get("FLEET_TINKER_BASE_MODEL", "").strip()
    if not checkpoint or not base_model:
        print(
            "[tinker_shim] FATAL: FLEET_TINKER_CHECKPOINT and FLEET_TINKER_BASE_MODEL must be set.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    if not os.environ.get("TINKER_API_KEY"):
        print("[tinker_shim] FATAL: TINKER_API_KEY missing.", file=sys.stderr, flush=True)
        sys.exit(2)

    _model_id = checkpoint  # the OpenAI 'model' field is informational; surface the path

    import tinker
    from transformers import AutoTokenizer  # type: ignore

    hf_model_name = base_model.split(":peft:")[0]

    _family = get_family(family_for_model(hf_model_name))
    print(
        f"[tinker_shim] family={_family.name if _family else None} for base={hf_model_name}",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"[tinker_shim] loading tokenizer for {hf_model_name} (cold cache may take 1–2 min)…",
        file=sys.stderr,
        flush=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    print(f"[tinker_shim] creating sampling_client for {checkpoint}…", file=sys.stderr, flush=True)
    _service_client = tinker.ServiceClient()
    try:
        _sampling_client = await _service_client.create_sampling_client_async(model_path=checkpoint)
    except AttributeError:
        # Older SDK exposes a sync constructor.
        _sampling_client = _service_client.create_sampling_client(model_path=checkpoint)
    except Exception as e:
        print(
            f"[tinker_shim] FATAL: failed to create sampling_client for {checkpoint!r}: " f"{type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    _ready = True
    print(f"[tinker_shim] ready on model={_model_id}", file=sys.stderr, flush=True)


# ─── CLI / uvicorn launcher ──────────────────────────────────────────────


def main() -> None:
    import uvicorn

    host = os.environ.get("TINKER_SHIM_HOST", "0.0.0.0")
    port = int(os.environ.get("TINKER_SHIM_PORT", "8000"))
    uvicorn.run(
        "integrations.fleet.serving.tinker_shim:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
