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
- Hermes-style tool_call parser: extract <tool_call>{...}</tool_call> blocks
  from decoded text, JSON-parse, expose as response.message.tool_calls.
- No streaming (stream=true -> 400). No multi-checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from typing import Any, Iterable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# NOTE: not importing from integrations.fleet.entrypoints.main_fleet_tinker —
# that module pulls in wandb / training-only deps not in the [tinker] extra.
# _normalize_token_list below handles the AutoTokenizer return-type variance
# (list[int] vs BatchEncoding) directly.


# ─── Globals (bound in startup) ──────────────────────────────────────────


_service_client: Any = None             # tinker.ServiceClient
_sampling_client: Any = None            # tinker.SamplingClient
_tokenizer: Any = None                  # transformers.AutoTokenizer
_ready: bool = False
_model_id: str = "tinker"               # served-model name surfaced in /v1/models
_token_logs: list[tuple[int, int]] = [] # (prompt_tokens, completion_tokens) running log


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
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = None
    stop: Optional[list[str] | str] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False


# ─── FastAPI app ─────────────────────────────────────────────────────────


app = FastAPI(title="fleet tinker shim", version="0.1.0")


@app.get("/health")
async def health():
    return {"ok": _ready, "model": _model_id}


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
    if req.stream:
        raise HTTPException(400, "streaming not supported (v0)")

    # Build the chat history. Pass `tools` through apply_chat_template so the
    # template renders them in the model's expected tool-spec block.
    msgs = [_msg_to_dict(m) for m in req.messages]
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
            msgs, add_generation_prompt=True, tokenize=True,
        )
    input_ids = _normalize_token_list(input_ids)

    # Sample.
    import tinker
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

    content_text, tool_calls = _parse_tool_calls(text)
    # Some OpenAI clients (tau2-bench) reject a message that has neither
    # content nor tool_calls. If the parser strips everything (e.g. model
    # emits only a </think> with no following prose and no valid tool-call
    # section), fall back to the raw decoded text so the message is non-empty.
    if not tool_calls and not (content_text and content_text.strip()):
        content_text = text or " "
    finish_reason = "tool_calls" if tool_calls else _finish_reason_from_result(result)

    completion_tokens = len(output_tokens)
    prompt_tokens = len(input_ids)
    _token_logs.append((prompt_tokens, completion_tokens))
    if len(_token_logs) % 10 == 0:
        # Periodic cost surface for the operator.
        pp = sum(p for p, _ in _token_logs)
        cc = sum(c for _, c in _token_logs)
        print(
            f"[tinker_shim] {len(_token_logs)} calls — "
            f"prompt={pp:,} completion={cc:,} tokens",
            file=sys.stderr,
            flush=True,
        )

    return JSONResponse(
        {
            "id": f"tinker-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


# ─── Tool-call parsing ───────────────────────────────────────────────────


# Hermes / Qwen3 convention.
_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL,
)

# Kimi-K2.x convention (observed live in Kimi-K2.6 sampling output):
#   <|tool_calls_section_begin|>
#     <|tool_call_begin|>functions.NAME:IDX <|tool_call_argument_begin|>{JSON}<|tool_call_end|>
#     ...
#   <|tool_calls_section_end|>
_KIMI_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(?P<body>.*?)<\|tool_calls_section_end\|>",
    re.DOTALL,
)
_KIMI_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*functions\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(?P<args>.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)

# Some chat templates surface only the closing reasoning tag; strip the prefix
# up through it so reasoning doesn't leak as user-facing content.
_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def _parse_tool_calls(text: str) -> tuple[Optional[str], Optional[list[dict]]]:
    """Returns (remaining_content, tool_calls_or_None).

    Recognises Kimi-K2 section/call delimiters first, then falls back to
    Hermes-style <tool_call>{...}</tool_call>. Strips a leading reasoning
    block ending in </think> so it isn't surfaced as content.
    """
    # Drop the reasoning prefix (Kimi emits </think> without an opener — the
    # opener was a special token that got stripped during decode).
    text = _THINK_CLOSE_RE.sub("", text, count=1)

    tool_calls: list[dict] = []
    stripped = text

    # Kimi format.
    section = _KIMI_SECTION_RE.search(text)
    if section is not None:
        for cm in _KIMI_CALL_RE.finditer(section.group("body")):
            name = cm.group("name")
            args_raw = cm.group("args").strip()
            try:
                args_obj = json.loads(args_raw)
                args_str = json.dumps(args_obj)
            except json.JSONDecodeError:
                # Pass through raw — OpenAI tool_calls.arguments is a string.
                args_str = args_raw
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        stripped = _KIMI_SECTION_RE.sub("", stripped)

    # Hermes fallback (additive — some templates mix both).
    for m in _HERMES_TOOL_CALL_RE.finditer(text):
        body = m.group("body").strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        args = parsed.get("arguments")
        if name is None:
            continue
        args_str = json.dumps(args) if not isinstance(args, str) else args
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    stripped = _HERMES_TOOL_CALL_RE.sub("", stripped)

    stripped = stripped.strip()
    if not tool_calls:
        return (stripped or text), None
    return (stripped or None), tool_calls


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
    global _service_client, _sampling_client, _tokenizer, _ready, _model_id

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

    print(f"[tinker_shim] loading tokenizer for {hf_model_name} (cold cache may take 1–2 min)…",
          file=sys.stderr, flush=True)
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
            f"[tinker_shim] FATAL: failed to create sampling_client for {checkpoint!r}: "
            f"{type(e).__name__}: {e}",
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
