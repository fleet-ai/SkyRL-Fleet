"""Tests for tinker_shim's SSE streaming response shape.

Some clients (e.g. claw-eval) send `stream: true` in OpenAI chat-completion
requests. Before the fix the shim raised HTTP 400 ("streaming not
supported"), which surfaced inside the bench as an `Error code: 400 -
{'detail': 'streaming not supported (v0)'}` per-task error — every task
failed at the LLM call layer, pass_rate landed as a hard 0.0 even though
the model was never given a chance to attempt anything.

Tinker's `sample_async` is one-shot, so the shim wraps the completion as
a single SSE content chunk followed by a final delta with finish_reason
and `[DONE]`. These tests assert that strict OpenAI SSE shape so a future
refactor doesn't quietly regress claweval back to 0%.

Run:
    uv run --extra dev --extra tinker pytest \
        integrations/fleet/tests/test_tinker_shim_streaming.py
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import integrations.fleet.serving.tinker_shim as shim


# Some FastAPI lifespan handlers need `_ready=True` to accept requests.
# We swap globals for a deterministic, dependency-free request path.


def _fake_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = [1, 2, 3, 4]
    tok.decode.return_value = "hello world"
    return tok


def _fake_sample_result(tokens=(10, 20, 30)):
    seq = MagicMock()
    seq.tokens = list(tokens)
    result = MagicMock()
    result.sequences = [seq]
    return result


def _fake_sampling_client():
    client = MagicMock()
    client.sample_async = AsyncMock(return_value=_fake_sample_result())
    return client


@pytest.fixture
def shim_client(monkeypatch):
    monkeypatch.setattr(shim, "_ready", True)
    monkeypatch.setattr(shim, "_tokenizer", _fake_tokenizer())
    monkeypatch.setattr(shim, "_sampling_client", _fake_sampling_client())
    monkeypatch.setattr(shim, "_family", None)  # passthrough message-builder
    monkeypatch.setattr(shim, "_model_id", "test-model")
    return TestClient(shim.app)


def _sse_chunks(text: str) -> list[dict | str]:
    """Parse the SSE body into a list of chunks (dict for JSON, '[DONE]' raw)."""
    out: list = []
    for raw in text.strip().split("\n\n"):
        if not raw.startswith("data: "):
            continue
        body = raw[len("data: "):]
        if body == "[DONE]":
            out.append("[DONE]")
        else:
            out.append(json.loads(body))
    return out


def test_streaming_returns_sse_when_stream_true(shim_client):
    resp = shim_client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_streaming_emits_content_then_final_then_done(shim_client):
    resp = shim_client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    chunks = _sse_chunks(resp.text)

    # 1 content chunk + 1 final chunk + [DONE]
    assert len(chunks) == 3
    assert chunks[-1] == "[DONE]"

    content_chunk, final_chunk = chunks[0], chunks[1]
    assert content_chunk["object"] == "chat.completion.chunk"
    assert content_chunk["choices"][0]["delta"]["role"] == "assistant"
    assert content_chunk["choices"][0]["finish_reason"] is None

    assert final_chunk["choices"][0]["delta"] == {}
    assert final_chunk["choices"][0]["finish_reason"] in {"stop", "tool_calls"}


def test_streaming_usage_fields_present_on_final_chunk(shim_client):
    """OpenAI clients (incl. claw-eval) read usage from the final chunk
    when stream_options.include_usage is set; some always read it. Include
    it unconditionally — it's free and harmless."""
    resp = shim_client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    chunks = _sse_chunks(resp.text)
    final = chunks[1]
    assert "usage" in final
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert k in final["usage"]
        assert isinstance(final["usage"][k], int)


def test_nonstreaming_path_still_works(shim_client):
    """Regression guard: adding the streaming branch must not break the
    default non-streaming response shape that all other benches rely on."""
    resp = shim_client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        # no stream=true
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body


def test_streaming_with_tool_calls_propagates_in_delta(shim_client, monkeypatch):
    """If the family adapter emits tool_calls, the SSE content chunk must
    include them in delta.tool_calls — without this, claw-eval would see
    a plain-text completion and miss the structured call."""
    from skyrl_gym.envs.fleet_task.families import Kimi
    monkeypatch.setattr(shim, "_family", Kimi())
    # Tokenizer decode returns a Kimi-syntax tool call
    tok = _fake_tokenizer()
    tok.decode.return_value = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.foo:0"
        "<|tool_call_argument_begin|>{\"x\":1}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    monkeypatch.setattr(shim, "_tokenizer", tok)

    resp = shim_client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    chunks = _sse_chunks(resp.text)
    content_chunk = chunks[0]
    tcs = content_chunk["choices"][0]["delta"].get("tool_calls")
    assert tcs and len(tcs) == 1
    assert tcs[0]["function"]["name"] == "foo"
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
