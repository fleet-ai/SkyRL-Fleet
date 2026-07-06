"""Tests for tinker_shim history normalization + tool-call response shape.

The bug these tests guard: Kimi's chat template renders `tool_calls[].id`
verbatim into the prompt. When the shim used `id="call_<uuid>"`, the model
echoed `call_<uuid>` on subsequent turns and the parser dropped the call
(Kimi grammar requires `functions.<NAME>` prefix). Trainer side has always
used `functions.<NAME>:<turn>`, so the shim must match.

The tests pull the canonical id format from the family adapter itself
(`Kimi().build_assistant_message(...)['tool_calls'][0]['id']`) rather than
hardcoding `functions.foo:0`. If the trainer changes the format, these tests
stay correct without edits.

Run:
    uv run --extra dev --extra tinker pytest \
        integrations/fleet/tests/test_tinker_shim_history.py
"""

from __future__ import annotations

import pytest

from integrations.fleet.serving.tinker_shim import (
    _build_response_message,
    _canonical_id,
    _count_assistant_turns,
    _normalize_history_ids,
)
from skyrl_gym.envs.fleet_task.families import Kimi, get_family

# ---------------------------------------------------------------------------
# _normalize_history_ids
# ---------------------------------------------------------------------------


def _trainer_id(name: str, turn: int) -> str:
    """The id the trainer's Kimi adapter generates for (name, turn). Pulled
    from the adapter itself so tests track format drift automatically."""
    return _canonical_id(Kimi(), name, turn)


def test_normalize_empty():
    assert _normalize_history_ids([], Kimi()) == []


def test_normalize_no_tool_calls_passthrough():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out == msgs


def test_normalize_single_call_gets_trainer_id():
    msgs = [
        {"role": "user", "content": "find email"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "emails_search", "arguments": '{"q":"x"}'},
                }
            ],
        },
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[1]["tool_calls"][0]["id"] == _trainer_id("emails_search", 0)


def test_normalize_tool_result_id_remapped():
    msgs = [
        {"role": "user", "content": "find email"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "foo", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc123", "content": "result"},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    canonical = _trainer_id("foo", 0)
    assert out[1]["tool_calls"][0]["id"] == canonical
    assert out[2]["tool_call_id"] == canonical


def test_normalize_two_assistant_turns_get_different_turn_indices():
    msgs = [
        {"role": "user", "content": "a"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "r2"},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[1]["tool_calls"][0]["id"] == _trainer_id("foo", 0)
    assert out[2]["tool_call_id"] == _trainer_id("foo", 0)
    assert out[3]["tool_calls"][0]["id"] == _trainer_id("bar", 1)
    assert out[4]["tool_call_id"] == _trainer_id("bar", 1)


def test_normalize_content_only_assistant_still_advances_turn_counter():
    """An assistant message without tool_calls still counts as a turn —
    the next tool-calling assistant turn should land at index 1, not 0."""
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "thinking..."},  # no tool_calls
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
        },
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[3]["tool_calls"][0]["id"] == _trainer_id("foo", 1)


def test_normalize_idempotent_on_canonical_ids():
    canonical = _trainer_id("foo", 0)
    msgs = [
        {"role": "user", "content": "a"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": canonical, "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": canonical, "content": "r"},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    # Format identical → tc.id unchanged, tool_call_id unchanged, no remap
    assert out[1]["tool_calls"][0]["id"] == canonical
    assert out[2]["tool_call_id"] == canonical


def test_normalize_multi_tool_calls_in_one_assistant_turn():
    """Both calls in the same assistant message share the same turn index.
    The trainer's adapter would only emit one call_per_message, but the
    shim accepts multi-call inputs for compatibility — they all map to the
    same turn."""
    msgs = [
        {"role": "user", "content": "do two things"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[1]["tool_calls"][0]["id"] == _trainer_id("foo", 0)
    assert out[1]["tool_calls"][1]["id"] == _trainer_id("bar", 0)


def test_normalize_orphan_tool_result_id_passes_through():
    """If a tool result references an id we never saw in a prior assistant
    tool_call, leave it unchanged — don't crash, don't fabricate."""
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "tool", "tool_call_id": "call_orphan", "content": "??"},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[1]["tool_call_id"] == "call_orphan"


def test_normalize_missing_function_name_skipped():
    """A malformed tool_call without function.name shouldn't crash; we just
    leave its id alone."""
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {}}]},
    ]
    out = _normalize_history_ids(msgs, Kimi())
    assert out[1]["tool_calls"][0]["id"] == "call_1"  # unchanged


# ---------------------------------------------------------------------------
# _count_assistant_turns
# ---------------------------------------------------------------------------


def test_count_assistant_turns_zero():
    assert _count_assistant_turns([{"role": "user", "content": "x"}]) == 0


def test_count_assistant_turns_mixed():
    msgs = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "r"},
        {"role": "assistant", "content": "b"},
    ]
    assert _count_assistant_turns(msgs) == 2


# ---------------------------------------------------------------------------
# _build_response_message (parses sampled text → OpenAI message)
# ---------------------------------------------------------------------------


def test_build_response_message_kimi_tool_call(monkeypatch):
    """The shim parses Kimi tool-call syntax and emits OpenAI tool_calls
    with the trainer's canonical id (`functions.NAME:turn`)."""
    import integrations.fleet.serving.tinker_shim as shim

    monkeypatch.setattr(shim, "_family", Kimi())

    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.emails_search:0"
        '<|tool_call_argument_begin|>{"query":"Schedule"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    msg = _build_response_message(text, asst_turn=0)
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == _trainer_id("emails_search", 0)
    assert msg["tool_calls"][0]["function"]["name"] == "emails_search"


def test_build_response_message_content_only_fallback(monkeypatch):
    """When the model emits prose with no tool call, fall back to
    content-only — but never empty content (some clients reject that)."""
    import integrations.fleet.serving.tinker_shim as shim

    monkeypatch.setattr(shim, "_family", Kimi())

    msg = _build_response_message("I cannot help with that.", asst_turn=0)
    assert msg["role"] == "assistant"
    assert "tool_calls" not in msg or not msg["tool_calls"]
    assert msg["content"].strip()  # non-empty


def test_build_response_message_empty_text_promoted_to_space(monkeypatch):
    """Empty model output → content=' ' so OpenAI clients don't reject."""
    import integrations.fleet.serving.tinker_shim as shim

    monkeypatch.setattr(shim, "_family", Kimi())

    msg = _build_response_message("", asst_turn=0)
    assert msg["content"] == " "


def test_build_response_message_no_family_uses_passthrough(monkeypatch):
    """Unknown base model (no family adapter): emit content as-is, no parse."""
    import integrations.fleet.serving.tinker_shim as shim

    monkeypatch.setattr(shim, "_family", None)

    msg = _build_response_message("hello world", asst_turn=0)
    assert msg == {"role": "assistant", "content": "hello world"}


# ---------------------------------------------------------------------------
# Family registry sanity (catches drift if get_family stops resolving 'kimi')
# ---------------------------------------------------------------------------


def test_kimi_family_registered():
    fam = get_family("kimi")
    assert fam is not None
    assert fam.name == "kimi"


# ---------------------------------------------------------------------------
# Integration: render normalized history through real Kimi-K2.6 tokenizer
# and assert the rendered prompt contains the trainer's id format, NOT the
# incoming OpenAI uuid id. This is the test that PROVES the bug is fixed.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_render_uses_trainer_id_in_prompt():
    """Round-trip: build a history with OpenAI-style ids → normalize →
    apply_chat_template via real Kimi-K2.6 tokenizer → assert rendered text
    contains the canonical id and NOT the original `call_<uuid>`."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        tok = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2.6",
            trust_remote_code=True,
        )
    except Exception as e:
        pytest.skip(f"Kimi-K2.6 tokenizer unavailable: {e}")

    bad_id = "call_93be45f0e65f"
    msgs = [
        {"role": "user", "content": "find the schedule email"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": bad_id,
                    "type": "function",
                    "function": {
                        "name": "emails_search",
                        "arguments": '{"query":"Schedule"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": bad_id, "content": "Subject: Schedule"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emails_search",
                "description": "Search emails",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    normalized = _normalize_history_ids(msgs, Kimi())
    rendered = tok.apply_chat_template(
        normalized,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )

    canonical = _trainer_id("emails_search", 0)
    assert canonical in rendered, (
        f"expected canonical id {canonical!r} in rendered prompt; " f"last 1500 chars:\n{rendered[-1500:]}"
    )
    assert bad_id not in rendered, (
        f"OpenAI id {bad_id!r} leaked into rendered prompt; " f"last 1500 chars:\n{rendered[-1500:]}"
    )
