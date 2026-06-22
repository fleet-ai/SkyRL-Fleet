"""Tests for the per-family adapter (Kimi, Qwen) in fleet_task/families.py.

The adapter is what env.py delegates assistant-message construction +
per-turn reminder + reject-message generation to. The Kimi adapter is
the load-bearing one: it splits <think>...</think> into reasoning_content
to avoid the Kimi chat template's double-think bug. A/B verified
2026-06-21 by rendering Shape A (raw content) vs Shape B (structured
fields) through `tokenizer.apply_chat_template` and counting `<think>`
tags: Shape A renders TWO blocks per assistant turn (template default
empty + our raw inline), Shape B renders ONE. Tests below pin that
behavior at the unit level + the integration level (render-through).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.families import (
    Kimi,
    Qwen,
    family_for_model,
    get_family,
)


# --------------------------------------------------------------------------- #
# Registry + name derivation
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_kimi_lookup(self):
        f = get_family("kimi")
        assert f is not None
        assert f.name == "kimi"

    def test_qwen_lookup(self):
        f = get_family("qwen")
        assert f is not None
        assert f.name == "qwen"

    def test_unknown_returns_none(self):
        assert get_family("does-not-exist") is None
        assert get_family(None) is None
        assert get_family("") is None


class TestFamilyForModel:
    def test_kimi_prefix(self):
        assert family_for_model("moonshotai/Kimi-K2.6") == "kimi"
        assert family_for_model("moonshotai/Kimi-K2.6:peft:131072") == "kimi"

    def test_qwen_prefix(self):
        assert family_for_model("Qwen/Qwen3-8B") == "qwen"
        assert family_for_model("Qwen/Qwen2.5-7B-Instruct") == "qwen"

    def test_unknown_model(self):
        assert family_for_model("meta-llama/Llama-3.2-1B") is None
        assert family_for_model("") is None
        assert family_for_model(None) is None


# --------------------------------------------------------------------------- #
# Kimi adapter
# --------------------------------------------------------------------------- #

class TestKimiBuildAssistantMessage:
    def setup_method(self):
        self.kimi = Kimi()

    def test_canonical_tool_call_is_concrete_not_placeholder(self):
        """The YAML's previous placeholder template `functions.<name>:<n>`
        was being copied LITERALLY by the model into tool call names,
        causing MCP "Tool '<name>:<n>' not found" errors. The Kimi
        canonical must be a concrete real example."""
        canon = Kimi.canonical_tool_call
        assert "<name>" not in canon, "must not contain placeholder text"
        assert "<n>" not in canon
        # All 5 Kimi specials present (they're what makes this canonical)
        assert "<|tool_calls_section_begin|>" in canon
        assert "<|tool_call_begin|>" in canon
        assert "<|tool_call_argument_begin|>" in canon
        assert "<|tool_call_end|>" in canon
        assert "<|tool_calls_section_end|>" in canon

    def test_splits_think_into_reasoning_content(self):
        """The load-bearing transformation: <think>X</think> in raw text
        becomes reasoning_content="X", removed from `content`. Kimi chat
        template uses reasoning_content field to render ONE <think> block;
        leaving the raw <think> in content causes the double-think bug."""
        raw = (
            "<think>I should click the menu.</think>"
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.computer:5'
            '<|tool_call_argument_begin|>{"action":"screenshot"}'
            '<|tool_call_end|><|tool_calls_section_end|>'
        )
        parsed = {"name": "computer", "arguments": {"action": "screenshot"}}
        msg = self.kimi.build_assistant_message(raw, parsed, turn=5)
        assert msg["reasoning_content"] == "I should click the menu."
        # content has BOTH <think> AND the tool section stripped out
        assert "<think>" not in msg["content"]
        assert "<|tool_call" not in msg["content"]
        assert msg["content"] == ""

    def test_tool_calls_use_canonical_kimi_id_format(self):
        """Kimi was pretrained on `functions.NAME:N` id format. Using
        OpenAI-style `call_N` makes the chat template render a non-
        canonical id (loses the `functions.` prefix and the numeric
        counter the model is used to)."""
        parsed = {"name": "computer", "arguments": {"action": "screenshot"}}
        msg = self.kimi.build_assistant_message("<think>x</think>", parsed, turn=7)
        assert msg["tool_calls"][0]["id"] == "functions.computer:7"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "computer"
        # arguments is a JSON STRING per OpenAI spec, not a dict
        assert msg["tool_calls"][0]["function"]["arguments"] == json.dumps(
            {"action": "screenshot"}
        )

    def test_no_tool_call_omits_tool_calls_field(self):
        msg = self.kimi.build_assistant_message("<think>thinking</think>", None, turn=3)
        assert "tool_calls" not in msg

    def test_no_think_block_omits_reasoning_content(self):
        msg = self.kimi.build_assistant_message("just text response", None, turn=1)
        assert "reasoning_content" not in msg
        assert msg["content"] == "just text response"

    def test_multiple_think_blocks_concatenated(self):
        """If the model emits multiple <think> blocks, concatenate them
        for reasoning_content rather than picking one. Preserves the
        model's actual reasoning trace."""
        raw = "<think>first</think>middle<think>second</think>more"
        msg = self.kimi.build_assistant_message(raw, None, turn=1)
        assert msg["reasoning_content"] == "first\n\nsecond"
        # Free text between/after think blocks survives in content
        assert "middle" in msg["content"]
        assert "more" in msg["content"]


class TestKimiTemplateRoundTrip:
    """Render the structured shape through the actual Kimi tokenizer and
    confirm the double-think bug is gone. This is the load-bearing
    integration test."""

    def test_no_double_think_through_apply_chat_template(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        try:
            tok = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2.6", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Kimi tokenizer not available: {e}")

        kimi = Kimi()
        raw = (
            "<think>I should click the menu.</think>"
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.computer:1'
            '<|tool_call_argument_begin|>{"action":"screenshot"}'
            '<|tool_call_end|><|tool_calls_section_end|>'
        )
        parsed = {"name": "computer", "arguments": {"action": "screenshot"}}
        asst_msg = kimi.build_assistant_message(raw, parsed, turn=1)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            asst_msg,
        ]
        tools = [{"type": "function", "function": {"name": "computer", "description": "c",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        rendered = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)

        # PRE-fix: 3 <think> tags (template default empty + our raw + next-turn prompt).
        # POST-fix: 2 <think> tags (single reasoning_content render + next-turn prompt).
        # The empty-think regression hook fires only when reasoning_content is absent.
        assert rendered.count("<think>") == 2, (
            f"expected 2 <think> after fix (one for reasoning_content, one for next-turn "
            f"generation prompt), got {rendered.count('<think>')}. Rendered prompt:\n{rendered}"
        )
        assert "I should click the menu." in rendered
        # Tool call section still rendered canonically
        assert "<|tool_call_begin|>functions.computer:1<|tool_call_argument_begin|>" in rendered

    def test_pre_fix_shape_pins_the_bug(self):
        """Regression pin: if someone reverts to raw-content shape, the
        double-think bug surfaces. Documents what the bug looks like."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        try:
            tok = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2.6", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Kimi tokenizer not available: {e}")

        raw_shape = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            # Pre-fix shape: raw model emission stuffed into content
            {"role": "assistant",
             "content": "<think>x</think><|tool_calls_section_begin|>...<|tool_calls_section_end|>"},
        ]
        tools = [{"type": "function", "function": {"name": "computer", "description": "c",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        rendered = tok.apply_chat_template(raw_shape, tools=tools, add_generation_prompt=True, tokenize=False)
        assert rendered.count("<think>") == 3, (
            f"expected 3 <think> in the BUGGY pre-fix shape "
            f"(template empty + raw + next-turn), got {rendered.count('<think>')}"
        )


# --------------------------------------------------------------------------- #
# Qwen adapter
# --------------------------------------------------------------------------- #

class TestQwenBuildAssistantMessage:
    def setup_method(self):
        self.qwen = Qwen()

    def test_canonical_tool_call_is_none(self):
        """Qwen's chat template renders the format spec via the `tools`
        argument; echoing canonical_tool_call into the system prompt or
        reject would be dead weight. Confirm we don't try to inject."""
        assert Qwen.canonical_tool_call is None

    def test_passthrough_keeps_raw_content(self):
        """Qwen's chat template extracts <think>...</think> from content
        directly (see Qwen3 chat_template.jinja: content.split('</think>')
        ... split('<think>')[-1]). Raw passthrough renders cleanly with
        one think block. No restructuring needed."""
        raw = (
            "<think>thinking</think>"
            '<tool_call>{"name":"computer","arguments":{"action":"screenshot"}}</tool_call>'
        )
        parsed = {"name": "computer", "arguments": {"action": "screenshot"}}
        msg = self.qwen.build_assistant_message(raw, parsed, turn=3)
        # Content is the raw emission verbatim; Qwen template handles it
        assert msg["content"] == raw
        # Tool calls attached for trace-viewer linkage
        assert msg["tool_calls"][0]["id"] == "call_3"
        assert msg["tool_calls"][0]["function"]["name"] == "computer"

    def test_no_reasoning_content_field_set(self):
        """Setting reasoning_content on Qwen is harmless but redundant
        (Qwen template extracts from content). Don't bloat the dict."""
        msg = self.qwen.build_assistant_message("<think>x</think>foo", None, turn=1)
        assert "reasoning_content" not in msg


class TestQwenTemplateRoundTrip:
    def test_qwen_renders_one_think_block_from_raw_content(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        except Exception as e:
            pytest.skip(f"Qwen tokenizer not available: {e}")

        qwen = Qwen()
        raw = (
            "<think>plan</think>"
            '<tool_call>{"name":"c","arguments":{"a":1}}</tool_call>'
        )
        msg = qwen.build_assistant_message(raw, {"name": "c", "arguments": {"a": 1}}, turn=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            msg,
        ]
        tools = [{"type": "function", "function": {"name": "c", "description": "c",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        rendered = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        # Qwen template renders 1 <think> for the asst turn's reasoning.
        # Unlike Kimi, Qwen's next-turn generation prompt doesn't force a
        # leading <think> token, so total count is 1 not 2.
        assert rendered.count("<think>") == 1


# --------------------------------------------------------------------------- #
# Per-turn reminder + reject message
# --------------------------------------------------------------------------- #

class TestPerTurnReminderAndReject:
    def test_kimi_reminder_contains_indicator_and_canonical(self):
        s = Kimi().per_turn_reminder(turn=5, max_turns=64)
        assert "[Turn 5/64]" in s
        assert "[Next tool call format]" in s
        assert Kimi.canonical_tool_call in s

    def test_qwen_reminder_indicator_only(self):
        s = Qwen().per_turn_reminder(turn=5, max_turns=64)
        assert "[Turn 5/64]" in s
        assert "[Next tool call format]" not in s
        # Qwen reminder must NOT contain the Kimi-specific markers
        assert "<|tool_call_begin|>" not in s

    def test_kimi_reject_echoes_canonical(self):
        msg = Kimi().reject_message()
        assert "No tool call landed" in msg
        assert Kimi.canonical_tool_call in msg

    def test_qwen_reject_generic(self):
        msg = Qwen().reject_message()
        assert "No tool call landed" in msg
        # Qwen reject must NOT echo Kimi specials
        assert "<|tool_call_begin|>" not in msg


# --------------------------------------------------------------------------- #
# Special-token IDs land in context (Kimi-specific)
# --------------------------------------------------------------------------- #

class TestKimiSpecialTokenIdsInReminder:
    def test_kimi_per_turn_reminder_encodes_all_5_special_tokens(self):
        """The reminder is intended to plant the 5 Kimi tool special-token
        IDs (163595-163599) in the model's context every turn. Verify the
        canonical string we echo actually tokenizes to those IDs."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        try:
            tok = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2.6", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Kimi tokenizer not available: {e}")

        reminder = Kimi().per_turn_reminder(turn=1, max_turns=64)
        ids = set(tok.encode(reminder, add_special_tokens=False))
        for special_id in [163595, 163596, 163597, 163598, 163599]:
            assert special_id in ids, (
                f"Kimi tool special token {special_id} not in encoded reminder. "
                f"Encoded ids: {sorted(ids)[:30]}..."
            )
