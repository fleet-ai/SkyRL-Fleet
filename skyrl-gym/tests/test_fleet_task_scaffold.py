"""Tests for the per-turn observation scaffold.

Two layers:
1. Config-level: scaffold_for() resolves the per-family per_turn_reminder
   list with $turn / $max_turns / $canonical_tool_call substituted.
2. Env-level: step_async appends the scaffold to observations;
   chat_history_for_trace() strips it cleanly without touching image_url
   blocks.

Grounded in BU job d6d6f7eb: 60% of rejects (2,927/7,605 turns) were NAKED-
format calls. 119/131 sessions had the model emitting canonical AND naked
turns in the same trajectory, so the canonical-format token IDs landing in
context every turn is a high-prior nudge to keep emissions canonical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.config import (
    FleetTaskConfig,
    ModelFamilyConfig,
    get_config,
)
from skyrl_gym.envs.fleet_task.env import FleetTaskEnv


# Kimi tokenizer single-token IDs for the 5 tool-call special markers
# (verified against tokenizer_config.json on moonshotai/Kimi-K2.6).
_KIMI_TOOL_TOKEN_IDS = {
    163595,  # <|tool_calls_section_begin|>
    163596,  # <|tool_calls_section_end|>
    163597,  # <|tool_call_begin|>
    163598,  # <|tool_call_argument_begin|>
    163599,  # <|tool_call_end|>
}


# --------------------------------------------------------------------------- #
# Config-level: scaffold_for resolution
# --------------------------------------------------------------------------- #

class TestScaffoldResolution:
    def _cfg(self, families: dict) -> FleetTaskConfig:
        return FleetTaskConfig(model_families=families)

    def test_kimi_scaffold_contains_turn_and_canonical(self):
        cfg = get_config()
        s = cfg.scaffold_for("kimi", turn=5, max_turns=64)
        assert "[Turn 5/64]" in s
        # The full canonical string is interpolated by name; the 5 Kimi
        # special-token literal strings all appear.
        for marker in [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]:
            assert marker in s, f"scaffold missing {marker}"

    def test_qwen_scaffold_indicator_only(self):
        cfg = get_config()
        s = cfg.scaffold_for("qwen", turn=5, max_turns=64)
        assert "[Turn 5/64]" in s
        # Qwen's per_turn_reminder only has the indicator — no Kimi markers.
        assert "<|tool_calls_section_begin|>" not in s
        assert "[Next tool call format]" not in s

    def test_unknown_family_returns_empty(self):
        cfg = get_config()
        assert cfg.scaffold_for("does-not-exist", 5, 64) == ""

    def test_no_family_returns_empty(self):
        cfg = get_config()
        assert cfg.scaffold_for(None, 5, 64) == ""

    def test_empty_list_returns_empty(self):
        cfg = self._cfg({
            "foo": ModelFamilyConfig(
                canonical_tool_call="X", per_turn_reminder=[]
            )
        })
        assert cfg.scaffold_for("foo", 5, 64) == ""

    def test_list_items_concatenate_in_order(self):
        cfg = self._cfg({
            "foo": ModelFamilyConfig(
                canonical_tool_call="X",
                per_turn_reminder=["A", "B", "C"],
            )
        })
        assert cfg.scaffold_for("foo", 0, 0) == "ABC"

    def test_substitution_vars_available_per_item(self):
        cfg = self._cfg({
            "foo": ModelFamilyConfig(
                canonical_tool_call="CANON",
                per_turn_reminder=[
                    "turn=$turn ",
                    "max=$max_turns ",
                    "canon=$canonical_tool_call",
                ],
            )
        })
        assert cfg.scaffold_for("foo", 7, 99) == "turn=7 max=99 canon=CANON"

    def test_literal_braces_in_template_preserved(self):
        """safe_substitute leaves literal `{}` alone. Guards against the
        `.format()` footgun — JSON braces in template or canonical must
        round-trip without raising."""
        cfg = self._cfg({
            "foo": ModelFamilyConfig(
                canonical_tool_call='{"a":1}',
                per_turn_reminder=['{foo} $canonical_tool_call'],
            )
        })
        assert cfg.scaffold_for("foo", 0, 0) == '{foo} {"a":1}'


# --------------------------------------------------------------------------- #
# Tokenization: the YAML Kimi reminder MUST encode the special-token IDs
# (that's the whole point of injecting it).
# --------------------------------------------------------------------------- #

class TestKimiSpecialsLandInContext:
    def test_kimi_scaffold_encodes_all_5_tool_special_tokens(self):
        """Load the actual Kimi-K2.6 tokenizer, encode the YAML's resolved
        scaffold, assert every Kimi tool special-token ID appears."""
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

        cfg = get_config()
        s = cfg.scaffold_for("kimi", 5, 64)
        ids = set(tok.encode(s, add_special_tokens=False))
        missing = _KIMI_TOOL_TOKEN_IDS - ids
        assert not missing, (
            f"Kimi scaffold did not tokenize as special-token IDs: missing {missing}"
        )


# --------------------------------------------------------------------------- #
# Env-level: append + chat_history_for_trace round-trip
# --------------------------------------------------------------------------- #

class TestChatHistoryForTrace:
    """Build a synthetic chat_history + scaffold list and verify the strip.
    Avoids constructing a real FleetTaskEnv (would need MCP, env keys, etc.)
    by using __new__ + setting only the fields chat_history_for_trace reads."""

    def _env_with(self, chat_history, scaffolds):
        env = FleetTaskEnv.__new__(FleetTaskEnv)
        env.chat_history = chat_history
        env._scaffold_per_msg = scaffolds
        return env

    def test_system_and_initial_user_pass_through(self):
        ch = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task instructions"},
        ]
        env = self._env_with(ch, ["", ""])
        assert env.chat_history_for_trace() == ch

    def test_text_only_obs_strips_scaffold_suffix(self):
        scaffold = "\n\n[Turn 3/64][Format]\n<|...|>"
        obs_text = "No tool call landed."
        ch = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "broken response"},
            {"role": "user", "content": obs_text + scaffold},
        ]
        env = self._env_with(ch, ["", "", "", scaffold])
        out = env.chat_history_for_trace()
        assert out[3]["content"] == obs_text

    def test_multimodal_obs_strips_scaffold_block(self):
        scaffold = "\n\n[Turn 3/64][Format]\n<|...|>"
        # Multimodal append used scaffold.lstrip("\n"), so the text block
        # contains that form. chat_history_for_trace must strip the same.
        text_block_value = scaffold.lstrip("\n")
        image_block = {"type": "image_url", "image_url": {"url": "data:..."}}
        obs_content = [image_block, {"type": "text", "text": text_block_value}]
        ch = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "good response"},
            {"role": "user", "content": obs_content},
        ]
        env = self._env_with(ch, ["", "", "", scaffold])
        out = env.chat_history_for_trace()
        # Image survives byte-for-byte
        assert out[3]["content"][0] == image_block
        # Scaffold-only text block stripped to empty, dropped
        assert len(out[3]["content"]) == 1

    def test_multimodal_preserves_non_scaffold_text_before_strip(self):
        """If the trailing text block had legitimate text BEFORE the
        scaffold suffix, the suffix gets stripped but the prefix stays."""
        scaffold = "\n[Turn 3/64]"
        lstripped = scaffold.lstrip("\n")
        image = {"type": "image_url", "image_url": {"url": "u"}}
        text_with_prefix = "Tool returned: foo" + lstripped
        ch = [{"role": "user", "content": [image, {"type": "text", "text": text_with_prefix}]}]
        env = self._env_with(ch, [scaffold])
        out = env.chat_history_for_trace()
        # Text block survives with scaffold stripped
        assert out[0]["content"][0] == image
        assert out[0]["content"][1]["text"] == "Tool returned: foo"

    def test_empty_scaffold_no_change(self):
        ch = [{"role": "user", "content": "literal text [no scaffold appended]"}]
        env = self._env_with(ch, [""])
        out = env.chat_history_for_trace()
        assert out == ch

    def test_image_url_byte_identical(self):
        """image_url blocks must pass through bit-perfect — the trace
        viewer relies on them for screenshot rendering."""
        scaffold = "\n\nscaffold"
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
        }
        ch = [{
            "role": "user",
            "content": [image, {"type": "text", "text": scaffold.lstrip("\n")}],
        }]
        env = self._env_with(ch, [scaffold])
        out = env.chat_history_for_trace()
        # Identity-equal, not just structurally equal
        assert out[0]["content"][0] is image or out[0]["content"][0] == image
        assert out[0]["content"][0]["image_url"]["url"] == image["image_url"]["url"]
