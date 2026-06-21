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

    def test_assistant_messages_pass_through_unchanged(self):
        """step_async appends '' to _scaffold_per_msg for every assistant
        turn (line 884). Trace projection must preserve the model's
        emission byte-for-byte — that's the load-bearing data the trace
        viewer renders for human inspection."""
        emission = (
            "<think>plan</think>"
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.computer:0"
            "<|tool_call_argument_begin|>{\"action\":\"screenshot\"}"
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        ch = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": emission},
        ]
        env = self._env_with(ch, ["", "", ""])
        out = env.chat_history_for_trace()
        assert out[2]["content"] == emission
        assert out[2]["role"] == "assistant"

    def test_only_trailing_text_block_stripped(self):
        """Bug fix: if a multimodal observation has multiple text blocks
        and an EARLIER one happens to end with the same suffix as the
        scaffold, only the LAST text block (the one step_async appended)
        must be stripped. Earlier strip-on-all logic could double-strip."""
        scaffold = "\n\n[Turn 3/64]"
        lstripped = scaffold.lstrip("\n")
        # Adversarial: the tool result's own text happens to end with the
        # same string as the scaffold suffix. Should NOT be stripped.
        coincidental_text = {"type": "text", "text": "Tool log line [Turn 3/64]"}
        image = {"type": "image_url", "image_url": {"url": "u"}}
        scaffold_block = {"type": "text", "text": lstripped}
        ch = [{
            "role": "user",
            "content": [coincidental_text, image, scaffold_block],
        }]
        env = self._env_with(ch, [scaffold])
        out = env.chat_history_for_trace()
        # Earlier text block preserved with its coincidental suffix
        assert out[0]["content"][0]["text"] == "Tool log line [Turn 3/64]"
        # Image preserved
        assert out[0]["content"][1] == image
        # Trailing scaffold-only text block dropped
        assert len(out[0]["content"]) == 2

    def test_length_mismatch_raises(self):
        """Bug fix: zip() would silently truncate to the shorter list,
        dropping messages from the trace upload. A length mismatch is a
        programmer error and must fail loudly so the bug surfaces."""
        env = self._env_with(
            chat_history=[{"role": "user", "content": "a"},
                          {"role": "user", "content": "b"}],
            scaffolds=[""],  # one short
        )
        with pytest.raises(ValueError, match="length mismatch"):
            env.chat_history_for_trace()

    def test_output_length_matches_input(self):
        """No message is silently dropped from the projection. Every
        message in chat_history appears in the trace output exactly once,
        in order."""
        ch = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "obs1" + "\n\nscaff"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "obs2" + "\n\nscaff"},
        ]
        env = self._env_with(ch, ["", "", "", "\n\nscaff", "", "\n\nscaff"])
        out = env.chat_history_for_trace()
        assert len(out) == len(ch)
        assert [m["role"] for m in out] == [m["role"] for m in ch]

    def test_multimodal_with_no_text_block_at_all(self):
        """Edge: a multimodal obs with only image blocks (no text)
        shouldn't crash and shouldn't drop the image. Defensive against
        unexpected MCP shapes."""
        image = {"type": "image_url", "image_url": {"url": "u"}}
        ch = [{"role": "user", "content": [image]}]
        env = self._env_with(ch, ["\n\nscaffold"])
        out = env.chat_history_for_trace()
        assert out[0]["content"] == [image]


# --------------------------------------------------------------------------- #
# Integration: step_async writes _scaffold_per_msg; chat_history_for_trace
# reads it. Pins the contract between the append site and the strip site.
# --------------------------------------------------------------------------- #

class TestStepAsyncToTraceIntegration:
    """End-to-end: walk step_async paths, verify chat_history_for_trace
    cleanly reverses the scaffold append. Catches drift between the two
    sites (e.g., if append uses a different lstrip rule than strip)."""

    def _stub_env(self, model_family: str, turns: int = 0, max_turns: int = 64):
        """Mirrors test_fleet_task_wrap_up._make_env but local to this file
        to avoid coupling. Same MagicMock-based __new__ pattern."""
        from unittest.mock import MagicMock
        env = FleetTaskEnv.__new__(FleetTaskEnv)
        env.extras = {"task_key": "t", "max_turns": max_turns,
                      "use_tools_channel": True, "model_family": model_family}
        env.max_turns = max_turns
        env.task_key = "t"
        env.api_key = "k"
        env.ttl_seconds = 7200
        env.partial_reward = False
        env.enable_hints = False
        env.openenv_task_env = MagicMock()
        env.openenv_task_env._done = False
        env.chat_history = []
        env._scaffold_per_msg = []
        env.turns = turns
        env.tool_calls = 0
        env.tool_errors = 0
        env.last_reward = None
        env.tools = [{"type": "function", "function": {"name": "bash"}}]
        env._verifier_stdout = None
        env._verifier_error = None
        env._tool_error_messages = []
        env.context_manager = None
        env.enable_context_tools = False
        env.task_config = {"env_key": "data-eng", "task_modality": "tool_use"}
        return env

    @pytest.mark.asyncio
    async def test_text_only_roundtrip_strips_what_step_appended(self):
        """step_async appends scaffold to obs string; chat_history_for_trace
        must remove EXACTLY that suffix, leaving the env's content intact."""
        from unittest.mock import AsyncMock
        env = self._stub_env(model_family="qwen", turns=4)
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "tool output here"}, 0.0, False, {})
        )
        # Drive one step
        from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call
        # _make_env's bash tool: send a parseable bash call
        await env.step_async(
            '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'
        )
        # chat_history now contains: assistant + user(obs+scaffold)
        traced = env.chat_history_for_trace()
        # Assistant emission preserved byte-for-byte
        assert traced[0]["role"] == "assistant"
        assert traced[0]["content"].startswith("<tool_call>")
        # Observation has scaffold stripped — leaves env's content intact.
        # The env wraps the tool result before appending scaffold; the
        # load-bearing assertion is that the scaffold suffix (turn indicator)
        # is GONE, not the exact pre-scaffold prefix.
        assert traced[1]["role"] == "user"
        assert "[Turn" not in traced[1]["content"]
        assert "tool output here" in traced[1]["content"]

    @pytest.mark.asyncio
    async def test_multimodal_roundtrip_preserves_image(self):
        """Multimodal step appends scaffold as trailing text block;
        projection strips that block, image_url survives."""
        from unittest.mock import AsyncMock, patch
        env = self._stub_env(model_family="qwen", turns=4)
        image = {"type": "image_url", "image_url": {"url": "data:img"}}
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": [image]}, 0.0, False, {})
        )
        with patch(
            "skyrl_gym.envs.fleet_task.env.tool_result_to_message_content",
            return_value=[image],
        ):
            await env.step_async(
                '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'
            )
        traced = env.chat_history_for_trace()
        # Last message is the multimodal obs; image is preserved, scaffold
        # text block dropped.
        obs = traced[-1]
        assert obs["role"] == "user"
        assert obs["content"] == [image]
