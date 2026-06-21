"""Tests for the per-turn footer, force-verifier, and tool-output truncation
behavior in FleetTaskEnv.step_async.

Written blind from the spec:

  - Every observation message MUST end with `[Turn N/MAX]` (where N is the
    current turn count post-step). This gives the model continuous pacing
    context the way the skyrl harness does, replacing the earlier threshold
    nudge ("emit <done> NOW or reward 0") that the canonical-run workflow
    analysis tied to 98/271 bare-<done> surrenders.

  - The text-only branch appends the footer to the trailing observation string.
    The multimodal branch (computer_use / browser_use) appends it as a final
    `{"type": "text"}` block — never overwriting image_url blocks.

  - The footer MUST appear at every turn, not just near the cap.

  - When `max_turns_reached`: the call to `openenv_task_env.step_async` MUST
    receive `done=True` even if the model never emitted `<done>`. This
    triggers the verifier so a non-zero reward is possible on the final turn.

  - Tool outputs > MAX_TOOL_OUTPUT_CHARS are truncated with a clear marker
    even when `context_manager` is None.

Mocks OpenEnv's FleetTaskEnv so step_async is testable without a live env.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.env import (
    FleetTaskEnv,
    MAX_TOOL_OUTPUT_CHARS,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bash_call_action() -> str:
    """A model-style assistant message with a valid <tool_call> for bash."""
    return '<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>'


def _make_env(max_turns: int = 50, modality: str = "tool_use") -> FleetTaskEnv:
    """Build a FleetTaskEnv with the bare minimum so step_async is callable.

    No real OpenEnv connection is made; openenv_task_env is replaced with a
    Mock that returns whatever the test wants.
    """
    env_config = OmegaConf.create({"tasks_file": "/dev/null", "ttl_seconds": 7200})
    extras = {"task_key": "dummy_task", "max_turns": max_turns, "use_tools_channel": True}
    # We can't run __init__ end-to-end (it loads tasks from JSON); skip it.
    env = FleetTaskEnv.__new__(FleetTaskEnv)
    env.env_config = env_config
    env.extras = extras
    env.max_turns = max_turns
    env.task_key = "dummy_task"
    env.api_key = "test-key"
    env.ttl_seconds = 7200
    env.partial_reward = False
    env.enable_hints = False
    env.openenv_task_env = MagicMock()
    # MagicMock auto-creates `_done` as a truthy Mock by default, which trips
    # the "episode already finished" short-circuit at the top of step_async.
    # Force it to False so the step actually runs.
    env.openenv_task_env._done = False
    env.chat_history = []
    env._scaffold_per_msg = []
    env.turns = 0
    env.tool_calls = 0
    env.tool_errors = 0
    env.last_reward = None
    env.tools = [{"type": "function", "function": {"name": "bash"}}]
    env._verifier_stdout = None
    env._verifier_error = None
    env._tool_error_messages = []
    env.context_manager = None
    env.enable_context_tools = False
    env.task_config = {"env_key": "data-eng", "task_modality": modality}
    return env


async def _step_and_get_obs(env: FleetTaskEnv, mock_step_return, action: str = None) -> tuple:
    """Run step_async with a mocked openenv_task_env response.

    Returns (step_output, openenv_call_kwargs).
    """
    env.openenv_task_env.step_async = AsyncMock(return_value=mock_step_return)
    out = await env.step_async(action or _bash_call_action())
    call = env.openenv_task_env.step_async.call_args
    return out, call


# --------------------------------------------------------------------------- #
# 1. Per-turn footer — text-only branch
# --------------------------------------------------------------------------- #

# Per-turn scaffold is now per-family in fleet_task.yaml; tests must opt
# into a family. Qwen's YAML has the turn indicator (and only the turn
# indicator), so it produces the literal "[Turn N/M]" trailing footer.
def _with_family(env, family: str):
    env.extras = dict(env.extras or {})
    env.extras["model_family"] = family
    return env


class TestTurnFooterTextOnly:
    @pytest.mark.asyncio
    async def test_footer_present_far_from_cap(self):
        env = _with_family(_make_env(max_turns=50), "qwen")
        env.turns = 10  # step takes it to 11
        step_ret = ({"observation": "ls\nfile.txt"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert body.rstrip().endswith("[Turn 11/50]")

    @pytest.mark.asyncio
    async def test_footer_present_near_cap(self):
        env = _with_family(_make_env(max_turns=50), "qwen")
        env.turns = 46  # step takes it to 47
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert body.rstrip().endswith("[Turn 47/50]")

    @pytest.mark.asyncio
    async def test_footer_present_first_turn(self):
        env = _with_family(_make_env(max_turns=64), "qwen")
        env.turns = 0  # step takes it to 1
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert body.rstrip().endswith("[Turn 1/64]")

    @pytest.mark.asyncio
    async def test_no_footer_when_model_family_unset(self):
        """When model_family is missing from extras (no YAML family match)
        the env appends no per-turn scaffold. Pinned to surface the
        migration gap: SkyRL's Qwen generator currently does NOT plumb
        model_family into env_extras, so production Qwen runs land here
        and silently lose the turn indicator until SkyRL is updated."""
        env = _make_env(max_turns=50)  # no model_family
        env.turns = 10
        step_ret = ({"observation": "ls\nfile.txt"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert "[Turn" not in body

    @pytest.mark.asyncio
    async def test_no_footer_after_max_turns_reached(self):
        """When max_turns is reached the env returns the empty-obs branch
        before any user-facing obs is built — footer doesn't apply."""
        env = _make_env(max_turns=50)
        env.turns = 49  # next step takes it to 50 → max_turns_reached
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        assert out["observations"] == []
        assert out["done"] is True


# --------------------------------------------------------------------------- #
# 2. Per-turn footer — multimodal branch (computer_use / browser_use)
# --------------------------------------------------------------------------- #

class TestTurnFooterMultimodal:
    @pytest.mark.asyncio
    async def test_footer_appended_as_trailing_text_block(self):
        env = _with_family(_make_env(max_turns=64), "qwen")
        env.turns = 20  # step takes it to 21
        mm_content = [
            {"type": "text", "text": "Saw screen"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]
        step_ret = ({"observation": mm_content}, 0.0, False, {})
        with patch(
            "skyrl_gym.envs.fleet_task.env.tool_result_to_message_content",
            return_value=mm_content,
        ):
            out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert isinstance(body, list)
        # image_url block preserved
        assert any(b.get("type") == "image_url" for b in body)
        # trailing text block contains the footer
        text_blocks = [b for b in body if b.get("type") == "text"]
        last_text = text_blocks[-1]["text"]
        assert last_text == "[Turn 21/64]"

    @pytest.mark.asyncio
    async def test_footer_present_in_multimodal_every_turn(self):
        env = _with_family(_make_env(max_turns=64), "qwen")
        env.turns = 0  # step takes it to 1
        mm_content = [{"type": "image_url", "image_url": {"url": "..."}}]
        step_ret = ({"observation": mm_content}, 0.0, False, {})
        with patch(
            "skyrl_gym.envs.fleet_task.env.tool_result_to_message_content",
            return_value=mm_content,
        ):
            out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        text_blocks = [b for b in body if b.get("type") == "text"]
        # at least one text block, and the last one is the footer
        assert text_blocks
        assert text_blocks[-1]["text"] == "[Turn 1/64]"


# --------------------------------------------------------------------------- #
# 3. No-tool-call hint is format-agnostic, covers truncation in one line
# --------------------------------------------------------------------------- #

class TestNoToolCallHint:
    @pytest.mark.asyncio
    async def test_no_tool_call_hint_is_format_agnostic(self):
        """Without model_family in extras, the env falls back to a
        generic hint — no per-family canonical shape included.

        The previous version of this test pinned the literal string
        'cut off' from the old reject message. That message ("If your
        response was cut off at the per-turn token limit, be more
        concise") was actively misleading the model: across 14 BU
        sessions in job c4b429ae, responses were well under the
        generation limit but the model believed the truncation hint
        and spiralled trying to be "more concise" — 55 such turns
        observed. Fixed by removing the truncation framing entirely.
        """
        env = _make_env(max_turns=50)
        env.turns = 5
        env.openenv_task_env.step_async = AsyncMock(return_value=({"observation": "ok"}, 0.0, False, {}))
        out = await env.step_async("plain reasoning with no tool call")
        body = out["observations"][0]["content"]
        assert "No tool call landed" in body
        # The misleading "cut off" framing is GONE. Pinning its absence.
        assert "cut off" not in body
        # Must not prescribe Qwen syntax (wrong for Kimi/Qwen3+ native).
        assert "<tool_call>" not in body

    @pytest.mark.asyncio
    async def test_no_tool_call_hint_includes_kimi_canonical(self):
        """When model_family='kimi' is in extras, the reject message
        must echo the canonical Kimi tool-call shape so the special
        tokens land in the model's next-turn context as anchors."""
        env = _make_env(max_turns=50)
        env.turns = 5
        env.extras = dict(env.extras or {})
        env.extras["model_family"] = "kimi"
        env.openenv_task_env.step_async = AsyncMock(return_value=({"observation": "ok"}, 0.0, False, {}))
        out = await env.step_async("plain reasoning with no tool call")
        body = out["observations"][0]["content"]
        assert "No tool call landed" in body
        # The literal Kimi special-token markers must appear in the
        # body — the tokenizer encodes these as single special-token
        # IDs when this message is rendered into the next prompt,
        # putting the right IDs in the model's context to copy.
        assert "<|tool_call_begin|>" in body
        assert "<|tool_call_argument_begin|>" in body
        assert "<|tool_call_end|>" in body

    @pytest.mark.asyncio
    async def test_no_tool_call_hint_includes_qwen_canonical(self):
        """When model_family='qwen' is in extras, the reject message
        prescribes Qwen's text-based grammar instead."""
        env = _make_env(max_turns=50)
        env.turns = 5
        env.extras = dict(env.extras or {})
        env.extras["model_family"] = "qwen"
        env.openenv_task_env.step_async = AsyncMock(return_value=({"observation": "ok"}, 0.0, False, {}))
        out = await env.step_async("plain reasoning with no tool call")
        body = out["observations"][0]["content"]
        assert "No tool call landed" in body
        assert "<tool_call>" in body
        assert "</tool_call>" in body


# --------------------------------------------------------------------------- #
# 4. Force-verifier on max_turns
# --------------------------------------------------------------------------- #

class TestForceDoneAtMaxTurns:
    @pytest.mark.asyncio
    async def test_done_true_sent_on_last_turn_with_tool_call(self):
        """When the model is on its final turn AND made a tool call, the
        openenv step MUST receive done=True even if no <done> was emitted."""
        env = _make_env(max_turns=50)
        env.turns = 49  # step takes it to 50
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        _, call = await _step_and_get_obs(env, step_ret)
        sent_action = call.args[0]
        assert sent_action.get("done") is True, f"expected done=True, got {sent_action}"
        assert sent_action.get("tool") == "bash"

    @pytest.mark.asyncio
    async def test_done_false_when_far_from_cap_and_no_explicit_done(self):
        env = _make_env(max_turns=50)
        env.turns = 5
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        _, call = await _step_and_get_obs(env, step_ret)
        sent_action = call.args[0]
        assert sent_action.get("done") is False

    @pytest.mark.asyncio
    async def test_done_true_on_last_turn_with_no_tool_call(self):
        """Model emits no tool call on its final turn — env still must trigger
        the verifier via the agent-done-without-tool-call branch."""
        env = _make_env(max_turns=50)
        env.turns = 49
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        env.openenv_task_env.step_async = AsyncMock(return_value=step_ret)
        # Action with NO tool call and NO <done>
        await env.step_async("just some reasoning, no tool call")
        # Verify openenv was called at all with done=True
        call = env.openenv_task_env.step_async.call_args
        assert call is not None, "openenv step_async was not called on the final turn"
        sent_action = call.args[0]
        assert sent_action.get("done") is True

    @pytest.mark.asyncio
    async def test_agent_done_sent_when_model_emits_done(self):
        env = _make_env(max_turns=50)
        env.turns = 10  # not near cap
        step_ret = ({"observation": "ok"}, 1.0, True, {})
        env.openenv_task_env.step_async = AsyncMock(return_value=step_ret)
        await env.step_async("Here is my answer: 42. <done>")
        call = env.openenv_task_env.step_async.call_args
        # Agent done WITHOUT a tool call → openenv sees just {"done": True}
        assert call.args[0].get("done") is True


# --------------------------------------------------------------------------- #
# 4. Tool-output truncation
# --------------------------------------------------------------------------- #

class TestToolOutputTruncation:
    @pytest.mark.asyncio
    async def test_huge_tool_output_truncated_with_marker(self):
        env = _make_env(max_turns=50)
        env.turns = 5
        env.context_manager = None  # always-on truncation should fire
        huge = "A" * (MAX_TOOL_OUTPUT_CHARS * 5)
        step_ret = ({"observation": huge}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert isinstance(body, str)
        assert len(body) < len(huge), "tool result should be shortened"
        assert "TRUNCATED" in body
        assert "chars elided" in body or "chars total" in body  # accept either marker

    @pytest.mark.asyncio
    async def test_short_tool_output_unmodified(self):
        env = _make_env(max_turns=50)
        env.turns = 5
        env.context_manager = None
        short = "small output\nline 2"
        step_ret = ({"observation": short}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert short in body
        assert "TRUNCATED" not in body

    @pytest.mark.asyncio
    async def test_truncation_does_not_apply_to_non_strings(self):
        """Multimodal content (list) is not a candidate for char-based
        truncation. It goes through the multimodal branch unchanged."""
        env = _make_env(max_turns=50)
        env.turns = 5
        env.context_manager = None
        mm_content = [{"type": "image_url", "image_url": {"url": "..."}}]
        step_ret = ({"observation": mm_content}, 0.0, False, {})
        with patch(
            "skyrl_gym.envs.fleet_task.env.tool_result_to_message_content",
            return_value=mm_content,
        ):
            out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert isinstance(body, list)
        assert any(b.get("type") == "image_url" for b in body)
