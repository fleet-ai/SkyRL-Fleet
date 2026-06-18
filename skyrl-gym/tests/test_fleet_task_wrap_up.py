"""Tests for the wrap-up nudge, force-verifier, and tool-output truncation
behavior in FleetTaskEnv.step_async.

Written blind from the spec:

  - The wrap-up nudge fires EXACTLY ONCE per episode, when
    `remaining = max_turns - turns` equals `WRAP_UP_NUDGE_AT` (3).
    Earlier shipping fired every turn for the last 5 with surrender-flavored
    text ("emit <done> NOW or you get reward 0"), which the canonical-run
    workflow analysis tied to 98/271 immediate bare-<done> surrenders.

  - When remaining != WRAP_UP_NUDGE_AT, the nudge MUST NOT appear, even if
    the episode is still within the old "5 turns left" window.

  - The text MUST tell the model to output a final answer and NOT surrender
    without one. It MUST NOT include "or you get reward 0".

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
    WRAP_UP_NUDGE_AT,
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
    env.turns = 0
    env.tool_calls = 0
    env.tool_errors = 0
    env.last_reward = None
    env.tools = [{"type": "function", "function": {"name": "bash"}}]
    env._verifier_stdout = None
    env._verifier_error = None
    env._tool_error_messages = []
    env._wrap_up_nudge_fired = False
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
# 1. Wrap-up nudge — text-only branch
# --------------------------------------------------------------------------- #

class TestWrapUpNudgeTextOnly:
    @pytest.mark.asyncio
    async def test_no_nudge_far_from_cap(self):
        env = _make_env(max_turns=50)
        env.turns = 10  # remaining = 40, well above the trigger point
        step_ret = ({"observation": "ls\nfile.txt"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert "turns left" not in body
        assert "Do NOT surrender" not in body

    @pytest.mark.asyncio
    async def test_nudge_fires_exactly_at_remaining_equals_WRAP_UP_NUDGE_AT(self):
        env = _make_env(max_turns=50)
        # After step increments env.turns from turns_in to turns_in+1, we want
        # max_turns - (turns_in+1) == WRAP_UP_NUDGE_AT.
        env.turns = 50 - WRAP_UP_NUDGE_AT - 1
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert f"{WRAP_UP_NUDGE_AT} turns left" in body
        assert "Do NOT surrender" in body
        # The discarded "or you get reward 0" framing must not return.
        assert "reward 0" not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize("remaining_after_step", [10, 5, 4, 2, 1])
    async def test_no_nudge_when_remaining_not_equal_to_threshold(self, remaining_after_step):
        # The old behavior fired the nudge whenever 0 < remaining <= 5. This
        # test pins that it now fires ONLY at remaining == WRAP_UP_NUDGE_AT.
        if remaining_after_step == WRAP_UP_NUDGE_AT:
            pytest.skip("covered by the on-threshold test")
        env = _make_env(max_turns=50)
        env.turns = 50 - remaining_after_step - 1
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        assert "turns left" not in body, f"unexpected nudge at remaining={remaining_after_step}"

    @pytest.mark.asyncio
    async def test_nudge_is_idempotent_per_episode(self):
        # Once fired, the nudge must not re-fire on a later step in the same
        # episode (e.g. the next turn after a model "ok let me try" reply).
        env = _make_env(max_turns=50)
        env.turns = 50 - WRAP_UP_NUDGE_AT - 1
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out_first, _ = await _step_and_get_obs(env, step_ret)
        assert f"{WRAP_UP_NUDGE_AT} turns left" in out_first["observations"][0]["content"]
        # Take another step; remaining drops to WRAP_UP_NUDGE_AT - 1.
        env.openenv_task_env.step_async = AsyncMock(return_value=step_ret)
        out_second = await env.step_async(_bash_call_action())
        assert "turns left" not in out_second["observations"][0]["content"]

    @pytest.mark.asyncio
    async def test_no_nudge_after_max_turns_reached(self):
        """When max_turns is reached the env returns the empty-obs branch
        before any user-facing obs is built — no nudge slot exists."""
        env = _make_env(max_turns=50)
        env.turns = 49  # next step takes it to 50 → max_turns_reached
        step_ret = ({"observation": "ok"}, 0.0, False, {})
        out, _ = await _step_and_get_obs(env, step_ret)
        # max_turns return path has observations=[]
        assert out["observations"] == []
        assert out["done"] is True


# --------------------------------------------------------------------------- #
# 2. Wrap-up nudge — multimodal branch (computer_use / browser_use)
# --------------------------------------------------------------------------- #

class TestWrapUpNudgeMultimodal:
    @pytest.mark.asyncio
    async def test_nudge_appended_as_text_block_at_threshold(self):
        env = _make_env(max_turns=50)
        env.turns = 50 - WRAP_UP_NUDGE_AT - 1
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
        assert any(b.get("type") == "image_url" for b in body)
        text_blocks = [b for b in body if b.get("type") == "text"]
        last_text = text_blocks[-1]["text"]
        assert f"{WRAP_UP_NUDGE_AT} turns left" in last_text
        assert "Do NOT surrender" in last_text

    @pytest.mark.asyncio
    async def test_no_nudge_in_multimodal_when_not_at_threshold(self):
        env = _make_env(max_turns=50)
        env.turns = 10
        mm_content = [{"type": "image_url", "image_url": {"url": "..."}}]
        step_ret = ({"observation": mm_content}, 0.0, False, {})
        with patch(
            "skyrl_gym.envs.fleet_task.env.tool_result_to_message_content",
            return_value=mm_content,
        ):
            out, _ = await _step_and_get_obs(env, step_ret)
        body = out["observations"][0]["content"]
        text_blocks = [b for b in body if b.get("type") == "text"]
        for b in text_blocks:
            assert "turns left" not in b.get("text", "")


# --------------------------------------------------------------------------- #
# 3. Force-verifier on max_turns
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
