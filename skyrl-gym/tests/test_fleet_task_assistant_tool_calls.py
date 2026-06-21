"""Tests for step_async attaching OpenAI-shaped tool_calls to the
assistant message after a successful parse.

Production gap this closes: the Fleet trace viewer keys off
assistant.tool_calls[0].id matching the next tool message's tool_call_id
to link a screenshot back to the call that produced it. Before this
change, env.py only set {"role": "assistant", "content": action} — no
tool_calls field — so trace.py's pending_tool_call_id stayed None and
every tool message got a synthesized id matching nothing, resulting in
Tool Calls=0 and no top-of-session screenshot preview.

Verified by an A/B dummy-job probe (variants A vs E, identical content,
only difference being assistant.tool_calls presence): the version with
tool_calls renders the screenshot top-level; the version without doesn't.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.env import FleetTaskEnv


def _make_env(max_turns: int = 64, modality: str = "tool_use"):
    env_config = OmegaConf.create({"tasks_file": "/dev/null", "ttl_seconds": 7200})
    env = FleetTaskEnv.__new__(FleetTaskEnv)
    env.env_config = env_config
    env.extras = {
        "task_key": "t",
        "max_turns": max_turns,
        "use_tools_channel": True,
        "model_family": "qwen",
    }
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


class TestAssistantToolCallsAttached:
    @pytest.mark.asyncio
    async def test_successful_parse_attaches_openai_shape_tool_calls(self):
        """Parser succeeds → assistant_msg in chat_history gets a
        tool_calls list with OpenAI shape: id, type='function',
        function.name, function.arguments as JSON string."""
        env = _make_env()
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 0.0, False, {})
        )
        await env.step_async(
            '<tool_call>{"name":"bash","arguments":{"command":"ls -la"}}</tool_call>'
        )
        # Last assistant message (turn 1) is at chat_history[0] since
        # _make_env starts with empty chat_history.
        asst = next(m for m in env.chat_history if m.get("role") == "assistant")
        assert "tool_calls" in asst
        assert len(asst["tool_calls"]) == 1
        tc = asst["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        # arguments is JSON string, not dict — matches OpenAI spec and
        # what trace.py forwards downstream.
        assert tc["function"]["arguments"] == json.dumps({"command": "ls -la"})

    @pytest.mark.asyncio
    async def test_failed_parse_no_tool_calls_attached(self):
        """Parser returns None → assistant message must NOT have a
        tool_calls field. A bogus empty tool_calls list would still
        confuse the viewer's linkage."""
        env = _make_env()
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 0.0, False, {})
        )
        await env.step_async("plain text response, no tool call markup at all")
        asst = next(m for m in env.chat_history if m.get("role") == "assistant")
        assert "tool_calls" not in asst

    @pytest.mark.asyncio
    async def test_tool_call_id_uses_turn_counter(self):
        """Multi-turn: each successful assistant call gets `call_{turn}`
        as its tool_calls[0].id. Stable across turns so trace.py's
        pending_tool_call_id correctly carries the latest forward."""
        env = _make_env()
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 0.0, False, {})
        )
        for n in range(1, 4):
            await env.step_async(
                f'<tool_call>{{"name":"bash","arguments":{{"command":"step-{n}"}}}}</tool_call>'
            )
        asst_msgs = [m for m in env.chat_history if m.get("role") == "assistant"]
        assert len(asst_msgs) == 3
        assert [m["tool_calls"][0]["id"] for m in asst_msgs] == ["call_1", "call_2", "call_3"]

    @pytest.mark.asyncio
    async def test_done_signal_call_dropped_no_tool_calls_attached(self):
        """If the model wraps `done` in a computer call (VL pattern), the
        env converts tool_call → None and treats it as agent_done. No
        tool_calls field should be attached since there's no real call."""
        env = _make_env(modality="computer_use")
        env.screen_width = 1366
        env.screen_height = 768
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 1.0, True, {})
        )
        await env.step_async(
            '<tool_call>{"name":"computer","arguments":{"action":"done"}}</tool_call>'
        )
        asst = next(m for m in env.chat_history if m.get("role") == "assistant")
        assert "tool_calls" not in asst

    @pytest.mark.asyncio
    async def test_kimi_canonical_parses_into_tool_calls(self):
        """Real-world Kimi-K2.6 canonical-shape emission gets parsed and
        attached. Same regression that hides screenshots in production
        when this PR isn't applied."""
        env = _make_env()
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 0.0, False, {})
        )
        kimi = (
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0"
            "<|tool_call_argument_begin|>{\"command\":\"pwd\"}"
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        await env.step_async(kimi)
        asst = next(m for m in env.chat_history if m.get("role") == "assistant")
        assert "tool_calls" in asst
        assert asst["tool_calls"][0]["function"]["name"] == "bash"
        assert asst["tool_calls"][0]["function"]["arguments"] == json.dumps({"command": "pwd"})

    @pytest.mark.asyncio
    async def test_arguments_serialized_as_json_string(self):
        """OpenAI spec requires function.arguments to be a JSON-encoded
        STRING, not a dict. trace.py forwards verbatim; downstream
        viewers and any OpenAI-compatible consumer expect a string."""
        env = _make_env()
        env.openenv_task_env.step_async = AsyncMock(
            return_value=({"observation": "ok"}, 0.0, False, {})
        )
        await env.step_async(
            '<tool_call>{"name":"bash","arguments":{"a":1,"b":[2,3],"c":null}}</tool_call>'
        )
        asst = next(m for m in env.chat_history if m.get("role") == "assistant")
        args = asst["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        # round-trip back to verify shape is preserved
        assert json.loads(args) == {"a": 1, "b": [2, 3], "c": None}
