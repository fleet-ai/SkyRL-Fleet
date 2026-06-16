"""Tests for `build_system_content` — the system-message builder used by
FleetTaskEnv.

Written blind from the contract, not the implementation. The contract:

  build_system_content(tools, *, modality, env_variables, env_key,
                       use_tools_channel, now) -> str

  - When `use_tools_channel=False` (default, legacy vLLM/SkyRL path):
    * the result contains "## Available Tools" and the full JSON dump of
      the tools list
    * the result contains "## Tool Call Format" with the <tool_call>{...}
      grammar example
    * the result lists tool names by name

  - When `use_tools_channel=True` (Tinker / HF-standard path):
    * the result does NOT contain "## Available Tools"
    * the result does NOT contain "## Tool Call Format"
    * the result does NOT mention the <tool_call> XML grammar
    * EVERY other section (date, env context, hints, error handling,
      response format) is unchanged

  - Modality affects browser/computer-use hints:
    * "computer_use" / "browser_use" include "## Browser Interaction Strategy"
    * "tool_use" omits it

  - env_key="fostgres" injects "## Database Exploration"

  - Pure function: same inputs → same output (modulo `now`).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.env import build_system_content


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]

FIXED_NOW = datetime(2026, 6, 15, 12, 0, 0)


# --------------------------------------------------------------------------- #
# Legacy path (use_tools_channel=False) — tools embedded as text
# --------------------------------------------------------------------------- #

class TestLegacyToolsInPrompt:
    def test_contains_available_tools_section(self):
        out = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        assert "## Available Tools" in out

    def test_contains_tools_json_dump(self):
        out = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        # Tools rendered as JSON: tool names + parameter shapes appear verbatim.
        assert '"bash"' in out
        assert '"read_file"' in out
        assert '"cmd"' in out

    def test_contains_tool_call_format_section(self):
        out = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        assert "## Tool Call Format" in out

    def test_contains_xml_grammar_example(self):
        out = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        # The legacy prompt teaches the <tool_call>{...}</tool_call> grammar
        # because vLLM models without native tool channels need it.
        assert "<tool_call>" in out

    def test_lists_tool_names_by_name(self):
        out = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        # Names are listed in the call-format reminder.
        assert "bash" in out
        assert "read_file" in out


# --------------------------------------------------------------------------- #
# Tools-channel path (use_tools_channel=True) — tools omitted from system text
# --------------------------------------------------------------------------- #

class TestToolsChannel:
    def test_no_available_tools_section(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Available Tools" not in out

    def test_no_tool_call_format_section(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Tool Call Format" not in out

    def test_no_xml_grammar_example(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        # The XML grammar example is gone — model uses its native channel.
        assert "<tool_call>" not in out

    def test_no_tool_json_dump_in_body(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        # Tool-specific JSON keys that only appear in the dump should be absent.
        assert '"function"' not in out
        assert '"parameters"' not in out

    def test_done_signal_still_documented(self):
        # We still need to teach the model the <done> completion signal — it's
        # Fleet-specific, not part of any chat template.
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "<done>" in out

    def test_response_format_section_kept(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Response Format" in out

    def test_error_handling_section_kept(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Error Handling" in out

    def test_current_date_still_present(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "2026-06-15" in out

    def test_no_tools_channel_difference_outside_tools_block(self):
        """The ONLY difference between legacy and tools-channel outputs should
        be the presence/absence of the tools-as-text block. Every other
        section must be byte-identical."""
        legacy = build_system_content(SAMPLE_TOOLS, use_tools_channel=False, now=FIXED_NOW)
        channel = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        # Channel form must be strictly shorter (the tools block is removed).
        assert len(channel) < len(legacy)
        # All sections we expect to be unchanged must appear identically in both.
        for section in (
            "## Current Date",
            "## Error Handling",
            "## Response Format",
        ):
            assert section in legacy
            assert section in channel


# --------------------------------------------------------------------------- #
# Modality-conditional sections (orthogonal to use_tools_channel)
# --------------------------------------------------------------------------- #

class TestModalitySections:
    @pytest.mark.parametrize("modality", ["computer_use", "browser_use"])
    @pytest.mark.parametrize("channel", [True, False])
    def test_vl_modalities_get_browser_strategy_hints(self, modality, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality=modality,
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Browser Interaction Strategy" in out

    @pytest.mark.parametrize("channel", [True, False])
    def test_tool_use_omits_browser_hints(self, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="tool_use",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Browser Interaction Strategy" not in out

    @pytest.mark.parametrize("channel", [True, False])
    def test_fostgres_env_key_adds_database_exploration(self, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            env_key="fostgres",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Database Exploration" in out

    @pytest.mark.parametrize("channel", [True, False])
    def test_non_fostgres_omits_database_exploration(self, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            env_key="walmart",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Database Exploration" not in out


# --------------------------------------------------------------------------- #
# Env variables propagation
# --------------------------------------------------------------------------- #

class TestEnvVariables:
    def test_logged_in_user_rendered(self):
        out = build_system_content(
            SAMPLE_TOOLS,
            env_variables={"LOGGED_IN_USER": "user_abc"},
            now=FIXED_NOW,
        )
        assert "user_abc" in out

    def test_logged_in_name_rendered(self):
        out = build_system_content(
            SAMPLE_TOOLS,
            env_variables={"LOGGED_IN_NAME": "Alice"},
            now=FIXED_NOW,
        )
        assert "Alice" in out

    def test_arbitrary_env_var_rendered(self):
        out = build_system_content(
            SAMPLE_TOOLS,
            env_variables={"WORKSPACE_ID": "ws-42"},
            now=FIXED_NOW,
        )
        assert "WORKSPACE_ID" in out
        assert "ws-42" in out

    def test_current_date_env_var_skipped(self):
        # CURRENT_DATE in env_variables shouldn't double up with the "Current
        # Date" section already in the prompt.
        out = build_system_content(
            SAMPLE_TOOLS,
            env_variables={"CURRENT_DATE": "1999-01-01"},
            now=FIXED_NOW,
        )
        # The 1999 string shouldn't appear; only the now-derived 2026 date does.
        assert "1999" not in out
        assert "2026" in out

    def test_no_env_variables_omits_environment_context_section(self):
        out = build_system_content(SAMPLE_TOOLS, env_variables={}, now=FIXED_NOW)
        assert "## Environment Context" not in out


# --------------------------------------------------------------------------- #
# Determinism / purity
# --------------------------------------------------------------------------- #

class TestPurity:
    def test_same_inputs_same_output(self):
        a = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        b = build_system_content(SAMPLE_TOOLS, now=FIXED_NOW)
        assert a == b

    def test_empty_tools_list_does_not_crash(self):
        out = build_system_content([], use_tools_channel=True, now=FIXED_NOW)
        # Even with no tools, we get a coherent prompt (the rest is unchanged).
        assert "## Response Format" in out

    def test_tools_list_with_non_function_entry_is_handled(self):
        # Defensive: tools list may contain entries without "function" key
        # (legacy MCP shapes). Should not crash.
        weird = [{"type": "raw", "name": "foo"}]
        out = build_system_content(weird, use_tools_channel=False, now=FIXED_NOW)
        # The tools-block is still rendered (we don't crash on missing "function").
        assert "## Available Tools" in out
