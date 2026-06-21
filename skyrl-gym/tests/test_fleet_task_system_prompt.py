"""Tests for `build_system_content` — the system-message builder used by
FleetTaskEnv.

Written blind from the contract, not the implementation. The contract:

  build_system_content(tools, *, modality, env_variables, env_key,
                       use_tools_channel, now) -> str

  - The result ALWAYS contains "## Available Tools" + the full JSON dump of
    the tools list (both `use_tools_channel=True` and `False`). The dump is
    the canonical schema reference the model can fall back to when prose
    hints in the system prompt are ambiguous. See job 4746408e_4 root cause:
    Kimi hallucinated `keys: [...]` instead of `text: "alt+Left"` for the
    `key` action because the prose hint suggested the former and the
    schema dump was suppressed.

  - When `use_tools_channel=False` (default, legacy vLLM/SkyRL path):
    * additionally contains "## Tool Call Format" with the <tool_call>{...}
      grammar example
    * lists tool names in the call-format reminder

  - When `use_tools_channel=True` (Tinker / HF-standard path):
    * "## Available Tools" still present (schema is the canonical fallback)
    * "## Tool Call Format" omitted (model uses native tool channel —
      forcing it onto Qwen's <tool_call> syntax would push it OFF its
      native <|tool_call_begin|> channel)
    * the <tool_call> XML grammar example is absent
    * EVERY non-tools section (date, env context, hints, error handling,
      response format) is unchanged between the two paths

  - Modality affects browser/computer-use hints:
    * "computer_use" / "browser_use" include "## Browser Interaction Strategy"
    * "tool_use" omits it

  - Hint examples must be schema-compliant: every literal JSON-shape
    example in the BU/CU hints must round-trip through json.loads and match
    the MCP `computer` tool schema. The misleading `key("Alt","Left")`
    prose hint that tripped job 4746408e is forbidden by regression test.

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
    def test_available_tools_section_still_present(self):
        # New contract: the schema dump is the canonical reference and must
        # be present even when the model has a native tool channel. See
        # job 4746408e_4: Kimi-K2.6 hallucinated a `keys: [...]` arg shape
        # when this block was suppressed, because the prose hint was the
        # only concrete example of the `key` action it had.
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Available Tools" in out

    def test_tool_json_dump_still_present(self):
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        # The JSON dump keys that came from the tools list must appear.
        assert '"bash"' in out
        assert '"read_file"' in out
        assert '"cmd"' in out

    def test_no_tool_call_format_section(self):
        # Tool Call Format teaches Qwen's <tool_call>{...}</tool_call> syntax.
        # Forcing a Kimi or Qwen3+ model onto that syntax pushes it off its
        # native <|tool_call_begin|>/<tools> channel — bad.
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "## Tool Call Format" not in out

    def test_no_xml_grammar_example(self):
        # The XML grammar example is gone — model uses its native channel.
        out = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        assert "<tool_call>" not in out

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

    def test_tools_channel_only_drops_call_format_block(self):
        """The ONLY difference between legacy and tools-channel outputs is
        the presence/absence of the `## Tool Call Format` example block.
        The schema dump (`## Available Tools`) and every non-tools section
        is identical between paths."""
        legacy = build_system_content(SAMPLE_TOOLS, use_tools_channel=False, now=FIXED_NOW)
        channel = build_system_content(SAMPLE_TOOLS, use_tools_channel=True, now=FIXED_NOW)
        # Channel form must be strictly shorter (no Tool Call Format block).
        assert len(channel) < len(legacy)
        # Schema dump present in BOTH.
        assert "## Available Tools" in legacy
        assert "## Available Tools" in channel
        # Call-format block present in legacy, absent in channel.
        assert "## Tool Call Format" in legacy
        assert "## Tool Call Format" not in channel
        # All non-tools sections appear identically in both.
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
    @pytest.mark.parametrize("channel", [True, False])
    def test_browser_use_gets_browser_strategy_hints(self, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="browser_use",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Browser Interaction Strategy" in out

    @pytest.mark.parametrize("channel", [True, False])
    def test_computer_use_gets_desktop_strategy_hints_not_browser(self, channel):
        # CU is a Linux desktop with SaaS apps in tabs, not a single browser.
        # The previous shared "browser strategy" framing caused 13/44 CU
        # rollouts in the canonical run to type invented URLs.
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="computer_use",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Desktop Interaction Strategy" in out
        assert "Linux desktop" in out
        # The browser framing string must NOT be present for CU.
        assert "You are controlling a web browser" not in out
        assert "## Browser Interaction Strategy" not in out

    @pytest.mark.parametrize("channel", [True, False])
    def test_tool_use_omits_browser_and_desktop_hints(self, channel):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="tool_use",
            use_tools_channel=channel,
            now=FIXED_NOW,
        )
        assert "## Browser Interaction Strategy" not in out
        assert "## Desktop Interaction Strategy" not in out

    # ---- BU: portal URL injection ----

    def test_browser_use_with_portal_url_injects_it(self):
        # 47/48 BU rollouts in the canonical run wasted turn 1 on a guessed
        # hostname because the prompt never said where the browser actually
        # was. Inject the live portal URL when available.
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="browser_use",
            portal_url="https://vfpi1menejjz-n.env.fleet-prod-wi2-us-east-1.fleetai.com",
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        assert "vfpi1menejjz-n.env.fleet-prod-wi2-us-east-1.fleetai.com" in out

    def test_browser_use_without_portal_url_falls_back_generically(self):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="browser_use",
            portal_url=None,
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        # Generic instruction: "Stay on the current domain"
        assert "current domain" in out
        # And no invented hostname appears.
        assert "fleetai.com" not in out

    def test_browser_use_bans_localhost(self):
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="browser_use",
            portal_url="https://x.example.fleetai.com",
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        # The prompt explicitly forbids made-up localhost URLs.
        assert "localhost" in out  # appears in the DO NOT example
        assert "403" in out  # the consequence is documented

    # ---- CU: app alias map ----

    def test_computer_use_includes_app_alias_map(self):
        # CU rollouts confused Signal (Sentry), Kernel (Jira), Ledger
        # (QuickBooks), Latch (Outlook), Cadence (HR), Float/Ramp (expenses).
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="computer_use",
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        for in_env, saas in [
            ("Signal", "Sentry"),
            ("Kernel", "Jira"),
            ("Ledger", "QuickBooks"),
            ("Latch", "Outlook"),
            ("Cadence", "HR"),
        ]:
            assert in_env in out, f"alias map missing {in_env}"
            assert saas in out, f"alias map missing {saas}"

    def test_computer_use_warns_against_terminal_escape(self):
        # 21/44 CU rollouts escaped to sqlite terminal — verifier checks UI
        # state, so terminal-derived knowledge doesn't score.
        out = build_system_content(
            SAMPLE_TOOLS,
            modality="computer_use",
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        assert "terminal" in out.lower()
        assert "sqlite" in out.lower() or "SQLite" in out

    # ---- Both modalities: action vocabulary ----

    @pytest.mark.parametrize("modality", ["browser_use", "computer_use"])
    def test_action_vocabulary_enumerated(self, modality):
        # 20/92 BU+CU rollouts wasted turn 2 emitting action="click" and
        # hitting a Pydantic validation error. Enumerate the enum upfront.
        out = build_system_content(
            SAMPLE_TOOLS,
            modality=modality,
            portal_url="https://x.example.fleetai.com" if modality == "browser_use" else None,
            use_tools_channel=True,
            now=FIXED_NOW,
        )
        for value in (
            "left_click",
            "right_click",
            "double_click",
            "type",
            "key",
            "scroll",
            "wait",
            "screenshot",
            "left_click_drag",
            "hold_key",
        ):
            assert value in out, f"missing action enum value: {value}"
        # And calls out the most common mistake.
        assert "Use `left_click`, NOT `click`" in out

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
# Hint-example schema compliance (regression: job 4746408e_4)
# --------------------------------------------------------------------------- #

# Minimal fixture for the Fleet MCP `computer` tool schema. Mirrors
# Anthropic's computer_20250124 contract (taiga/.../computer.py:104).
# `key` and `type` actions require `text: string`; the schema has NO
# `keys` field. Used for round-tripping hint examples against the
# canonical shape Kimi sees through apply_chat_template(tools=...).
_COMPUTER_ACTIONS_WITH_TEXT_ARG = {"key", "type", "hold_key"}
_COMPUTER_ACTIONS_WITHOUT_KEYS_ARG = {
    "screenshot", "left_click", "right_click", "middle_click",
    "double_click", "triple_click", "type", "key", "scroll",
    "wait", "mouse_move", "left_click_drag", "cursor_position",
    "left_mouse_down", "left_mouse_up", "hold_key",
}


class TestHintSchemaCompliance:
    """Every literal `{...}`-shaped example in BU/CU hints must round-trip
    through json.loads and conform to the MCP `computer` tool schema.

    The misleading prose hint `key("Alt","Left")` in the previous version
    caused Kimi-K2.6 to emit `{"action":"key","keys":["Alt","Left"]}` —
    a `keys` field that does not exist on the tool schema. The MCP server
    silently returned a null screenshot, the model spiraled, two real BU
    sessions burned. Regression-pinned here.
    """

    def _bu_hint_text(self) -> str:
        return build_system_content(
            SAMPLE_TOOLS,
            modality="browser_use",
            portal_url="https://x.env.fleet-prod.fleetai.com",
            use_tools_channel=True,
            now=FIXED_NOW,
        )

    def _cu_hint_text(self) -> str:
        return build_system_content(
            SAMPLE_TOOLS,
            modality="computer_use",
            use_tools_channel=True,
            now=FIXED_NOW,
        )

    def test_bu_hint_never_suggests_keys_array_shape(self):
        """`keys: ["Alt","Left"]` is the exact hallucination Kimi made in
        session fa2c9e97. The hint must not push the model toward that
        shape — `keys` is not on the schema."""
        text = self._bu_hint_text()
        assert '"keys"' not in text, (
            "BU hint contains `\"keys\"` which is not on the MCP computer "
            "schema; the `key` action takes `text: string` (xdotool combo). "
            "See job 4746408e_4 root cause."
        )
        # The function-call sugar that originally caused the hallucination
        # must also be gone.
        assert 'key("Alt","Left")' not in text
        assert 'key("Alt", "Left")' not in text

    def test_bu_hint_teaches_text_arg_for_key_action(self):
        """The hint must direct the model to the `text` arg, not invent a
        `keys` arg. Either via prose or via a literal JSON example."""
        text = self._bu_hint_text()
        # Either a literal JSON example with action=key+text=alt+Left, or
        # prose that names the `text` arg + xdotool syntax. We accept any
        # of these to leave room for hint rephrasing without breaking
        # the test.
        accepts = (
            '"action": "key"' in text and '"text": "alt+Left"' in text,
            '"action":"key"' in text and '"text":"alt+Left"' in text,
            ('"text"' in text and 'alt+Left' in text),
        )
        assert any(accepts), (
            "BU hint must teach the schema-compliant `key` action shape: "
            '{"action": "key", "text": "alt+Left"}. Current hint suggests '
            "neither the `text` field nor xdotool combo syntax."
        )

    def test_bu_hint_json_examples_match_computer_schema(self):
        """Every literal JSON object in the BU hint that looks like a
        computer tool call must:
          - parse as JSON
          - have an `action` field whose value is on the action vocabulary
          - not introduce arg fields outside the known schema
        """
        text = self._bu_hint_text()
        examples = _extract_json_objects_containing(text, '"action"')
        assert examples, (
            "BU hint has no literal JSON `action` examples to validate; "
            "the model has nothing concrete to anchor on for `key`/`type`."
        )
        for ex in examples:
            assert "action" in ex, f"example missing action: {ex!r}"
            # action must be in the documented vocabulary
            assert ex["action"] in _COMPUTER_ACTIONS_WITHOUT_KEYS_ARG | {"navigate"}, (
                f"action {ex['action']!r} not in computer vocabulary"
            )
            # `keys` is not a real field on the schema.
            assert "keys" not in ex, (
                f"example uses `keys` arg which is not on schema: {ex!r}"
            )
            # key/type/hold_key actions must provide `text`.
            if ex["action"] in _COMPUTER_ACTIONS_WITH_TEXT_ARG:
                assert "text" in ex, (
                    f"action {ex['action']!r} requires `text`: {ex!r}"
                )
                assert isinstance(ex["text"], str), (
                    f"`text` must be a string for {ex['action']!r}: {ex!r}"
                )

    def test_cu_hint_does_not_reintroduce_keys_shape(self):
        # CU hint never had the bad example, but pin it so a future edit
        # that copies from BU doesn't bring the bug across.
        text = self._cu_hint_text()
        assert '"keys"' not in text
        assert 'key("Alt","Left")' not in text


def _extract_json_objects_containing(text: str, marker: str) -> list:
    """Pull `{...}` substrings out of `text` that contain `marker`, parse as
    JSON, return the parsed objects. Best-effort balanced-brace scan; skips
    substrings that don't json.loads cleanly so unrelated `{` in prose
    don't fail the test."""
    import json as _json
    results = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        # Find the matching close brace.
        depth = 0
        end = start
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if depth != 0:
            break
        candidate = text[start : end + 1]
        if marker in candidate:
            try:
                results.append(_json.loads(candidate))
            except _json.JSONDecodeError:
                pass
        i = end + 1
    return results


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


# --------------------------------------------------------------------------- #
# Model-family-specific format example block
# --------------------------------------------------------------------------- #

class TestModelFamilyFormatBlock:
    """When use_tools_channel=True AND a known model_family is passed,
    the system prompt must include `## Tool Call Format` with the
    canonical shape that family uses. The shape's special tokens land
    as their actual single-special-token IDs when this prompt is
    tokenized, anchoring the model's marginal probability on those IDs
    at every turn so format drift can't compound on long rollouts.

    Without family info (or use_tools_channel=False), the block must be
    omitted or use the legacy Qwen text-grammar example respectively.
    See env.py:build_system_content branches.
    """

    def test_kimi_family_emits_kimi_canonical_block(self):
        out = build_system_content(
            SAMPLE_TOOLS, modality="browser_use",
            use_tools_channel=True, model_family="kimi", now=FIXED_NOW,
        )
        assert "## Tool Call Format" in out
        # Literal Kimi special-token markers must be present in the text
        # so the tokenizer re-encodes them as the right special-token IDs.
        assert "<|tool_call_begin|>" in out
        assert "<|tool_call_argument_begin|>" in out
        assert "<|tool_call_end|>" in out

    def test_qwen_family_emits_qwen_canonical_block(self):
        out = build_system_content(
            SAMPLE_TOOLS, modality="tool_use",
            use_tools_channel=True, model_family="qwen", now=FIXED_NOW,
        )
        assert "## Tool Call Format" in out
        assert "<tool_call>" in out
        assert "</tool_call>" in out
        # Kimi markers must NOT be present.
        assert "<|tool_call_begin|>" not in out

    def test_unknown_family_omits_block(self):
        out = build_system_content(
            SAMPLE_TOOLS, modality="tool_use",
            use_tools_channel=True, model_family="llama", now=FIXED_NOW,
        )
        # No format example for unknown families — safer than guessing.
        assert "## Tool Call Format" not in out

    def test_no_family_omits_block(self):
        out = build_system_content(
            SAMPLE_TOOLS, modality="tool_use",
            use_tools_channel=True, model_family=None, now=FIXED_NOW,
        )
        assert "## Tool Call Format" not in out

    def test_legacy_path_unaffected_by_family(self):
        """When use_tools_channel=False, build_system_content uses the
        text-grammar Qwen example (the legacy path) regardless of
        model_family — this branch was never gated on family."""
        out = build_system_content(
            SAMPLE_TOOLS, modality="tool_use",
            use_tools_channel=False, model_family="kimi", now=FIXED_NOW,
        )
        assert "## Tool Call Format" in out
        assert "<tool_call>" in out
