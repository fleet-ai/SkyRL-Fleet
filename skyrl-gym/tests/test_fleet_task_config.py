"""Tests for the fleet_task YAML config + the is_done_signal helper.

Two pieces under test:
  1. `config.get_config()` reads fleet_task.yaml and exposes
     post_action_wait, done_signals, and per-family canonical_tool_call.
  2. `env.is_done_signal()` decides whether the model's last response is
     a done signal — must be the LITERAL end of the response, not a
     substring match anywhere. The previous substring match fired on
     quoted system-prompt references (14/14 sessions in job c4b429ae
     terminated this way with score=0).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.config import (
    FleetTaskConfig,
    get_config,
    load_config,
)
from skyrl_gym.envs.fleet_task.env import is_done_signal


# --------------------------------------------------------------------------- #
# Config loader
# --------------------------------------------------------------------------- #

class TestConfigLoader:
    def test_default_yaml_loads(self):
        """The shipped YAML must parse and produce a valid config."""
        cfg = get_config()
        assert isinstance(cfg, FleetTaskConfig)

    def test_post_action_wait_per_modality(self):
        cfg = get_config()
        # BU and CU wait for the browser to repaint; tool_use does not.
        assert cfg.post_action_wait_for("browser_use") > 0
        assert cfg.post_action_wait_for("computer_use") > 0
        assert cfg.post_action_wait_for("tool_use") == 0.0

    def test_post_action_wait_unknown_modality_returns_zero(self):
        """A modality not in the YAML must not error — return 0 and
        proceed without a wait. Future-modality safety."""
        cfg = get_config()
        assert cfg.post_action_wait_for("future_modality") == 0.0

    def test_done_signals_present(self):
        cfg = get_config()
        assert "<done>" in cfg.done_signals
        assert "[done]" in cfg.done_signals

    def test_kimi_family_present(self):
        cfg = get_config()
        canon = cfg.canonical_tool_call_for("kimi")
        assert canon is not None
        # The kimi canonical must reference the native special tokens.
        # When this string is tokenized, those <|...|> markers become
        # single special-token IDs — that's the whole point.
        assert "<|tool_call_begin|>" in canon
        assert "<|tool_call_argument_begin|>" in canon
        assert "<|tool_call_end|>" in canon

    def test_qwen_family_canonical_intentionally_unset(self):
        """Qwen has NO canonical_tool_call. Its chat template injects the
        `<tool_call>{...}</tool_call>` spec via apply_chat_template's
        `tools` argument, and Qwen knows the grammar from pretraining —
        echoing it back in the system prompt + reject path is dead
        weight. The system-prompt format block and the no-tool-call
        reject canonical example are both skipped for Qwen as a result.
        Pinned so a future re-add must be justified."""
        cfg = get_config()
        assert cfg.canonical_tool_call_for("qwen") is None

    def test_unknown_family_returns_none(self):
        cfg = get_config()
        assert cfg.canonical_tool_call_for("llama") is None
        assert cfg.canonical_tool_call_for(None) is None
        assert cfg.canonical_tool_call_for("") is None

    def test_load_config_rejects_malformed_yaml(self, tmp_path):
        # Non-mapping at top level
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just_a_list\n- of_items\n")
        with pytest.raises(ValueError):
            load_config(bad)

    def test_load_config_accepts_minimal_yaml(self, tmp_path):
        """Empty config (all defaults) must parse — useful for tests
        that disable post-action waits."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("{}\n")
        cfg = load_config(empty)
        assert cfg.post_action_wait_for("browser_use") == 0.0
        assert "<done>" in cfg.done_signals  # default still applied
        assert cfg.canonical_tool_call_for("kimi") is None  # no families


# --------------------------------------------------------------------------- #
# is_done_signal — the bug fix
# --------------------------------------------------------------------------- #

DONE_SIGS = ["<done>", "[done]"]


class TestIsDoneSignal:
    """Walk through every case from the failure analysis of job c4b429ae.

    The previous logic was `"<done>" in action.lower()` which fired on
    any occurrence. The fix requires done to be the LITERAL end of the
    response after stripping trailing whitespace and common terminal
    punctuation.
    """

    # ---- Legitimate done signals: must return True ----

    def test_done_alone(self):
        assert is_done_signal("<done>", DONE_SIGS)

    def test_done_after_final_answer(self):
        """The system prompt instructs: write the final answer, then
        <done>. This must terminate."""
        assert is_done_signal("The answer is 42. <done>", DONE_SIGS)

    def test_done_with_trailing_period(self):
        # Sampler sometimes adds punctuation after the marker.
        assert is_done_signal("Done. <done>.", DONE_SIGS)

    def test_done_with_trailing_whitespace(self):
        assert is_done_signal("The answer is 42. <done>   \n\n", DONE_SIGS)

    def test_bracket_done_signal(self):
        """[done] is also a valid terminator per the YAML."""
        assert is_done_signal("All good. [done]", DONE_SIGS)

    def test_done_case_insensitive(self):
        # The previous code lowercased; preserve that.
        assert is_done_signal("<DONE>", DONE_SIGS)

    # ---- The bug: quoted/in-body done references must NOT terminate ----

    def test_quoted_system_prompt_does_not_terminate(self):
        """THE REGRESSION TEST. From session 168914cd: the model quotes
        the system prompt back to itself while debugging its tool-call
        format. The string `<done>` appears in the body of the response;
        the actual response ends with a (broken) tool call attempt. The
        old code matched the quote and ended the episode prematurely
        with score=0 — 14/14 sessions in job c4b429ae died this way."""
        action = textwrap.dedent('''\
            Looking at the instructions: "Done signal: <done> - ONLY when
            the task is fully complete."
            Let me try the tool call again.
            <|tool_calls_section_begin|>functions.computer:5{"action":"screenshot"}<|tool_call_end|>
        ''')
        assert not is_done_signal(action, DONE_SIGS)

    def test_done_in_middle_then_tool_call_section_end_does_not_terminate(self):
        """Trailing content after the in-body done must not be considered."""
        action = (
            "I'll emit <done> after this tool call.\n"
            "<|tool_calls_section_begin|>...<|tool_calls_section_end|>"
        )
        assert not is_done_signal(action, DONE_SIGS)

    def test_done_followed_by_more_text_does_not_terminate(self):
        """Done should be a terminator. Anything after means the model
        isn't actually finalizing."""
        action = "<done> Let me know if you need anything else."
        assert not is_done_signal(action, DONE_SIGS)

    def test_tool_call_alone_does_not_terminate(self):
        action = "<|tool_calls_section_begin|>...<|tool_calls_section_end|>"
        assert not is_done_signal(action, DONE_SIGS)

    def test_empty_action_does_not_terminate(self):
        assert not is_done_signal("", DONE_SIGS)

    def test_only_whitespace_does_not_terminate(self):
        assert not is_done_signal("   \n\n  ", DONE_SIGS)

    def test_unrelated_text_does_not_terminate(self):
        assert not is_done_signal("just thinking out loud", DONE_SIGS)

    # ---- Edge cases around the trailing punctuation strip ----

    def test_done_with_multiple_terminal_punctuation(self):
        # Stripper handles . ! ? ' " ` * — all common sampler artifacts.
        assert is_done_signal("Done!!!! <done>!!!", DONE_SIGS)
        assert is_done_signal("Done? <done>?", DONE_SIGS)
        assert is_done_signal("Done. <done>*", DONE_SIGS)
