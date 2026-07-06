"""Tests for `skyrl_gym.envs.fleet_task.tool_call_parser.parse_tool_call`.

Written blind from the supported-format spec (module docstring + Kimi's
chat_template.jinja), not by reading the implementation.

Each grammar should round-trip to `{"name": str, "arguments": dict}`.

Tag-based grammars (Qwen, Llama 3.x, etc.):
  <tool_call>{"name": ..., "arguments": ...}</tool_call>
  <function_call>{"name": ..., "arguments": ...}</function_call>

Kimi-K2 native grammar:
  <|tool_calls_section_begin|>
    <|tool_call_begin|>{call_id}<|tool_call_argument_begin|>{args_json}<|tool_call_end|>
    ...
  <|tool_calls_section_end|>

  where `{call_id}` is the OpenAI-style `functions.{name}:{index}` reference;
  the parser must return `name=...` (stripping prefix and suffix).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call  # noqa: E402  (import after sys.path setup)


# --------------------------------------------------------------------------- #
# Tag-based formats (Qwen / Llama / others)
# --------------------------------------------------------------------------- #


class TestTagFormats:
    def test_tool_call_with_closing_tag(self):
        text = '<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>'
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "ls"}}

    def test_function_call_with_closing_tag(self):
        text = '<function_call>{"name": "search", "arguments": {"q": "hello"}}</function_call>'
        result = parse_tool_call(text)
        assert result == {"name": "search", "arguments": {"q": "hello"}}

    def test_tool_call_missing_closing_tag(self):
        # When </tool_call> is the sampling stop string it gets stripped.
        text = '<tool_call>{"name": "bash", "arguments": {"cmd": "echo hi"}}'
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "echo hi"}}

    def test_tool_call_with_surrounding_text(self):
        text = 'Sure, let me run that.\n<tool_call>{"name": "bash", "arguments": {"cmd": "pwd"}}</tool_call>\nDone.'
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "pwd"}}

    def test_alternative_keys_tool_and_params(self):
        # The parser normalizes "tool" -> "name" and "params" -> "arguments".
        text = '<tool_call>{"tool": "bash", "params": {"cmd": "uname -a"}}</tool_call>'
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "uname -a"}}

    def test_missing_trailing_brace_repaired(self):
        # Models sometimes drop the trailing closing brace; parser must repair.
        text = '<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}</tool_call>'
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "ls"}}

    def test_no_tool_call_returns_none(self):
        assert parse_tool_call("Just some reasoning, no tool call here.") is None
        assert parse_tool_call("") is None

    def test_malformed_json_returns_none(self):
        text = '<tool_call>{"name": bash, "arguments": {"cmd"}}</tool_call>'
        assert parse_tool_call(text) is None


# --------------------------------------------------------------------------- #
# Kimi-K2 native format
# --------------------------------------------------------------------------- #


class TestKimiK2Format:
    # Real output captured from `moonshotai/Kimi-K2.6:peft:131072` against
    # a Fleet data-eng env (single-rollout debug script, see CHANGELOG).
    KIMI_REAL_OUTPUT = (
        "The previous response didn't include a tool call. I need to use the "
        "correct format. Let me call aws s3 ls.</think>"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.aws:1"
        "<|tool_call_argument_begin|>"
        '{"args": ["s3","ls"]}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    def test_parses_real_captured_output(self):
        result = parse_tool_call(self.KIMI_REAL_OUTPUT)
        # `functions.` prefix and `:1` index suffix should be stripped from the call_id.
        assert result is not None
        assert result["name"] == "aws"
        assert result["arguments"] == {"args": ["s3", "ls"]}

    def test_minimal_single_call(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.bash:1"
            "<|tool_call_argument_begin|>"
            '{"cmd": "ls"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "ls"}}

    def test_strips_functions_prefix(self):
        text = "<|tool_call_begin|>functions.my_tool:5" "<|tool_call_argument_begin|>{}" "<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result["name"] == "my_tool"

    def test_strips_index_suffix(self):
        text = "<|tool_call_begin|>functions.x:42" "<|tool_call_argument_begin|>{}" "<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result["name"] == "x"

    def test_handles_no_functions_prefix(self):
        # If a future Kimi config drops the prefix, the parser should still work.
        text = "<|tool_call_begin|>bash:1" "<|tool_call_argument_begin|>" '{"cmd": "pwd"}' "<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "pwd"}}

    def test_handles_no_index_suffix(self):
        text = "<|tool_call_begin|>functions.bash" "<|tool_call_argument_begin|>" '{"cmd": "pwd"}' "<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "pwd"}}

    def test_handles_missing_end_marker(self):
        # When `<|tool_calls_section_end|>` is a stop sequence, the per-call
        # `<|tool_call_end|>` may also get stripped.
        text = "<|tool_call_begin|>functions.bash:1" "<|tool_call_argument_begin|>" '{"cmd": "ls"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "bash"
        assert result["arguments"] == {"cmd": "ls"}

    def test_handles_nested_json_args(self):
        text = (
            "<|tool_call_begin|>functions.run:1"
            "<|tool_call_argument_begin|>"
            '{"config": {"timeout": 30, "env": {"FOO": "bar"}}}'
            "<|tool_call_end|>"
        )
        result = parse_tool_call(text)
        assert result["name"] == "run"
        assert result["arguments"] == {"config": {"timeout": 30, "env": {"FOO": "bar"}}}

    def test_handles_reasoning_before_tool_call(self):
        text = (
            "<think>Let me think... I'll run ls.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.bash:1"
            "<|tool_call_argument_begin|>"
            '{"cmd": "ls"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"cmd": "ls"}}

    def test_first_call_wins_for_multi_call_section(self):
        # If Kimi emits multiple calls in one section, parse_tool_call only
        # returns the first (matches existing tag-format behavior).
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.first:1"
            "<|tool_call_argument_begin|>"
            '{"a": 1}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.second:2"
            "<|tool_call_argument_begin|>"
            '{"b": 2}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_call(text)
        assert result["name"] == "first"
        assert result["arguments"] == {"a": 1}

    def test_empty_args_dict(self):
        text = "<|tool_call_begin|>functions.noop:1" "<|tool_call_argument_begin|>{}" "<|tool_call_end|>"
        result = parse_tool_call(text)
        assert result == {"name": "noop", "arguments": {}}


# --------------------------------------------------------------------------- #
# Cross-format isolation
# --------------------------------------------------------------------------- #


class TestCrossFormatIsolation:
    def test_text_with_neither_format_returns_none(self):
        text = "I'd like to call ls, but I don't know the right format. " "Let me think about this more carefully."
        assert parse_tool_call(text) is None

    def test_tag_format_takes_precedence_when_both_present(self):
        # If a model somehow emits both formats, the tag format is tried
        # first (matches the iteration order in parse_tool_call).
        text = (
            '<tool_call>{"name": "tag_winner", "arguments": {}}</tool_call>\n'
            "<|tool_call_begin|>functions.kimi_loser:1<|tool_call_argument_begin|>{}<|tool_call_end|>"
        )
        result = parse_tool_call(text)
        assert result["name"] == "tag_winner"


# --------------------------------------------------------------------------- #
# Qwen3.6 XML-function grammar (chat_template.jinja, new in 3.6)
# --------------------------------------------------------------------------- #


class TestQwen36XmlFunctionGrammar:
    def test_wrapped_single_param(self):
        text = (
            "I'll check the file.\n"
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            "ls -la /home\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        assert parse_tool_call(text) == {"name": "bash", "arguments": {"command": "ls -la /home"}}

    def test_multiline_param_value_preserved(self):
        text = (
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            "cat <<'EOS' > /tmp/x\nline1\nline2\nEOS\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        result = parse_tool_call(text)
        assert result["name"] == "bash"
        assert result["arguments"]["command"] == "cat <<'EOS' > /tmp/x\nline1\nline2\nEOS"

    def test_multiple_params_and_json_literal_coercion(self):
        text = (
            "<tool_call>\n<function=search_listings>\n"
            "<parameter=query>\n3 bed house\n</parameter>\n"
            "<parameter=limit>\n5\n</parameter>\n"
            "<parameter=verified>\ntrue\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parse_tool_call(text)
        assert result == {
            "name": "search_listings",
            "arguments": {"query": "3 bed house", "limit": 5, "verified": True},
        }

    def test_missing_closing_tags_stop_string_case(self):
        # </tool_call> (or even </function>) can be eaten by a stop string.
        text = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n"
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"command": "pwd"}}

    def test_bare_function_without_tool_call_wrapper(self):
        text = "<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>"
        result = parse_tool_call(text)
        assert result == {"name": "bash", "arguments": {"command": "echo hi"}}

    def test_no_params_yields_empty_arguments(self):
        text = "<tool_call>\n<function=list_tools>\n</function>\n</tool_call>"
        assert parse_tool_call(text) == {"name": "list_tools", "arguments": {}}

    def test_json_grammar_still_wins_inside_same_tag(self):
        text = '<tool_call>{"name": "json_style", "arguments": {"a": 1}}</tool_call>'
        assert parse_tool_call(text) == {"name": "json_style", "arguments": {"a": 1}}
