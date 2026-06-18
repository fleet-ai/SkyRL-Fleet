"""Behavioral spec for `truncate_tool_result`.

A tool can return any JSON-ish shape: a string (`execute_python` stdout),
a dict (`computer` tool screenshot+output), a list of records (`query_*`),
a primitive (numeric), or None. All of those can blow the next prompt past
max_input_length if returned in full. The helper must cap any of them at
`max_chars` and emit a "[TRUNCATED — N chars elided.]" suffix when it does.

Run:
    pytest skyrl-gym/tests/test_fleet_task_truncate.py -v
"""

from __future__ import annotations

import json

import pytest

from skyrl_gym.envs.fleet_task.env import (
    MAX_TOOL_OUTPUT_CHARS,
    truncate_tool_result,
)


SUFFIX_RE = r"\[TRUNCATED — \d+ chars elided\.\]"


# --- None pass-through ---------------------------------------------------

def test_none_passes_through():
    assert truncate_tool_result(None) is None


# --- short payloads (under cap) are unchanged ----------------------------

def test_short_string_unchanged():
    s = "hello world"
    out = truncate_tool_result(s, max_chars=100)
    assert out == s
    assert isinstance(out, str)


def test_short_dict_serialized_within_cap_returned_as_dict():
    """Dicts under the cap come back as dict (so downstream layers like
    tool_result_to_message_content can still detect 'this is a dict')."""
    d = {"status": "ok", "n": 3}
    out = truncate_tool_result(d, max_chars=1000)
    assert isinstance(out, dict)
    assert out == d


def test_short_list_within_cap_returned_as_list():
    rows = [{"id": 1}, {"id": 2}]
    out = truncate_tool_result(rows, max_chars=1000)
    assert isinstance(out, list)
    assert out == rows


def test_primitive_int_unchanged_when_short():
    out = truncate_tool_result(42, max_chars=100)
    assert out == 42 or out == "42"  # implementation may stringify; both fine


# --- overflowing payloads get truncated ----------------------------------

def test_long_string_truncated():
    s = "x" * 5000
    out = truncate_tool_result(s, max_chars=100)
    assert isinstance(out, str)
    assert len(out) < 5000
    assert out.startswith("x" * 100)
    import re
    assert re.search(SUFFIX_RE, out), out[-200:]


def test_huge_dict_serialized_and_truncated():
    """A dict whose JSON-serialization exceeds the cap must be reduced to a
    truncated string body — caller can't process a 50K-token dict either."""
    big = {f"k{i}": "v" * 200 for i in range(200)}
    serialized_len = len(json.dumps(big, default=str))
    assert serialized_len > 1000, "fixture should be over the cap"

    out = truncate_tool_result(big, max_chars=1000)
    assert isinstance(out, str), "huge dict should be cast to string"
    assert len(out) < serialized_len
    import re
    assert re.search(SUFFIX_RE, out)


def test_huge_list_of_records_serialized_and_truncated():
    """The production failure mode: query_data_lake returns 10K rows."""
    rows = [{"id": i, "name": "row" * 20} for i in range(5000)]
    serialized_len = len(json.dumps(rows, default=str))
    assert serialized_len > 16_000

    out = truncate_tool_result(rows, max_chars=MAX_TOOL_OUTPUT_CHARS)
    assert isinstance(out, str)
    assert len(out) < serialized_len
    import re
    assert re.search(SUFFIX_RE, out)


# --- non-JSON-serializable shapes fall back to str(...) ------------------

def test_non_json_serializable_falls_back_to_str():
    class Weird:
        def __repr__(self):
            return "Weird(" + "x" * 5000 + ")"

    out = truncate_tool_result(Weird(), max_chars=200)
    assert isinstance(out, str)
    assert len(out) < 5000
    import re
    assert re.search(SUFFIX_RE, out)


# --- suffix accuracy -----------------------------------------------------

def test_truncated_suffix_reports_correct_elided_count():
    s = "a" * 1500
    out = truncate_tool_result(s, max_chars=1000)
    # We kept the first 1000 chars; 500 should be reported as elided.
    import re
    m = re.search(r"\[TRUNCATED — (\d+) chars elided\.\]", out)
    assert m, out[-300:]
    assert int(m.group(1)) == 500


# --- default max_chars uses MAX_TOOL_OUTPUT_CHARS ------------------------

def test_default_max_chars_is_module_constant():
    s = "y" * (MAX_TOOL_OUTPUT_CHARS + 100)
    out = truncate_tool_result(s)
    assert len(out) > MAX_TOOL_OUTPUT_CHARS  # includes the suffix
    assert out.startswith("y" * MAX_TOOL_OUTPUT_CHARS)
    import re
    assert re.search(SUFFIX_RE, out)
