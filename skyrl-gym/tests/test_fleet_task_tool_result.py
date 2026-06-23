"""Behavioral spec for how FleetTaskEnv renders a tool_result into the
user-message content that goes into chat_history.

Written against the contract, not the implementation. The contract is:

  apply_chat_template only accepts two message-content shapes:
    1. a string body, or
    2. a list of OpenAI content blocks — list of dicts EACH carrying a
       "type" key ("text" / "image_url" / ...).
  Anything else (a plain list of records returned by a Fleet tool, an
  empty list, a mixed list with non-content-blocks, a dict, a number)
  must be rendered as a STRING for the tokenizer to accept it.

We test the helper `tool_result_to_message_content` that performs this
normalization. It returns whatever should go into `message["content"]`.

Run:
    pytest skyrl-gym/tests/test_fleet_task_tool_result.py -v
"""

from __future__ import annotations

import json


from skyrl_gym.envs.fleet_task.env import tool_result_to_message_content


# --- multimodal pass-through ---------------------------------------------


def test_openai_content_block_list_passes_through_unchanged():
    blocks = [
        {"type": "text", "text": "Look at this:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
    ]
    out = tool_result_to_message_content(blocks)
    assert out == blocks
    assert isinstance(out, list)


def test_single_text_content_block_list_passes_through():
    blocks = [{"type": "text", "text": "just text"}]
    out = tool_result_to_message_content(blocks)
    assert out == blocks


# --- plain Fleet tool returns must become strings ------------------------


def test_plain_list_of_records_is_json_string():
    """List of dicts WITHOUT a 'type' key (Fleet email/calendar/team
    listings) must be JSON-stringified — sending it raw crashes
    apply_chat_template with 'Input is not valid'."""
    records = [
        {"id": "1", "subject": "RE: API integration", "from": "alice@x.com"},
        {"id": "2", "subject": "Listing match", "from": "zillow@x.com"},
    ]
    out = tool_result_to_message_content(records)
    assert isinstance(out, str)
    # Content survives in the rendered body.
    assert "subject" in out
    assert "API integration" in out


def test_dict_tool_result_is_json_string():
    out = tool_result_to_message_content({"id": "abc", "count": 7})
    assert isinstance(out, str)
    assert '"id"' in out
    assert "abc" in out
    assert "7" in out


def test_empty_list_is_string():
    """Empty list is not meaningful multimodal content; treat as text."""
    out = tool_result_to_message_content([])
    assert isinstance(out, str)


def test_list_with_one_non_content_block_is_string():
    """If ANY element lacks the 'type' field, the whole list isn't valid
    multimodal content and must be JSON-stringified."""
    mixed = [{"type": "text", "text": "hi"}, {"id": "raw_record", "value": 1}]
    out = tool_result_to_message_content(mixed)
    assert isinstance(out, str)
    assert "raw_record" in out


def test_list_of_non_dicts_is_string():
    """e.g. a Fleet tool returning a list of IDs."""
    ids = ["id-1", "id-2", "id-3"]
    out = tool_result_to_message_content(ids)
    assert isinstance(out, str)
    assert "id-1" in out


# --- primitives ----------------------------------------------------------


def test_string_passes_through_in_text_body():
    out = tool_result_to_message_content("hello world")
    assert isinstance(out, str)
    assert "hello world" in out


def test_number_renders_as_string():
    out = tool_result_to_message_content(42)
    assert isinstance(out, str)
    assert "42" in out


# --- the string body is well-formed JSON when applicable -----------------


def test_string_body_is_round_trippable_for_dicts():
    """The rendered body for a dict tool_result should be JSON we can parse
    back out — preserves downstream debuggability."""
    obj = {"a": 1, "b": [True, None]}
    out = tool_result_to_message_content(obj)
    # Strip the human-friendly prefix if any, then ensure the rest parses.
    json_part = out.split("\n", 1)[-1] if "\n" in out else out
    parsed = json.loads(json_part)
    assert parsed == obj


def test_string_body_is_round_trippable_for_lists():
    records = [{"id": "1"}, {"id": "2"}]
    out = tool_result_to_message_content(records)
    json_part = out.split("\n", 1)[-1] if "\n" in out else out
    parsed = json.loads(json_part)
    assert parsed == records
