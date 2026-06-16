"""Tool call parser for LLM-generated tool calls.

Recognizes the tool-call grammars used by the model families we train. Each
grammar maps onto the same `{"name": ..., "arguments": ...}` shape.

Tag-based (Qwen, Llama 3.x, etc.):
- <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- <function_call>{"name": "...", "arguments": {...}}</function_call>

Kimi-K2 native (multi-token specials):
- <|tool_calls_section_begin|>
    <|tool_call_begin|>{name}<|tool_call_argument_begin|>{args_json}<|tool_call_end|>
    (... more calls ...)
  <|tool_calls_section_end|>
  Where {name} may carry a "functions." prefix and ":N" id suffix.

Handles missing closing tags (e.g., when </tool_call> is the stop string)
and repairs common JSON issues like missing trailing braces.
"""

import json
import re
from typing import Any, Dict, Optional


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON, repairing missing trailing braces if needed."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Repair: models often drop trailing closing braces on nested JSON.
    # Try appending up to 3 closing braces.
    for extra in range(1, 4):
        try:
            parsed = json.loads(raw + "}" * extra)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return None


# Kimi-K2 native tool-call grammar. Each delimiter is a single vocab token
# in Kimi's tokenizer but decodes as literal text, so we can regex-match it.
_KIMI_CALL_BEGIN = "<|tool_call_begin|>"
_KIMI_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
_KIMI_CALL_END = "<|tool_call_end|>"
_KIMI_CALL_RE = re.compile(
    re.escape(_KIMI_CALL_BEGIN) + r"\s*(.*?)\s*"
    + re.escape(_KIMI_CALL_ARG_BEGIN) + r"\s*(.*?)\s*"
    + r"(?:" + re.escape(_KIMI_CALL_END) + r"|\Z)",
    re.DOTALL,
)
# Strip optional "functions." namespace prefix and optional ":N" id suffix.
_KIMI_NAME_RE = re.compile(r"^(?:functions\.)?(.+?)(?::\d+)?$")


def _parse_kimi_call(action: str) -> Optional[Dict[str, Any]]:
    """Parse a Kimi-K2 native tool call (first one in the action)."""
    if _KIMI_CALL_BEGIN not in action:
        return None
    m = _KIMI_CALL_RE.search(action)
    if not m:
        return None
    raw_name = m.group(1).strip()
    raw_args = m.group(2).strip()
    nm = _KIMI_NAME_RE.match(raw_name)
    name = nm.group(1) if nm else raw_name
    if not name:
        return None
    parsed_args = _try_parse_json(raw_args) if raw_args else {}
    args = parsed_args if parsed_args is not None else {}
    return {"name": name, "arguments": args}


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from LLM response.

    Tries each known grammar in order; returns the first hit. See module
    docstring for the supported formats.

    Returns:
        Dict with "name" and "arguments" keys, or None if no tool call found.
    """
    # Tag-based grammars (Qwen, Llama 3.x, ...)
    for tag in ["tool_call", "function_call"]:
        # First try with closing tag
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if not match:
            # Try without closing tag (for when </tool_call> is the stop string)
            match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", action, re.DOTALL)
        if match:
            parsed = _try_parse_json(match.group(1))
            if parsed is None:
                continue
            # Normalize keys
            name = parsed.get("name") or parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("params", {})
            if name:
                return {"name": name, "arguments": args}

    # Kimi-K2 native grammar
    kimi = _parse_kimi_call(action)
    if kimi is not None:
        return kimi

    return None
