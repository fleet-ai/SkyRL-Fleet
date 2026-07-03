"""Tool call parser for LLM-generated tool calls.

Recognizes the tool-call grammars used by the model families we train. Each
grammar maps onto the same `{"name": ..., "arguments": ...}` shape.

Tag-based (Qwen3.x, Llama 3.x, etc.):
- <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- <function_call>{"name": "...", "arguments": {...}}</function_call>

Qwen3.6 XML-function (same <tool_call> tag, non-JSON payload):
- <tool_call>
  <function=name>
  <parameter=key>
  value
  </parameter>
  </function>
  </tool_call>
  Parameter values are raw text; pure JSON literals are type-coerced.

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


# Qwen3.6 XML-function grammar (chat_template.jinja):
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE            (raw text, may span lines)
#   </parameter>
#   ...
#   </function>
#   </tool_call>
# Same outer <tool_call> tag as the Qwen3-era JSON grammar, different payload.
_XML_FN_RE = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|\Z)", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _coerce_param(raw: str) -> Any:
    """JSON-literal values (numbers, bools, null, objects, arrays) become
    typed; anything else stays a raw string. The grammar itself is untyped,
    but MCP tools with e.g. integer params reject "5"-as-string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _parse_xml_function_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first Qwen3.6-style <function=...> call in `text`."""
    m = _XML_FN_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    args = {k: _coerce_param(v) for k, v in _XML_PARAM_RE.findall(m.group(2))}
    return {"name": name, "arguments": args}


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
                # Same tag, non-JSON payload: Qwen3.6's XML-function grammar.
                xml = _parse_xml_function_call(match.group(1))
                if xml is not None:
                    return xml
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

    # Bare XML-function call (model dropped the <tool_call> wrapper)
    return _parse_xml_function_call(action)
