"""Per-model-family adapters for FleetTaskEnv.

Each model family (Kimi-K2.6, Qwen3, ...) has its own chat-template
expectations for how assistant messages should be structured. The adapter
in this file owns all of that family-specific behavior:

  - parsing the model's raw text emission into structured fields
    (reasoning_content + content + tool_calls) the way that family's chat
    template renders cleanly
  - generating the canonical-format example string (used in system-prompt
    format block + reject-after-no-tool-call message)
  - generating the per-turn reminder text appended to every observation

env.py is family-agnostic; it looks up the family by name from
`extras["model_family"]` and delegates these decisions to the adapter.

## Why a Family class instead of YAML config

The previous approach kept canonical-tool-call strings and per-turn-reminder
templates in fleet_task.yaml. That worked for static values but couldn't
express LOGIC: the Kimi adapter needs to split <think>...</think> out of
the model's emission and route it to a separate `reasoning_content` field
so Kimi's chat template renders one clean <think> block instead of two
(empty default + raw inline). That's protocol code, not a YAML value.

Folding the YAML values into the family class puts the values next to
the logic that uses them — one place to look, one place to change when a
new family arrives.

## What's family-specific (and why)

| Concern | Kimi-K2.6 | Qwen3 |
|---|---|---|
| Chat template behavior | Injects empty `<think></think>` when `reasoning_content` is absent → double-think bug if we pass raw `<think>X</think>` in `content` | Parses inline `<think>X</think>` from content itself; renders one clean block either way |
| Tool call section | `<|tool_call_begin|>`/`<|tool_call_argument_begin|>` special tokens | `<tool_call>{json}</tool_call>` text grammar |
| Tool call id format | `functions.NAME:N` (model's training prior) | `call_N` (arbitrary OpenAI id) |
| Per-turn format reminder | Worth echoing the canonical shape so special-token IDs land in context | Token IDs are text-grammar anyway; mostly noise |

The Kimi double-think bug source (Kimi chat template, lines that emit
<think>): when `rc` (reasoning_content) is not set on the assistant
message, the template emits `<think></think>{render_content(message)}`.
Our raw-string `content` already contains its own `<think>...</think>`
text, so two blocks land in the rendered prompt. Setting
reasoning_content explicitly takes the other branch and renders one
clean block. A/B-verified 2026-06-21 via apply_chat_template diff.

Qwen's template extracts `<think>...</think>` from content directly
(see Qwen3 chat_template.jinja: `content.split('</think>')[0]....split('<think>')[-1]`),
so raw passthrough renders cleanly. No restructuring needed for Qwen.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol


class Family(Protocol):
    """Per-family adapter protocol. env.py looks up by name and delegates."""

    name: str
    # Canonical tool-call shape used in the system-prompt format block and
    # in the no-tool-call reject body. Concrete example (not <name>:<n>
    # placeholders) so the model has something it can imitate verbatim.
    # None when the family's chat template injects format guidance itself
    # via the `tools` parameter (e.g. Qwen).
    canonical_tool_call: Optional[str]

    def build_assistant_message(
        self, action: str, parsed_tool_call: Optional[Dict[str, Any]], turn: int
    ) -> Dict[str, Any]:
        """Transform the model's raw text emission into a structured
        assistant message dict ready to append to chat_history. The shape
        depends on what this family's chat template renders cleanly."""

    def per_turn_reminder(self, turn: int, max_turns: int) -> str:
        """Text appended to every observation message. Empty string means
        no per-turn injection for this family."""

    def reject_message(self) -> str:
        """Body of the 'No tool call landed' observation that fires when
        parse_tool_call returns None."""


# --------------------------------------------------------------------------- #
# Kimi-K2.6
# --------------------------------------------------------------------------- #

_KIMI_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_KIMI_TOOL_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
    re.DOTALL,
)


class Kimi:
    """Kimi-K2.6 adapter.

    Splits the model's raw emission into:
      - reasoning_content: concatenation of all <think>...</think> bodies
      - content: free text outside both <think> blocks AND the tool-call section
      - tool_calls: OpenAI shape with id `functions.NAME:turn`

    The structured shape avoids the Kimi chat template's double-think bug
    and makes the trace viewer renderable as separate reasoning/action
    panels.
    """

    name = "kimi"
    # Concrete example, NOT <name>:<n> placeholders — the model previously
    # copied placeholder text literally as a call name. Use a real call
    # the model can imitate verbatim and have it work end-to-end.
    canonical_tool_call = (
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.computer:0'
        '<|tool_call_argument_begin|>{"action":"screenshot"}'
        '<|tool_call_end|><|tool_calls_section_end|>'
    )

    def build_assistant_message(
        self, action: str, parsed_tool_call: Optional[Dict[str, Any]], turn: int
    ) -> Dict[str, Any]:
        thinks = [t.strip() for t in _KIMI_THINK_RE.findall(action) if t.strip()]
        reasoning = "\n\n".join(thinks)
        free = _KIMI_TOOL_SECTION_RE.sub("", _KIMI_THINK_RE.sub("", action)).strip()
        msg: Dict[str, Any] = {"role": "assistant", "content": free}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if parsed_tool_call:
            msg["tool_calls"] = [{
                "id": f"functions.{parsed_tool_call['name']}:{turn}",
                "type": "function",
                "function": {
                    "name": parsed_tool_call["name"],
                    "arguments": json.dumps(parsed_tool_call.get("arguments", {})),
                },
            }]
        return msg

    def per_turn_reminder(self, turn: int, max_turns: int) -> str:
        return (
            f"\n\n[Turn {turn}/{max_turns}]\n"
            f"[Next tool call format]\n{self.canonical_tool_call}"
        )

    def reject_message(self) -> str:
        return (
            "No tool call landed. End your response with exactly this shape "
            "(the <|...|> markers below are single special tokens, not literal "
            f"text):\n{self.canonical_tool_call}"
        )


# --------------------------------------------------------------------------- #
# Qwen3 (and Qwen2.5)
# --------------------------------------------------------------------------- #


class Qwen:
    """Qwen3 adapter.

    Qwen's chat template parses inline `<think>...</think>` from content
    directly (see Qwen3 chat_template.jinja), so raw-text passthrough in
    `content` renders cleanly with one think block. No restructuring needed.

    canonical_tool_call is None: Qwen's template renders the tool-format
    spec via the `tools=` argument to apply_chat_template; echoing it
    back in the system prompt or reject is redundant and noisy.
    """

    name = "qwen"
    canonical_tool_call: Optional[str] = None

    def build_assistant_message(
        self, action: str, parsed_tool_call: Optional[Dict[str, Any]], turn: int
    ) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": action}
        if parsed_tool_call:
            msg["tool_calls"] = [{
                "id": f"call_{turn}",
                "type": "function",
                "function": {
                    "name": parsed_tool_call["name"],
                    # Dict, NOT json.dumps: Qwen3.6's chat template iterates
                    # `tool_call.arguments|items` to render <parameter=...>
                    # blocks and raises "Can only get item pairs from a
                    # mapping" on a string. Qwen3.x's `|tojson` renders a
                    # dict correctly (a string would double-encode).
                    "arguments": parsed_tool_call.get("arguments", {}),
                },
            }]
        return msg

    def per_turn_reminder(self, turn: int, max_turns: int) -> str:
        # Indicator only — Qwen knows its format from training + apply_chat_template's tools=.
        return f"\n\n[Turn {turn}/{max_turns}]"

    def reject_message(self) -> str:
        return (
            "No tool call landed. End your response with a tool call in "
            "the canonical format for your model."
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Family] = {
    "kimi": Kimi(),
    "qwen": Qwen(),
}


def get_family(name: Optional[str]) -> Optional[Family]:
    """Look up a family adapter by name. Returns None for missing/unknown
    family — env.py falls back to its byte-identical passthrough behavior
    so a new caller without family registration sees no regression."""
    if not name:
        return None
    return _REGISTRY.get(name)


def family_for_model(model_name: Optional[str]) -> Optional[str]:
    """Prefix-based derivation of family name from a HuggingFace model id.
    Used by main_fleet_tinker (Tinker entrypoint) and skyrl-train's
    generator to set extras['model_family'] from the model name."""
    if not model_name:
        return None
    n = model_name.lower()
    if "moonshotai" in n or "kimi" in n:
        return "kimi"
    if "qwen" in n:
        return "qwen"
    return None
