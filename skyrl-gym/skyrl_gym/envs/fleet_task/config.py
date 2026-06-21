"""Load and validate `fleet_task.yaml`, the env's operator-tunable config.

Lives as a tiny module so env.py can `from .config import get_config` and
get a cached, pydantic-validated object with no boilerplate. Reload by
calling `get_config.cache_clear()` from a test.

See fleet_task.yaml for the keys and rationale.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


_DEFAULT_YAML = Path(__file__).parent / "fleet_task.yaml"


class ModelFamilyConfig(BaseModel):
    """Per-model-family prompt content."""

    # Canonical tool-call shape the model emits. Inserted into the reject
    # message and into the `## Tool Call Format` block in the system
    # prompt. For Kimi the <|...|> markers are single special-token IDs
    # the tokenizer recognizes at encode time; same string lands the
    # right token IDs in the model's context.
    canonical_tool_call: str = Field(..., min_length=1)

    # List of strings appended to every observation, in order. Each item is
    # templated via string.Template.safe_substitute ($-syntax — avoids
    # collision with literal `{}` in the canonical value or tool arguments).
    # Substitutable vars: $turn, $max_turns, $canonical_tool_call.
    # Empty list / None = no per-turn injection (byte-identical to today's
    # behavior for callers that don't configure it).
    per_turn_reminder: List[str] = Field(default_factory=list)


class FleetTaskConfig(BaseModel):
    """All operator-tunable knobs for FleetTaskEnv."""

    # Seconds to await after each MCP call_tool before composing the next
    # observation. Keyed by modality. Missing keys default to 0.
    post_action_wait: Dict[str, float] = Field(default_factory=dict)

    # End-of-response terminators. Matched via endswith() after stripping
    # whitespace and common terminal punctuation — must be the literal
    # last meaningful content, not a substring anywhere.
    done_signals: List[str] = Field(default_factory=lambda: ["<done>", "[done]"])

    # Per-family prompt content. Keyed by family name (kimi, qwen, ...).
    model_families: Dict[str, ModelFamilyConfig] = Field(default_factory=dict)

    def post_action_wait_for(self, modality: str) -> float:
        """Wait duration for a given modality. 0 if unset."""
        return float(self.post_action_wait.get(modality, 0.0))

    def canonical_tool_call_for(self, family: Optional[str]) -> Optional[str]:
        """Canonical shape for a family, or None if family is missing or
        not configured. None means: emit the generic reject message and
        omit the format example. Safe degradation for unknown families."""
        if not family:
            return None
        fam = self.model_families.get(family)
        return fam.canonical_tool_call if fam else None

    def scaffold_for(
        self, family: Optional[str], turn: int, max_turns: int
    ) -> str:
        """Resolved per-turn observation scaffold: concatenation of the
        family's per_turn_reminder list with $turn/$max_turns/$canonical_tool_call
        substituted. '' when the family is missing / unconfigured / has an
        empty list — appending it is a no-op."""
        fam = self.model_families.get(family) if family else None
        if not fam or not fam.per_turn_reminder:
            return ""
        return "".join(
            Template(item).safe_substitute(
                turn=turn,
                max_turns=max_turns,
                canonical_tool_call=fam.canonical_tool_call,
            )
            for item in fam.per_turn_reminder
        )


def load_config(path: Path = _DEFAULT_YAML) -> FleetTaskConfig:
    """Load and validate the YAML config. Raises pydantic.ValidationError
    if the file is malformed — failing loud at startup is preferable to
    silently wrong behavior at rollout time."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return FleetTaskConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> FleetTaskConfig:
    """Cached accessor used at env init. Call cache_clear() to reload
    after editing the YAML in a test."""
    return load_config()
