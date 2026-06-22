"""Load and validate `fleet_task.yaml`, the env's operator-tunable config.

Per-family logic (canonical tool-call shape, per-turn reminder, reject
message, assistant-message structuring) lives in families.py. This file
is for runtime knobs that don't depend on the model family.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


_DEFAULT_YAML = Path(__file__).parent / "fleet_task.yaml"


class FleetTaskConfig(BaseModel):
    """Operator-tunable knobs for FleetTaskEnv."""

    # Seconds to await after each MCP call_tool before composing the next
    # observation. Keyed by modality. Missing keys default to 0.
    post_action_wait: Dict[str, float] = Field(default_factory=dict)

    # End-of-response terminators. Matched via endswith() after stripping
    # whitespace and common terminal punctuation: must be the literal
    # last meaningful content, not a substring anywhere.
    done_signals: List[str] = Field(default_factory=lambda: ["<done>", "[done]"])

    def post_action_wait_for(self, modality: str) -> float:
        """Wait duration for a given modality. 0 if unset."""
        return float(self.post_action_wait.get(modality, 0.0))


def load_config(path: Path = _DEFAULT_YAML) -> FleetTaskConfig:
    """Load and validate the YAML config. Raises pydantic.ValidationError
    if the file is malformed: failing loud at startup is preferable to
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
