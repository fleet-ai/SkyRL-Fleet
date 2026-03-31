"""
WitnessHarness — optional enhancement layer for RL training.

Combines action semantics discovery, exploration tracking, cross-level
memory, and cognitive priors into a single interface. Only instantiated
when harness_mode=true; otherwise env.py behaves exactly as before.

Sub-switches allow ablation of individual mechanisms:
  harness_action_mapper: true/false
  harness_exploration: true/false
  harness_memory: true/false
  harness_priors: true/false

No LLM required — all mechanisms are algorithmic.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .action_mapper import ActionMapper
from .exploration import ExplorationTracker
from .memory import LevelMemory


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class WitnessHarness:
    """
    Optional enhancement layer providing richer observations and rewards.

    Activated by setting harness_mode=true in env_extras.
    Individual mechanisms can be toggled via sub-switches for ablation.
    """

    def __init__(
        self,
        game_id: str,
        initial_grid: np.ndarray,
        enable_action_mapper: bool = True,
        enable_exploration: bool = True,
        enable_memory: bool = True,
        enable_priors: bool = True,
    ):
        self.game_id = game_id
        self._enable_action_mapper = enable_action_mapper
        self._enable_exploration = enable_exploration
        self._enable_memory = enable_memory
        self._enable_priors = enable_priors

        self.action_mapper = ActionMapper() if enable_action_mapper else None
        self.exploration = ExplorationTracker() if enable_exploration else None
        self.memory = LevelMemory() if enable_memory else None
        self._action_history: List[int] = []

        # Record initial state
        if self.exploration:
            self.exploration.record_visit(initial_grid)

    def on_step(
        self,
        prev_grid: np.ndarray,
        action_id: int,
        new_grid: np.ndarray,
    ):
        """Called after each env step to update internal state."""
        if self.action_mapper:
            self.action_mapper.record(prev_grid, action_id, new_grid)
        if self.exploration:
            self.exploration.record_visit(new_grid)
        self._action_history.append(action_id)

    def on_level_solved(self, level_index: int, steps: int):
        """Called when a level is solved."""
        if self.memory:
            self.memory.store_level_result(
                level_index=level_index,
                action_history=self._action_history.copy(),
                steps=steps,
            )
        self._action_history.clear()

    def enrich_observation(
        self,
        grid: np.ndarray,
        step_count: int,
        level_index: int,
    ) -> str:
        """Return additional observation text to append after the board."""
        parts = []

        if self.action_mapper:
            sem = self.action_mapper.format_semantics()
            if sem:
                parts.append(sem)

        if self.exploration:
            cov = self.exploration.format_coverage()
            if cov:
                parts.append(cov)

        if self.memory:
            rules = self.memory.format_known_rules()
            if rules:
                parts.append(rules)

        return "\n".join(parts)

    def compute_bonus_reward(
        self,
        grid: np.ndarray,
        action_id: int,
        solved: bool,
    ) -> float:
        """Compute intrinsic reward bonus from enabled mechanisms."""
        bonus = 0.0
        if self.exploration:
            bonus += self.exploration.novelty_bonus(grid)
        if self.action_mapper:
            bonus += self.action_mapper.discovery_bonus(action_id)
        return bonus

    def get_system_prompt_addition(self) -> str:
        """Return cognitive priors text to append to the system prompt."""
        if not self._enable_priors:
            return ""
        path = os.path.join(_PROMPTS_DIR, "core_knowledge.txt")
        if os.path.exists(path):
            with open(path) as f:
                return "\n\n" + f.read().strip()
        return ""
