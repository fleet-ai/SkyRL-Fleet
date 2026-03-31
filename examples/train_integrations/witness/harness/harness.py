"""
WitnessHarness — optional enhancement layer for RL training.

Combines action semantics discovery, exploration tracking, cross-level
memory, and cognitive priors into a single interface. Only instantiated
when harness_mode=true; otherwise env.py behaves exactly as before.

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
    When not activated, env.py skips all harness logic entirely.
    """

    def __init__(self, game_id: str, initial_grid: np.ndarray):
        self.game_id = game_id
        self.action_mapper = ActionMapper()
        self.exploration = ExplorationTracker()
        self.memory = LevelMemory()
        self._action_history: List[int] = []

        # Record initial state
        self.exploration.record_visit(initial_grid)

    def on_step(
        self,
        prev_grid: np.ndarray,
        action_id: int,
        new_grid: np.ndarray,
    ):
        """Called after each env step to update internal state."""
        self.action_mapper.record(prev_grid, action_id, new_grid)
        self.exploration.record_visit(new_grid)
        self._action_history.append(action_id)

    def on_level_solved(self, level_index: int, steps: int):
        """Called when a level is solved."""
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
        """
        Return additional observation text to append after the board.

        Includes: action semantics, exploration stats, known rules.
        """
        parts = []

        # Action semantics (e.g., "Actions: 1=UP 2=DOWN 3=LEFT 4=RIGHT 5=CONFIRM")
        sem = self.action_mapper.format_semantics()
        if sem:
            parts.append(sem)

        # Exploration coverage
        cov = self.exploration.format_coverage()
        if cov:
            parts.append(cov)

        # Cross-level knowledge
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
        """
        Compute intrinsic reward bonus from harness mechanisms.

        Returns a small bonus on top of the base shaped/sparse reward.
        """
        bonus = 0.0
        bonus += self.exploration.novelty_bonus(grid)
        bonus += self.action_mapper.discovery_bonus(action_id)
        return bonus

    def get_system_prompt_addition(self) -> str:
        """
        Return cognitive priors text to append to the system prompt.

        Loaded from prompts/core_knowledge.txt.
        """
        path = os.path.join(_PROMPTS_DIR, "core_knowledge.txt")
        if os.path.exists(path):
            with open(path) as f:
                return "\n\n" + f.read().strip()
        return ""
