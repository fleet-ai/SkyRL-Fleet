"""
Exploration tracking and intrinsic reward signals.

Inspired by SCOUT's count-based novelty: rewards for visiting new states
and covering more of the state space.

No LLM required — pure state counting.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict

import numpy as np


def _hash_grid(grid: np.ndarray) -> str:
    """Fast hash of a 64x64 grid for state identity."""
    return hashlib.md5(grid.tobytes()).hexdigest()[:12]


class ExplorationTracker:
    """
    Tracks state visits and provides novelty-based intrinsic rewards.

    Rewards:
      - Visiting a never-before-seen state: +novelty_bonus
      - Bonus decays as 1/sqrt(visit_count) — diminishing returns
    """

    def __init__(self, novelty_scale: float = 0.005):
        self._visit_counts: Dict[str, int] = {}
        self._total_steps: int = 0
        self._novelty_scale = novelty_scale

    def record_visit(self, grid: np.ndarray):
        """Record a visit to the state represented by this grid."""
        h = _hash_grid(grid)
        self._visit_counts[h] = self._visit_counts.get(h, 0) + 1
        self._total_steps += 1

    def novelty_bonus(self, grid: np.ndarray) -> float:
        """
        Compute novelty bonus for the current state.

        Returns novelty_scale / sqrt(visit_count).
        First visit: full bonus. Repeated visits: diminishing.
        """
        h = _hash_grid(grid)
        count = self._visit_counts.get(h, 0)
        if count == 0:
            return self._novelty_scale  # First visit
        return self._novelty_scale / math.sqrt(count + 1)

    def format_coverage(self) -> str:
        """Format exploration stats for injection into observations."""
        unique = len(self._visit_counts)
        if self._total_steps == 0:
            return ""
        return f"Explored: {unique} unique states in {self._total_steps} steps"

    def record_initial(self, grid: np.ndarray):
        """Record initial state without incrementing step counter."""
        h = _hash_grid(grid)
        self._visit_counts[h] = self._visit_counts.get(h, 0) + 1

    def reset(self):
        """Reset for a new episode (but NOT between levels)."""
        self._visit_counts.clear()
        self._total_steps = 0
