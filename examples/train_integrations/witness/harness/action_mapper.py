"""
Action semantics discovery — extracted from arc-witness-agent.

Discovers what each ACTION1-5 does by observing before/after frames:
  - Direction inference (UP/DOWN/LEFT/RIGHT)
  - Inverse pair detection (UP↔DOWN, LEFT↔RIGHT)
  - Noop rate (how often an action does nothing)

No LLM required — pure algorithmic analysis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# Cursor color in Witness games (yellow = color index 11)
_CURSOR_COLOR = 11
_BLOCK_SIZE = 4  # 64/16 grid cells


def _find_cursor_cell(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """Find the grid cell (row, col) containing the cursor color."""
    h, w = grid.shape
    for r in range(h // _BLOCK_SIZE):
        for c in range(w // _BLOCK_SIZE):
            patch = grid[
                r * _BLOCK_SIZE:(r + 1) * _BLOCK_SIZE,
                c * _BLOCK_SIZE:(c + 1) * _BLOCK_SIZE,
            ]
            if _CURSOR_COLOR in patch:
                return (r, c)
    return None


class ActionMapper:
    """
    Learns action semantics from (before_grid, action, after_grid) observations.

    After enough observations, can report:
      - Which action maps to which direction
      - Which actions are inverse pairs
      - Noop rates per action
    """

    def __init__(self):
        # (action_id) -> list of (delta_row, delta_col) movements
        self._movements: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        # (action_id) -> count of times action caused no change
        self._noops: Dict[int, int] = defaultdict(int)
        # (action_id) -> total observations
        self._counts: Dict[int, int] = defaultdict(int)
        # Cache for discovered semantics
        self._directions: Dict[int, str] = {}
        self._inverse_pairs: List[Tuple[int, int]] = []
        self._dirty = True  # needs recomputation

    def record(self, prev_grid: np.ndarray, action_id: int, new_grid: np.ndarray):
        """Record one (state, action, next_state) observation."""
        self._counts[action_id] += 1

        prev_pos = _find_cursor_cell(prev_grid)
        new_pos = _find_cursor_cell(new_grid)

        if prev_pos is None or new_pos is None:
            self._noops[action_id] += 1
            return

        dr = new_pos[0] - prev_pos[0]
        dc = new_pos[1] - prev_pos[1]

        if dr == 0 and dc == 0:
            self._noops[action_id] += 1
        else:
            self._movements[action_id].append((dr, dc))

        self._dirty = True

    def _recompute(self):
        """Infer directions and inverse pairs from accumulated observations."""
        if not self._dirty:
            return

        self._directions = {}
        self._inverse_pairs = []

        # For each action, find dominant direction
        action_dirs: Dict[int, Tuple[int, int]] = {}
        for action_id, moves in self._movements.items():
            if not moves:
                continue
            # Majority vote on direction
            counter = Counter(moves)
            dominant, count = counter.most_common(1)[0]
            if count >= 2 or (count == 1 and len(moves) == 1):
                action_dirs[action_id] = dominant

        # Map (dr, dc) to direction name
        dir_names = {
            (-1, 0): "UP",
            (1, 0): "DOWN",
            (0, -1): "LEFT",
            (0, 1): "RIGHT",
        }

        # Assign directions (constraint: each direction used at most once)
        used_dirs: Set[str] = set()
        for action_id, (dr, dc) in sorted(action_dirs.items()):
            name = dir_names.get((dr, dc))
            if name and name not in used_dirs:
                self._directions[action_id] = name
                used_dirs.add(name)

        # Detect inverse pairs
        inverse_map = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        assigned = {v: k for k, v in self._directions.items()}
        for d1, d2 in [("UP", "DOWN"), ("LEFT", "RIGHT")]:
            if d1 in assigned and d2 in assigned:
                self._inverse_pairs.append((assigned[d1], assigned[d2]))

        self._dirty = False

    def format_semantics(self) -> str:
        """Format discovered semantics as text for injection into observations."""
        self._recompute()

        if not self._directions:
            return ""

        parts = []

        # Direction mapping
        dir_strs = []
        for aid in sorted(self._directions):
            dir_strs.append(f"{aid}={self._directions[aid]}")
        # Add ACTION5 if not assigned a direction
        if 5 not in self._directions:
            dir_strs.append("5=CONFIRM")
        parts.append("Actions: " + "  ".join(dir_strs))

        # Inverse pairs
        if self._inverse_pairs:
            pair_strs = [
                f"({self._directions.get(a, f'A{a}')}↔{self._directions.get(b, f'A{b}')})"
                for a, b in self._inverse_pairs
            ]
            parts.append("Inverse pairs: " + " ".join(pair_strs))

        return "\n".join(parts)

    def discovery_bonus(self, action_id: int) -> float:
        """Return a one-time bonus for discovering new action semantics."""
        self._recompute()
        # Small bonus for first observation of each action
        if self._counts[action_id] == 1:
            return 0.01
        return 0.0

    def get_directions(self) -> Dict[int, str]:
        """Return action_id -> direction_name mapping."""
        self._recompute()
        return dict(self._directions)
