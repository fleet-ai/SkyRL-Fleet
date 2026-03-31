"""
Lightweight cross-level memory for RL training.

Stores knowledge discovered during earlier levels and injects it
into observations for later levels. Enables transfer without re-exploration.

No LLM required — stores action history and simple pattern observations.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class LevelMemory:
    """
    Cross-level knowledge accumulator.

    After solving a level, stores:
      - The winning action sequence
      - Number of steps taken
      - Any patterns observed (action semantics, etc.)

    Before starting the next level, injects a summary into the prompt.
    """

    def __init__(self):
        self._solved_levels: Dict[int, dict] = {}
        self._knowledge_notes: List[str] = []

    def store_level_result(
        self,
        level_index: int,
        action_history: List[int],
        steps: int,
    ):
        """Record a successful level completion."""
        self._solved_levels[level_index] = {
            "actions": action_history[-min(len(action_history), 20):],  # last 20 actions
            "steps": steps,
        }

    def add_note(self, note: str):
        """Add a knowledge note (e.g., 'ACTION5 resets the board')."""
        if note not in self._knowledge_notes:
            self._knowledge_notes.append(note)

    def format_known_rules(self) -> str:
        """Format accumulated knowledge for injection into observations."""
        parts = []

        if self._solved_levels:
            solved = sorted(self._solved_levels.keys())
            parts.append(f"Solved levels: {solved}")

        if self._knowledge_notes:
            parts.append("Known patterns:")
            for note in self._knowledge_notes[-5:]:  # last 5 notes
                parts.append(f"  - {note}")

        return "\n".join(parts) if parts else ""

    def reset(self):
        """Full reset (new game)."""
        self._solved_levels.clear()
        self._knowledge_notes.clear()
