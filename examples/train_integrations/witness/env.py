"""
SkyRL-Gym environment for arc-witness-envs puzzle games.

Wraps Witness games (tw01-tw13) as a BaseTextEnv for GRPO training.
The model sees a text representation of the 64x64 grid and outputs
actions in <action>ACTION_ID</action> format.

Two observation modes:
  - "grid": Raw 16x16 downsampled color grid (baseline)
  - "ascii": Semantic ASCII encoding (requires semantic_ascii module)
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

# ── Ensure arc-witness-envs is importable ──────────────────────────────
_WITNESS_REPO = os.environ.get(
    "WITNESS_ENVS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "arc-witness-envs"),
)
_WITNESS_REPO = os.path.normpath(_WITNESS_REPO)
if _WITNESS_REPO not in sys.path:
    sys.path.insert(0, _WITNESS_REPO)

from arcengine import ActionInput, GameAction

# ── Game registry ──────────────────────────────────────────────────────
_GAME_REGISTRY = {
    f"tw{i:02d}": (f"environment_files.tw{i:02d}.tw{i:02d}", f"Tw{i:02d}")
    for i in range(1, 14)
}

_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "CONFIRM"}
_NAME_TO_ID = {v: k for k, v in _ACTION_NAMES.items()}
_ID_TO_GAME_ACTION = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
}


def _load_game_class(game_id: str):
    mod_name, cls_name = _GAME_REGISTRY[game_id]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _load_baselines(game_id: str) -> List[int]:
    meta_path = os.path.join(_WITNESS_REPO, "environment_files", game_id, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("baseline_actions", [])
    return []


def _frame_to_grid(frame_data) -> np.ndarray:
    """Extract 64x64 numpy array from game frame data."""
    if frame_data and frame_data.frame:
        arr = frame_data.frame[0]
        if isinstance(arr, np.ndarray):
            return arr
        return np.array(arr, dtype=np.int32)
    return np.zeros((64, 64), dtype=np.int32)


def _downsample(grid: np.ndarray, block: int = 4) -> np.ndarray:
    """Downsample 64x64 grid to (64/block)x(64/block) via majority vote."""
    h, w = grid.shape
    out_h, out_w = h // block, w // block
    out = np.zeros((out_h, out_w), dtype=np.int32)
    for r in range(out_h):
        for c in range(out_w):
            patch = grid[r * block:(r + 1) * block, c * block:(c + 1) * block].ravel()
            values, counts = np.unique(patch, return_counts=True)
            out[r, c] = values[counts.argmax()]
    return out


def _grid_to_text(grid: np.ndarray) -> str:
    """Convert a downsampled grid to a text representation with coordinates."""
    h, w = grid.shape
    # Column headers
    header = "   " + " ".join(f"{c:2d}" for c in range(w))
    lines = [header]
    for r in range(h):
        row_str = f"{r:2d} " + " ".join(f"{grid[r, c]:2d}" for c in range(w))
        lines.append(row_str)
    return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are playing a puzzle game on a grid. You see the board as a grid of color numbers.

Available actions:
  1 = UP    (move up)
  2 = DOWN  (move down)
  3 = LEFT  (move left)
  4 = RIGHT (move right)
  5 = CONFIRM (submit your solution)

Your goal: figure out the puzzle rules by exploring, then solve it.
Respond with ONLY your chosen action in this format: <action>NUMBER</action>
For example: <action>4</action> to move RIGHT."""


class WitnessEnv(BaseTextEnv):
    """
    BaseTextEnv wrapper for arc-witness-envs games.

    Config via env_extras:
      - game_id: str (default "tw01")
      - seed: int (default 0)
      - reward_mode: str (default "shaped")
      - max_steps_multiplier: int (default 3)
      - obs_mode: str (default "grid") — "grid" or "ascii"
    """

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.extras = extras
        self.max_turns = extras.get("max_turns", 1)

        # Game config from extras (set per-sample in parquet)
        self.game_id = extras.get("game_id", "tw01")
        self.seed = extras.get("seed", 0)
        self.reward_mode = extras.get("reward_mode", "shaped")
        self.max_steps_multiplier = extras.get("max_steps_multiplier", 3)
        self.obs_mode = extras.get("obs_mode", "grid")

        # Load game
        game_cls = _load_game_class(self.game_id)
        self.game = game_cls(seed=self.seed)
        self.baselines = _load_baselines(self.game_id)

        # Episode state
        self.step_count = 0
        self.levels_completed = 0
        self.level_index = 0
        self.total_levels = len(self.baselines) if self.baselines else getattr(self.game, '_win_score', 5)

        # Get initial frame via RESET
        self.last_frame_data = self.game.perform_action(
            ActionInput(id=GameAction.RESET), raw=True
        )
        self.last_grid = _frame_to_grid(self.last_frame_data)

        # Chat history
        self.chat_history: ConversationType = []

    def _baseline(self) -> int:
        if self.level_index < len(self.baselines):
            return self.baselines[self.level_index]
        return 30

    def _max_steps(self) -> int:
        return self._baseline() * self.max_steps_multiplier

    def _render_obs(self) -> str:
        """Render current frame as text observation."""
        ds = _downsample(self.last_grid)

        if self.obs_mode == "ascii":
            # Try to use semantic ASCII encoder if available
            try:
                from .semantic_ascii import encode_grid
                return encode_grid(self.last_grid)
            except ImportError:
                pass  # Fall back to grid mode

        # Default: raw color grid
        board = _grid_to_text(ds)
        meta = (
            f"Game: {self.game_id} | Level: {self.level_index}/{self.total_levels} | "
            f"Step: {self.step_count}/{self._max_steps()}"
        )
        return f"{meta}\n{board}"

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Set up initial prompt with system message and first observation."""
        initial_obs = self._render_obs()
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"New puzzle. Here is the initial board:\n\n{initial_obs}"},
        ]
        self.chat_history = chat.copy()
        return chat, {}

    def _parse_action(self, text: str) -> int:
        """Extract action ID from model output."""
        # Try <action>N</action> format
        matches = re.findall(r"<action>\s*(\d+)\s*</action>", text)
        if matches:
            return int(matches[-1])

        # Try <action>NAME</action> format
        name_matches = re.findall(r"<action>\s*(\w+)\s*</action>", text)
        if name_matches:
            name = name_matches[-1].upper()
            if name in _NAME_TO_ID:
                return _NAME_TO_ID[name]

        # Fallback: first bare digit 1-5 in the text
        bare = re.findall(r"\b([1-5])\b", text)
        if bare:
            return int(bare[-1])

        # Default: no-op (UP)
        return 1

    def _compute_reward(self, solved: bool, wrong_confirm: bool) -> float:
        if self.reward_mode == "sparse":
            return 1.0 if solved else 0.0
        elif self.reward_mode == "shaped":
            if solved:
                return 1.0
            elif wrong_confirm:
                return -0.1
            else:
                return -0.01
        elif self.reward_mode == "arc_score":
            if solved:
                return min(self._baseline() / max(self.step_count, 1), 1.0)
            elif wrong_confirm:
                return -0.1
            else:
                return 0.0
        return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        # Parse action
        action_id = self._parse_action(action)
        game_action = _ID_TO_GAME_ACTION.get(action_id, GameAction.ACTION1)

        # Execute in game
        prev_completed = (
            self.last_frame_data.levels_completed if self.last_frame_data else 0
        )
        self.last_frame_data = self.game.perform_action(
            ActionInput(id=game_action), raw=True
        )
        self.last_grid = _frame_to_grid(self.last_frame_data)
        self.step_count += 1

        curr_completed = (
            self.last_frame_data.levels_completed if self.last_frame_data else 0
        )

        # Determine outcome
        solved = curr_completed > prev_completed
        wrong_confirm = (action_id == 5) and not solved
        truncated = self.step_count >= self._max_steps()

        if solved:
            self.levels_completed = curr_completed
            self.level_index = curr_completed

        done = solved or truncated
        max_turns_reached = self.turns >= self.max_turns

        reward = self._compute_reward(solved, wrong_confirm)

        # Build message
        if solved:
            msg = f"Level solved in {self.step_count} steps! (baseline: {self._baseline()})"
        elif truncated:
            msg = f"Truncated at {self.step_count} steps."
        elif wrong_confirm:
            msg = "Wrong solution, try again."
        else:
            msg = ""

        if done or max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True, metadata={
                    "game_id": self.game_id,
                    "solved": solved,
                    "truncated": truncated,
                    "step_count": self.step_count,
                    "levels_completed": self.levels_completed,
                }
            )

        # Multi-turn: return observation for next action
        obs_text = self._render_obs()
        if msg:
            obs_text = f"{msg}\n\n{obs_text}"
        obs_text += f"\nYou played: {_ACTION_NAMES.get(action_id, '?')}. Choose your next action."

        new_obs = {"role": "user", "content": obs_text}
        self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=False,
            metadata={
                "game_id": self.game_id,
                "action_id": action_id,
                "step_count": self.step_count,
            },
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "levels_completed": self.levels_completed,
            "total_steps": self.step_count,
            "solved": self.levels_completed > 0,
        }

    def close(self):
        pass
