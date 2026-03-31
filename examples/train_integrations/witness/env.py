"""
SkyRL-Gym environment for arc-witness-envs puzzle games.

Wraps Witness games (tw01-tw13) as a BaseTextEnv for GRPO training.
The model sees a text representation of the 64x64 grid and outputs
actions in <action>ACTION_ID</action> format.

Observation modes:
  - "grid": Raw 16x16 downsampled color grid (baseline)
  - "ascii": Semantic ASCII encoding (requires semantic_ascii module)

Rules modes:
  - "rules_given": Inject known concept rules into system prompt
  - "rules_unknown": No rules provided, agent must discover them
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

# ── Lazy import for arcengine (arc-witness-envs) ─────────────────────
# Deferred to avoid ModuleNotFoundError when this module is imported
# in Ray workers that only need the class for aggregate_metrics.
_arcengine = None


def _ensure_arcengine():
    """Lazily import arcengine, adding arc-witness-envs to sys.path if needed."""
    global _arcengine
    if _arcengine is not None:
        return _arcengine
    # First try direct import (works if uv pip install -e . succeeded)
    try:
        import arcengine
        _arcengine = arcengine
        return _arcengine
    except ImportError:
        pass
    # Fallback: add repo to sys.path
    for candidate in [
        os.environ.get("WITNESS_ENVS_DIR", ""),
        os.path.expanduser("~/arc-witness-envs"),
        os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "arc-witness-envs",
        )),
    ]:
        if candidate and os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            try:
                import arcengine
                _arcengine = arcengine
                return _arcengine
            except ImportError:
                continue
    raise ImportError(
        "Cannot import arcengine. Ensure arc-witness-envs is installed "
        "(uv pip install -e $HOME/arc-witness-envs) or WITNESS_ENVS_DIR is set."
    )


# ── Game registry ──────────────────────────────────────────────────────
_GAME_REGISTRY = {
    f"tw{i:02d}": (f"environment_files.tw{i:02d}.tw{i:02d}", f"Tw{i:02d}")
    for i in range(1, 14)
}

_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "CONFIRM"}
_NAME_TO_ID = {v: k for k, v in _ACTION_NAMES.items()}

def _get_game_action(action_id: int):
    """Map action_id (1-5) to arcengine GameAction, lazy-loaded."""
    arc = _ensure_arcengine()
    mapping = {
        1: arc.GameAction.ACTION1,
        2: arc.GameAction.ACTION2,
        3: arc.GameAction.ACTION3,
        4: arc.GameAction.ACTION4,
        5: arc.GameAction.ACTION5,
    }
    return mapping.get(action_id, arc.GameAction.ACTION1)


def _load_game_class(game_id: str):
    _ensure_arcengine()  # ensure arc-witness-envs is on sys.path
    mod_name, cls_name = _GAME_REGISTRY[game_id]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _get_witness_repo() -> str:
    return os.environ.get(
        "WITNESS_ENVS_DIR",
        os.path.expanduser("~/arc-witness-envs"),
    )


def _load_baselines(game_id: str) -> List[int]:
    meta_path = os.path.join(_get_witness_repo(), "environment_files", game_id, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("baseline_actions", [])
    return []


# ── Ground truth rules (extracted from game source code) ──────────────
_RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")


def _load_ground_truth_rules(game_id: str) -> str:
    """Load ground truth rules for a game from the rules/ directory."""
    rules_path = os.path.join(_RULES_DIR, f"{game_id}.txt")
    if not os.path.exists(rules_path):
        return ""
    with open(rules_path) as f:
        return f.read().strip()


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
_BASE_SYSTEM_PROMPT = """You are playing a puzzle game on a grid.

Available actions:
  1 = UP    (move up)
  2 = DOWN  (move down)
  3 = LEFT  (move left)
  4 = RIGHT (move right)
  5 = CONFIRM (submit your solution)

{rules_section}Your goal: {goal}
Respond with ONLY your chosen action in this format: <action>NUMBER</action>
For example: <action>4</action> to move RIGHT."""


def _build_system_prompt(game_id: str, rules_mode: str, total_levels: int) -> str:
    """Build system prompt based on rules_mode."""
    if rules_mode == "rules_given":
        rules_text = _load_ground_truth_rules(game_id)
        rules_section = f"{rules_text}\n\n" if rules_text else ""
        goal = f"Use the rules above to solve all {total_levels} levels."
    else:
        rules_section = ""
        goal = f"Figure out the puzzle rules by exploring, then solve all {total_levels} levels."
    return _BASE_SYSTEM_PROMPT.format(rules_section=rules_section, goal=goal)


class WitnessEnv(BaseTextEnv):
    """
    BaseTextEnv wrapper for arc-witness-envs games.

    Config via env_extras:
      - game_id: str (default "tw01")
      - seed: int (default 0)
      - reward_mode: str (default "shaped")
      - max_steps_multiplier: int (default 3)
      - obs_mode: str (default "grid") — "grid" or "ascii"
      - rules_mode: str (default "rules_unknown") — "rules_given" or "rules_unknown"
      - harness_mode: bool (default False) — enable enhanced observations, exploration rewards, cross-level memory
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
        self.rules_mode = extras.get("rules_mode", "rules_unknown")
        self.max_levels = extras.get("max_levels", None)
        self.harness_mode = extras.get("harness_mode", False)

        # Load game
        game_cls = _load_game_class(self.game_id)
        self.game = game_cls(seed=self.seed)
        self.baselines = _load_baselines(self.game_id)

        # Episode state
        self.step_count = 0
        self.levels_completed = 0
        self.level_index = 0
        self.total_levels = len(self.baselines) if self.baselines else getattr(self.game, '_win_score', 5)
        if self.max_levels is not None:
            self.total_levels = min(self.total_levels, self.max_levels)

        # Get initial frame via RESET
        arc = _ensure_arcengine()
        self.last_frame_data = self.game.perform_action(
            arc.ActionInput(id=arc.GameAction.RESET), raw=True
        )
        self.last_grid = _frame_to_grid(self.last_frame_data)

        # Chat history
        self.chat_history: ConversationType = []

        # Harness: optional enhancement layer (only when harness_mode=true)
        # Sub-switches allow ablation of individual mechanisms
        self.harness = None
        if self.harness_mode:
            from .harness import WitnessHarness
            self.harness = WitnessHarness(
                game_id=self.game_id,
                initial_grid=self.last_grid,
                enable_action_mapper=extras.get("harness_action_mapper", True),
                enable_exploration=extras.get("harness_exploration", True),
                enable_memory=extras.get("harness_memory", True),
                enable_priors=extras.get("harness_priors", True),
            )

    def _baseline(self) -> int:
        if self.level_index < len(self.baselines):
            return self.baselines[self.level_index]
        return 30

    def _max_steps(self) -> int:
        return self._baseline() * self.max_steps_multiplier

    def _render_obs(self) -> str:
        """Render current frame as text observation."""
        meta = (
            f"Game: {self.game_id} | Level: {self.level_index}/{self.total_levels} | "
            f"Step: {self.step_count}/{self._max_steps()}"
        )

        if self.obs_mode == "ascii":
            try:
                from .semantic_ascii import encode_grid
                board = encode_grid(self.last_grid)
            except ImportError:
                ds = _downsample(self.last_grid)
                board = _grid_to_text(ds)
        else:
            ds = _downsample(self.last_grid)
            board = _grid_to_text(ds)

        obs = f"{meta}\n{board}"

        # Harness: append enriched observations (works with both grid and ascii)
        if self.harness:
            extra = self.harness.enrich_observation(self.last_grid, self.step_count, self.level_index)
            if extra:
                obs += "\n" + extra

        return obs

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Set up initial prompt with system message and first observation."""
        system_prompt = _build_system_prompt(self.game_id, self.rules_mode, self.total_levels)
        if self.harness:
            system_prompt += self.harness.get_system_prompt_addition()
        initial_obs = self._render_obs()
        chat = [
            {"role": "system", "content": system_prompt},
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
        game_action = _get_game_action(action_id)

        # Save previous grid for harness diff
        prev_grid = self.last_grid.copy() if self.harness else None

        # Execute in game
        arc = _ensure_arcengine()
        prev_completed = (
            self.last_frame_data.levels_completed if self.last_frame_data else 0
        )
        self.last_frame_data = self.game.perform_action(
            arc.ActionInput(id=game_action), raw=True
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
            level_steps = self.step_count  # save before reset
            self.levels_completed = curr_completed
            self.level_index = curr_completed
            # Reset step counter for next level
            self.step_count = 0

        # Harness: update state + track level completion
        if self.harness:
            self.harness.on_step(prev_grid, action_id, self.last_grid)
            if solved:
                self.harness.on_level_solved(self.level_index - 1, level_steps)

        # Episode ends when: all levels cleared, current level truncated, or max_turns reached
        all_cleared = self.levels_completed >= self.total_levels
        done = all_cleared or truncated
        max_turns_reached = self.turns >= self.max_turns

        reward = self._compute_reward(solved, wrong_confirm)

        # Harness: add intrinsic reward bonus
        if self.harness:
            reward += self.harness.compute_bonus_reward(self.last_grid, action_id, solved)

        # Build message
        if all_cleared:
            msg = f"All {self.total_levels} levels cleared! Total turns: {self.turns}"
        elif solved:
            msg = (f"Level {self.level_index - 1} solved in {self.turns} turns! "
                   f"(baseline: {self.baselines[self.level_index - 1] if self.level_index - 1 < len(self.baselines) else '?'}). "
                   f"Starting level {self.level_index}/{self.total_levels}.")
        elif truncated:
            msg = f"Level {self.level_index} truncated at {self.step_count} steps."
        elif wrong_confirm:
            msg = "Wrong solution, try again."
        else:
            msg = ""

        if done or max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True, metadata={
                    "game_id": self.game_id,
                    "all_cleared": all_cleared,
                    "truncated": truncated,
                    "step_count": self.step_count,
                    "levels_completed": self.levels_completed,
                    "total_levels": self.total_levels,
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
            "total_levels": self.total_levels,
            "level_progress": self.levels_completed / max(self.total_levels, 1),
            "total_turns": self.turns,
            "solved_any": 1.0 if self.levels_completed > 0 else 0.0,
            "all_cleared": 1.0 if self.levels_completed >= self.total_levels else 0.0,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes for wandb logging."""
        if not metrics:
            return {}

        levels = [m.get("levels_completed", 0) for m in metrics]
        progress = [m.get("level_progress", 0.0) for m in metrics]
        solved = [m.get("solved_any", 0.0) for m in metrics]
        cleared = [m.get("all_cleared", 0.0) for m in metrics]

        return {
            "avg_levels_completed": sum(levels) / len(levels),
            "max_levels_completed": max(levels),
            "avg_level_progress": sum(progress) / len(progress),
            "solve_rate": sum(solved) / len(solved),
            "all_cleared_rate": sum(cleared) / len(cleared),
            "num_episodes": len(metrics),
        }

    def close(self):
        pass
