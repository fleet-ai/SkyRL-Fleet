"""
SkyRL-Gym environment for PrimitivBench generated games (mini-flywheel, Sprint 2).

Wraps any pilot-generated `PrimitivBenchGame` (reset/step/render/valid_actions/
is_won — the pilot submission format) as a BaseTextEnv for GRPO training.

Design notes (vs the witness integration):
  - GENERIC text harness by design: observation = game.render("text") + an
    enumerated valid-action list; NO witness semantic-ASCII, no game-specific
    perception. The same recipe must serve all generators' games (benchmark
    fairness) and matches the proxy-battery probe interface.
  - Action format: <action>N</action> where N indexes the CURRENT valid-action
    list (action objects vary per game — ints, strings, tuples — so indexing
    is the only universal contract).
  - Instance multiplication: one parquet row per (game, seed); reset(seed)
    gives a distinct instance (verified 2026-06-09).

Config via env_extras (set per-row in parquet):
  - game_name: str  e.g. "PB-pilot-001__gemini-2.5-pro"  (required)
  - seed: int (default 0)
  - reward_mode: "shaped" (default) | "sparse"
  - max_turns: int (default 30)
Games dir resolution: $PRIMITIVBENCH_GAMES_DIR, else ./games next to this file.
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any, Dict, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

RENDER_CAP = 1600   # chars of render text per observation
ACTION_CAP = 30     # max actions enumerated per turn

_GAME_CLS_CACHE: Dict[str, Any] = {}


def _games_dir() -> str:
    env = os.environ.get("PRIMITIVBENCH_GAMES_DIR", "")
    if env and os.path.isdir(env):
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")


def _load_game_class(game_name: str):
    if game_name in _GAME_CLS_CACHE:
        return _GAME_CLS_CACHE[game_name]
    game_py = os.path.join(_games_dir(), game_name, "game.py")
    if not os.path.exists(game_py):
        raise FileNotFoundError(
            f"game.py not found for {game_name!r} under {_games_dir()} "
            f"(set PRIMITIVBENCH_GAMES_DIR or run prepare_primitivbench_dataset.py first)")
    spec = importlib.util.spec_from_file_location(f"pbgame_{re.sub(r'[^A-Za-z0-9_]', '_', game_name)}", game_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = mod.PrimitivBenchGame
    _GAME_CLS_CACHE[game_name] = cls
    return cls


_SYSTEM_PROMPT = (
    "You are playing an unfamiliar puzzle game. The rules, the goal and the win "
    "condition are NOT given — you must infer them from observations as you play.\n"
    "Each turn you receive the current game state and a numbered list of valid "
    "actions. Think briefly, then choose ONE action.\n"
    "Output format: end your reply with <action>N</action> where N is the number "
    "of your chosen action."
)


class PrimitivBenchEnv(BaseTextEnv):
    """BaseTextEnv wrapper for pilot-generated PrimitivBench games."""

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.extras = extras
        self.game_name = extras.get("game_name")
        if not self.game_name:
            raise ValueError("env_extras must include 'game_name'")
        self.seed = int(extras.get("seed", 0))
        self.reward_mode = extras.get("reward_mode", "shaped")
        self.max_turns = int(extras.get("max_turns", 30))

        cls = _load_game_class(self.game_name)
        self.game = cls()
        self.game.reset(seed=self.seed)

        self.step_count = 0
        self.cur_actions: list = []
        self.chat_history: ConversationType = []

    # ---- observation -------------------------------------------------

    def _render_text(self) -> str:
        try:
            r = self.game.render(mode="text")
            return str(r)[:RENDER_CAP]
        except Exception as e:
            return f"(render unavailable: {type(e).__name__})"

    def _obs_block(self) -> str:
        try:
            self.cur_actions = list(self.game.valid_actions())[:ACTION_CAP]
        except Exception:
            self.cur_actions = []
        lines = [self._render_text(), "", "Valid actions:"]
        if self.cur_actions:
            lines += [f"{i}: {a!r}" for i, a in enumerate(self.cur_actions)]
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("Reply with your reasoning, then <action>N</action>.")
        return "\n".join(lines)

    # ---- protocol ----------------------------------------------------

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        chat = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"New game (turn budget {self.max_turns}). Initial state:\n\n{self._obs_block()}"},
        ]
        self.chat_history = chat.copy()
        return chat, {"game_name": self.game_name, "seed": self.seed}

    def _parse_action_index(self, text: str) -> int:
        m = re.findall(r"<action>\s*(\d+)\s*</action>", text)
        if m:
            return int(m[-1])
        bare = re.findall(r"\b(\d{1,2})\b", text)
        if bare:
            return int(bare[-1])
        return 0

    def _reward(self, won: bool, invalid: bool, done: bool, env_reward: float) -> float:
        if self.reward_mode == "sparse":
            return 1.0 if won else 0.0
        # shaped (default): win bonus dominates; small step cost; invalid pick penalty;
        # clipped pass-through of the game's own per-step reward signal.
        r = 0.0
        if won:
            r += 1.0
        if invalid:
            r -= 0.05
        r -= 0.01
        r += max(-0.05, min(0.05, float(env_reward or 0.0)))
        return r

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.step_count += 1

        idx = self._parse_action_index(action)
        invalid = not (0 <= idx < len(self.cur_actions))
        won = False
        done = False
        env_reward = 0.0

        if self.cur_actions:
            game_action = self.cur_actions[min(max(idx, 0), len(self.cur_actions) - 1)]
            try:
                _, env_reward, done, _ = self.game.step(game_action)
            except Exception:
                done = True  # game crashed → terminal, no win
            if done:
                try:
                    won = bool(self.game.is_won())
                except Exception:
                    won = False
        else:
            done = True  # no valid actions → terminal

        out_of_turns = self.step_count >= self.max_turns
        if out_of_turns and not done:
            done = True
            try:
                won = bool(self.game.is_won())
            except Exception:
                won = False

        reward = self._reward(won, invalid, done, env_reward)
        metadata = {
            "game_name": self.game_name,
            "seed": self.seed,
            "step": self.step_count,
            "won": won,
            "invalid_action": invalid,
        }

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True, metadata=metadata)

        obs_msg = {"role": "user", "content": self._obs_block()}
        self.chat_history.append({"role": "assistant", "content": action})
        self.chat_history.append(obs_msg)
        return BaseTextEnvStepOutput(
            observations=[obs_msg], reward=reward, done=False, metadata=metadata)

    def close(self):
        self.game = None
