"""B7 RL — WitnessAgentEnv: SkyRL env that drives arc-witness-agent in bridged mode.

Replaces the v3-era WitnessEnv (in env.py with light-harness) with a thin
wrapper around AgentRolloutWrapper. Each env.step() advances ONE ORAI tick;
the trainer's generation becomes the agent's ORAI response.

Trajectory unit: one (game, seed) game playthrough across max_levels levels.
Memory persists across levels within the trajectory (rule transfer ↑).

Reward per step:
  - Geometric (waypoint / region / distance / valid_move) per game-action delta
  - Outcome (level solved +1.0; full game complete bonus +0.5)
  - Step penalty (-0.005 per game action, -0.05 per ORAI tick)
  See `agent/runtime/process_reward.py` for the formula.

Reference: artifacts/.../reports/2026-05-09_b7_rl_training_design.md §3
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

# Allow importing the local agent_wrapper.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class WitnessAgentEnv(BaseTextEnv):
    """SkyRL env wrapping arc-witness-agent's full ORAI loop.

    Required extras:
        game_id (str): tw01..tw13
        seed (int): per-task seed
        max_levels (int): max levels to play in one trajectory
        agent_config (dict): config loaded from arc-witness-agent's config.yaml
                            (or built programmatically). vLLM provider is
                            forced internally — base_url comes from
                            VLLM_BASE_URL env var or config.

    Optional extras:
        max_orai_steps (int): hard cap on ORAI ticks per trajectory (default 30)

    The trainer's prompt for init() is ignored — the agent generates its own
    first ORAI prompt. This is a documented protocol diff from the v3
    WitnessEnv (which used the trainer's prompt as the initial obs).
    """

    def __init__(self, env_config: Any, extras: Dict[str, Any] = None):
        super().__init__(env_config)
        extras = extras or {}

        self.game_id: str = extras["game_id"]
        self.seed: int = int(extras.get("seed", 0))
        self.max_levels: int = int(extras.get("max_levels", 5))
        self.max_orai_steps: int = int(extras.get("max_orai_steps", 30))
        self.agent_config: Dict[str, Any] = extras.get("agent_config") or self._load_default_agent_config()

        # vLLM endpoint (the policy server colocated with trainer).
        self.vllm_base_url: str = (
            extras.get("vllm_base_url")
            or os.environ.get("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )

        # Lazy: built in init().
        self._wrapper = None
        self._chat_history: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._last_event = None
        self._cumulative_reward: float = 0.0
        self._cumulative_breakdown: Dict[str, float] = {}
        self._levels_completed_view: int = 0

    # ─────────────────────────── BaseTextEnv API ────────────────────────────

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Spawn agent rollout in background; return first ORAI prompt as chat.

        The `prompt` arg from prepare_witness_dataset is ignored (agent
        generates its own first prompt from game state). Chat returned has
        agent's system + user message; trainer generates the assistant turn.
        """
        from agent_wrapper import AgentRolloutWrapper  # local module

        self._wrapper = AgentRolloutWrapper(
            game_id=self.game_id,
            seed=self.seed,
            agent_config=self.agent_config,
            vllm_base_url=self.vllm_base_url,
            max_levels=self.max_levels,
            mode="bridged",
        )

        # Start agent thread; block until first ORAI prompt is ready.
        first_prompt = self._wrapper.start_bridged_rollout()
        system_prompt, messages = first_prompt
        if not messages:
            # Agent finished without any ORAI call (mechanical solve via SCOUT).
            # Return a stub chat; step() will return done=True immediately.
            chat = [
                {"role": "system", "content": "Game completed without ORAI."},
                {"role": "user", "content": "Done."},
            ]
        else:
            user_content = messages[-1]["content"]
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        self._chat_history = chat.copy()
        return chat, {"game_id": self.game_id, "seed": self.seed}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Feed trainer's generation as agent's ORAI response; advance one tick."""
        if self._wrapper is None:
            raise RuntimeError("step() before init()")

        self._step_count += 1

        next_prompt, event = self._wrapper.feed_completion_and_get_next(
            completion=action,
            timeout=120.0,
        )
        self._last_event = event
        # Track cumulative reward for metrics.
        self._cumulative_reward += event.reward
        for k, v in event.reward_breakdown.items():
            self._cumulative_breakdown[k] = self._cumulative_breakdown.get(k, 0.0) + v
        self._levels_completed_view = max(self._levels_completed_view, event.level)

        # Determine terminal condition.
        done = (
            next_prompt is None
            or self._step_count >= self.max_orai_steps
        )

        # Construct next observation.
        if done:
            # Final wait for agent thread + collect metrics.
            try:
                summary = self._wrapper.join_bridged_rollout(timeout=30.0)
                metrics = summary.get("metrics")
                final_levels = (
                    len(metrics.levels_completed) if metrics and hasattr(metrics, "levels_completed") and isinstance(metrics.levels_completed, list)
                    else self._levels_completed_view
                )
            except Exception as e:
                final_levels = self._levels_completed_view
                metrics = None

            obs_text = (
                f"Trajectory complete. Levels completed: {final_levels}/{self.max_levels}. "
                f"Total reward: {self._cumulative_reward:.3f}."
            )
            metadata = {
                "done": True,
                "levels_completed": final_levels,
                "total_orai_steps": self._step_count,
                "cumulative_reward": self._cumulative_reward,
                "reward_breakdown": dict(self._cumulative_breakdown),
                "step_reward": event.reward,
                "step_reward_breakdown": event.reward_breakdown,
            }
            new_obs = [{"role": "user", "content": obs_text}]
            self._chat_history.extend([
                {"role": "assistant", "content": action},
                new_obs[0],
            ])
            return BaseTextEnvStepOutput(
                observations=new_obs,
                reward=event.reward,
                done=True,
                metadata=metadata,
            )

        # Non-terminal: emit next ORAI prompt as the new user observation.
        sys_prompt, messages = next_prompt
        next_user_content = messages[-1]["content"] if messages else ""
        new_obs = [{"role": "user", "content": next_user_content}]
        self._chat_history.extend([
            {"role": "assistant", "content": action},
            new_obs[0],
        ])
        return BaseTextEnvStepOutput(
            observations=new_obs,
            reward=event.reward,
            done=False,
            metadata={
                "level": event.level,
                "step": event.step,
                "step_reward": event.reward,
                "step_reward_breakdown": event.reward_breakdown,
            },
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Per-trajectory metrics for wandb logging."""
        return {
            f"{self.game_id}/cumulative_reward": self._cumulative_reward,
            f"{self.game_id}/levels_completed": self._levels_completed_view,
            f"{self.game_id}/orai_steps": self._step_count,
            **{
                f"{self.game_id}/reward_{k}": v
                for k, v in self._cumulative_breakdown.items()
            },
        }

    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-trajectory metrics into batch summary."""
        if not metrics_list:
            return {}
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        agg = {}
        for k in all_keys:
            vals = [m[k] for m in metrics_list if k in m]
            if vals:
                agg[f"{k}_mean"] = sum(vals) / len(vals)
                agg[f"{k}_max"] = max(vals)
                agg[f"{k}_min"] = min(vals)
        return agg

    def close(self):
        if self._wrapper is not None:
            self._wrapper.cleanup()
            self._wrapper = None

    @staticmethod
    def _load_default_agent_config() -> Dict[str, Any]:
        """Load arc-witness-agent's config.yaml from ARC_WITNESS_AGENT_DIR.

        Falls back to minimal config if file missing — the wrapper will still
        work but won't have full agent behavior (skill_bank etc. won't load).
        """
        agent_dir = os.environ.get("ARC_WITNESS_AGENT_DIR", os.path.expanduser("~/arc-witness-agent"))
        cfg_path = os.path.join(agent_dir, "config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        # Minimal fallback config — only fields the agent absolutely needs.
        return {
            "agent": {"max_steps_per_level": 500, "max_total_steps": 5000},
            "stages": {
                "reconnaissance": {"budget_pct": 0.30},
                "rule_synthesis": {"budget_pct": 0.10},
                "planning": {"budget_pct": 0.50},
                "crystallization": {"budget_pct": 0.10},
            },
            "llm": {"provider": "vllm", "model": "policy-v5"},
            "memory": {},
            "skill_bank": {},
            "paths": {"system_prompt": "prompts/core_knowledge.txt"},
            "semantic_ascii": {"enabled": True, "mllm_enabled": False},
        }
