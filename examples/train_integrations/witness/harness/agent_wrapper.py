"""B7 RL — wraps arc-witness-agent's AgentCore for use as the inner harness.

Replaces the v3-era light-harness (action_mapper/exploration/memory in this dir)
with the full ORAI agent. Designed for use inside SkyRL's WitnessEnv.step()
during GRPO rollouts.

Architecture (Path B per design doc 2026-05-09_b7_rl_training_design.md §3):
  trainer ─► WitnessEnv.step() ─► AgentRolloutWrapper.run_full_trajectory()
                                     │
                                     ▼ (blocking)
                                  AgentCore.run(game, game_id, seed)
                                     │
                                     ▼ each ORAI ── intercepted ──┐
                                  vLLM (the policy model)          │
                                     │                             │
                                     ▼ response                    │
                                  agent advances                   │
                                                                   ▼
                              (system, user, response, pre/post path) recorded
                                     │
              ┌──────── after run() returns ────────┐
              ▼                                     ▼
  TraceCollector reflections     interceptor records
              │                                     │
              └──────► combined into Events ────────┘
                              │
                              ▼ (s, a, r) tuples for GRPO

Design choice: monkey-patch `agent._planning._meta._llm` instead of
`agent._llm` so we capture only ORAI calls (not MLLM eager perception
or T5 plan calls). Each ORAI = one RL step.

Reward computed post-hoc from ground-truth env state snapshots taken
at each ORAI call site (BEFORE the LLM call returns, so we capture
the state agent is reasoning about).
"""

from __future__ import annotations

import copy
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add arc-witness-agent to path so we can import AgentCore et al.
# Trainer / VM yaml is responsible for cloning this repo at the right path.
_AGENT_REPO = os.environ.get(
    "ARC_WITNESS_AGENT_DIR",
    os.path.expanduser("~/arc-witness-agent"),
)
if _AGENT_REPO not in sys.path and os.path.isdir(_AGENT_REPO):
    sys.path.insert(0, _AGENT_REPO)


@dataclass
class OraiCallRecord:
    """One intercepted ORAI LLM call + game-state snapshot at call time."""
    system_prompt: str
    user_message: str
    response: str
    pre_path: List[Tuple[int, int]]
    pre_level_index: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class RolloutEvent:
    """One RL step ready for GRPO trainer."""
    system_prompt: str
    user_message: str
    response: str
    level: int
    step: int
    reward: float
    reward_breakdown: Dict[str, float]


class _PolicyBridge:
    """Sync between trainer's env.step() (main thread) and agent's LLM call
    (background thread). One ORAI tick = one prompt/response exchange.

    Lifecycle per trajectory:
      1. background thread: AgentCore.run() begins
      2. background: LLM call → bridge.request_completion(prompt) blocks
      3. main: bridge.next_prompt() returns the prompt, env.step exposes it
      4. trainer generates response from prompt
      5. main: bridge.feed_response(response) unblocks the agent's call
      6. background: agent advances; loop or finish
      7. on agent done: bridge.signal_done() unblocks any waiter
    """

    def __init__(self):
        self._prompt: Optional[Tuple[str, list]] = None
        self._response: Optional[str] = None
        self._has_prompt = threading.Event()
        self._has_response = threading.Event()
        self._done = False

    # Called by agent's LLM interceptor (background thread).
    def request_completion(self, system_prompt: str, messages: list) -> str:
        if self._done:
            # Agent shouldn't be calling after done — return empty as graceful no-op.
            return ""
        self._prompt = (system_prompt, messages)
        self._has_prompt.set()
        # Wait for trainer to feed a response.
        self._has_response.wait()
        self._has_response.clear()
        resp = self._response
        self._response = None
        return resp or ""

    # Called by env.step() (main thread).
    def next_prompt(self, timeout: Optional[float] = None) -> Optional[Tuple[str, list]]:
        """Block until agent submits next prompt OR rollout finishes.
        Returns None if trajectory is done."""
        ready = self._has_prompt.wait(timeout=timeout)
        if not ready:
            return None  # timeout; treat as no-prompt-yet
        self._has_prompt.clear()
        if self._done:
            return None
        return self._prompt

    def feed_response(self, response: str) -> None:
        self._response = response
        self._has_response.set()

    def signal_done(self) -> None:
        self._done = True
        # Unblock any waiting next_prompt().
        self._has_prompt.set()
        # Unblock any in-flight request_completion (graceful shutdown).
        self._has_response.set()


class AgentRolloutWrapper:
    """One-shot wrapper: construct, call run_full_trajectory(), discard.

    Per-rollout state (memory, traces, snapshots) is fully isolated. Multiple
    instances can run concurrently in different processes.
    """

    def __init__(
        self,
        game_id: str,
        seed: int,
        agent_config: Dict[str, Any],
        vllm_base_url: str = "http://localhost:8000/v1",
        max_levels: int = 5,
        mode: str = "standalone",  # "standalone" | "bridged"
    ):
        if mode not in ("standalone", "bridged"):
            raise ValueError(f"mode must be 'standalone' or 'bridged', got {mode!r}")
        self.game_id = game_id
        self.seed = seed
        self.max_levels = max_levels
        self.mode = mode
        self._bridge: Optional[_PolicyBridge] = _PolicyBridge() if mode == "bridged" else None
        self._thread: Optional[threading.Thread] = None
        self.metrics = None

        # Lazy imports — these come from the arc-witness-agent repo on path.
        from agent.core import AgentCore                        # noqa: E402
        from training.trace_collector import TraceCollector      # noqa: E402
        from agent.runtime.memory_dir import setup_memory_dir    # noqa: E402

        # Force LLM provider to vLLM, point at the policy server.
        # Also raise budget gates so RL rollout never has ORAI suppressed.
        cfg = copy.deepcopy(agent_config)
        cfg.setdefault("llm", {})
        cfg["llm"]["provider"] = "vllm"
        cfg["llm"]["base_url"] = vllm_base_url
        cfg["llm"].setdefault("temperature", 0.7)  # policy sampling temp
        # Soften the LLM-cost circuit breaker: in RL the policy is ours,
        # cost is meaningless. See design doc §1.4 detail #1.
        cfg.setdefault("llm_cost", {})
        cfg["llm_cost"]["max_total_usd"] = 1e9

        # Per-rollout ephemeral memory dir. Auto-cleaned at process exit.
        memory_dir, cleanup = setup_memory_dir(
            memory_dir=None,
            save_memory_to=None,
            persist=False,
            read_only=False,
        )
        self._memory_cleanup = cleanup
        self.memory_dir = memory_dir

        # Trace collector for ORAI / step record.
        self.trace = TraceCollector(output_dir=memory_dir, game_id=game_id)

        # Construct agent. base_dir points to the agent repo so prompt files
        # (core_knowledge.txt etc.) resolve correctly.
        self.agent = AgentCore(
            cfg,
            base_dir=_AGENT_REPO,
            trace_collector=self.trace,
            memory_dir=memory_dir,
        )

        # Game object — created lazily inside run_full_trajectory so the
        # interceptor can read game state.
        self.game = None
        self._records: List[OraiCallRecord] = []

        # Install ORAI LLM interceptor. We patch _planning._meta._llm because
        # that's the LLM client used by MetaReasoningEngine.reflect() — the
        # ORAI call site that produces RL "actions". Other LLM calls (MLLM
        # eager perception, T5 plan) bypass this interceptor.
        self._install_orai_interceptor()

    def _install_orai_interceptor(self):
        meta = self.agent._planning._meta
        original_llm = meta._llm

        wrapper_self = self

        class _Interceptor:
            """Records (system, user, response) + path snapshot per call.

            Forwards all other attributes to the wrapped LLMClient so existing
            agent code (token counting, model name access, etc.) keeps working.
            """

            def __init__(self, base):
                self._base = base

            def call(self, system_prompt, messages, **kwargs):
                # Pre-call snapshot — state the agent reasons over.
                game = wrapper_self.game
                pre_path = list(getattr(game, "_path", [])) if game else []
                pre_level = getattr(wrapper_self.agent, "_level_index", 0)

                # Bridged mode: hand prompt to trainer (main thread), block
                # for response. Standalone mode: forward to real vLLM directly.
                if wrapper_self._bridge is not None:
                    response = wrapper_self._bridge.request_completion(system_prompt, messages)
                else:
                    response = self._base.call(system_prompt, messages, **kwargs)

                # Record. The user message is the last message in the list;
                # ORAI ALWAYS uses single-turn [{"role": "user", "content": ...}].
                user_msg = messages[-1]["content"] if messages else ""
                wrapper_self._records.append(OraiCallRecord(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    response=response,
                    pre_path=pre_path,
                    pre_level_index=pre_level,
                ))
                return response

            def __getattr__(self, name):
                return getattr(self._base, name)

        meta._llm = _Interceptor(original_llm)

    def run_full_trajectory(self) -> Dict[str, Any]:
        """Block until the agent completes the (game, seed) playthrough.

        Returns:
            {
                "events": [RolloutEvent, ...],   # one per ORAI call
                "metrics": GameMetrics,           # totals + per-level
                "n_orai_calls": int,
                "memory_dir": str,
            }
        """
        from run_agent import load_game  # lazy import from agent repo

        from agent.runtime.process_reward import (    # noqa: E402
            compute_geometric_reward,
            compute_outcome_reward,
        )

        self.game = load_game(self.game_id, self.seed)
        metrics = self.agent.run(
            self.game,
            self.game_id,
            seed=self.seed,
            max_levels=self.max_levels,
        )

        # Build events. Each ORAI record gets a reward computed from the path
        # delta to the NEXT record (or the final game state for the last one).
        events: List[RolloutEvent] = []
        records = self._records
        if not records:
            return {
                "events": events,
                "metrics": metrics,
                "n_orai_calls": 0,
                "memory_dir": self.memory_dir,
            }

        # Tail snapshot from final game state.
        final_path = list(getattr(self.game, "_path", []))
        final_level = getattr(self.agent, "_level_index", 0)

        # Get a corresponding ReflectionTrace for level/step metadata if order matches.
        # ReflectionTrace records appear in same order as ORAI calls (1:1).
        refls = list(self.trace._reflections)

        for i, rec in enumerate(records):
            # `next` snapshot = next record's pre_path (state the agent saw NEXT
            # ORAI tick). For the final record, use the trajectory's final state.
            if i + 1 < len(records):
                next_path = records[i + 1].pre_path
                next_level = records[i + 1].pre_level_index
            else:
                next_path = final_path
                next_level = final_level

            # Process reward (geometric).
            geo_total, geo_breakdown = compute_geometric_reward(
                self.game,
                prev_path=rec.pre_path,
                new_path=next_path,
                is_valid_move=True,  # Phase 1 approximation: per-step validity
                                     # not tracked; trainer can refine later.
            )
            # Outcome reward (level transitions).
            total_levels = self.max_levels  # max_levels in trajectory
            out_total, out_breakdown = compute_outcome_reward(
                rec.pre_level_index,
                next_level,
                total_levels,
            )

            reward = geo_total + out_total
            breakdown = {**geo_breakdown, **out_breakdown}

            level = refls[i].level if i < len(refls) else rec.pre_level_index
            step = refls[i].step if i < len(refls) else 0

            events.append(RolloutEvent(
                system_prompt=rec.system_prompt,
                user_message=rec.user_message,
                response=rec.response,
                level=level,
                step=step,
                reward=reward,
                reward_breakdown=breakdown,
            ))

        return {
            "events": events,
            "metrics": metrics,
            "n_orai_calls": len(records),
            "memory_dir": self.memory_dir,
        }

    # ── Bridged-mode (step-by-step driven by trainer) API ───────────────

    def start_bridged_rollout(self) -> Tuple[str, list]:
        """Bridged mode: spawn agent in background thread; return first ORAI prompt.

        Returns (system_prompt, messages) for the FIRST ORAI call. Trainer should
        generate a completion and call feed_response() to advance.

        Raises if not in bridged mode.
        """
        if self._bridge is None:
            raise RuntimeError("start_bridged_rollout requires mode='bridged'")
        if self._thread is not None:
            raise RuntimeError("rollout already started")

        from run_agent import load_game  # noqa: E402

        self.game = load_game(self.game_id, self.seed)

        def _run_in_thread():
            try:
                self.metrics = self.agent.run(
                    self.game,
                    self.game_id,
                    seed=self.seed,
                    max_levels=self.max_levels,
                )
            except Exception as e:
                # Log; don't crash the bridge — main thread will see done.
                import traceback
                traceback.print_exc()
                self._thread_error = e
            finally:
                # Always signal done so main thread doesn't deadlock.
                self._bridge.signal_done()

        self._thread_error = None
        self._thread = threading.Thread(target=_run_in_thread, daemon=True)
        self._thread.start()

        prompt = self._bridge.next_prompt(timeout=120.0)
        if prompt is None:
            # Either agent never reached an ORAI call, or rollout ended fast
            # (e.g. all levels solved without any LLM call — unlikely but possible).
            return ("", [])
        return prompt

    def feed_completion_and_get_next(
        self,
        completion: str,
        timeout: float = 120.0,
    ) -> Tuple[Optional[Tuple[str, list]], "RolloutEvent"]:
        """Feed trainer's completion to the running agent; return next ORAI prompt
        (or None if rollout done) and the RolloutEvent corresponding to the call
        we just completed (with reward computed from path delta).
        """
        if self._bridge is None:
            raise RuntimeError("feed_completion_and_get_next requires mode='bridged'")

        # The bridge's interceptor has already pushed a record at request time
        # (with pre_path), but BEFORE our feed_response(). The completion is
        # what we now feed. The interceptor's record will be appended once
        # request_completion returns. To make the timing correct, we feed first,
        # then read the new record.
        self._bridge.feed_response(completion)

        # Wait for agent to either submit next prompt or finish.
        next_prompt = self._bridge.next_prompt(timeout=timeout)

        # The just-completed ORAI call recorded ONE entry in self._records
        # AFTER our feed_response (when request_completion returned).
        # Build the event for that call now using post-call game state.
        event = self._build_latest_event()

        return next_prompt, event

    def _build_latest_event(self) -> "RolloutEvent":
        """Construct a RolloutEvent for the most recently completed ORAI call,
        using the current game state as the 'after' snapshot."""
        from agent.runtime.process_reward import (    # noqa: E402
            compute_geometric_reward,
            compute_outcome_reward,
        )

        if not self._records:
            # No record yet — return a zero-reward placeholder.
            return RolloutEvent(
                system_prompt="", user_message="", response="",
                level=0, step=0,
                reward=0.0, reward_breakdown={},
            )

        rec = self._records[-1]
        new_path = list(getattr(self.game, "_path", []))
        new_level = getattr(self.agent, "_level_index", 0)

        geo_total, geo_breakdown = compute_geometric_reward(
            self.game,
            prev_path=rec.pre_path,
            new_path=new_path,
            is_valid_move=True,
        )
        out_total, out_breakdown = compute_outcome_reward(
            rec.pre_level_index, new_level, self.max_levels,
        )

        # Look up matching ReflectionTrace for level/step metadata.
        idx = len(self._records) - 1
        refls = self.trace._reflections
        level = refls[idx].level if idx < len(refls) else rec.pre_level_index
        step = refls[idx].step if idx < len(refls) else 0

        return RolloutEvent(
            system_prompt=rec.system_prompt,
            user_message=rec.user_message,
            response=rec.response,
            level=level,
            step=step,
            reward=geo_total + out_total,
            reward_breakdown={**geo_breakdown, **out_breakdown},
        )

    def join_bridged_rollout(
        self,
        timeout: float = 30.0,
        signal_done_first: bool = True,
    ) -> Dict[str, Any]:
        """Wait for the background rollout thread to finish; return final metrics.

        signal_done_first=True (default): signal the bridge to wind down BEFORE
        waiting. Handles the truncation case — trainer ran out of step budget
        mid-trajectory (or ORAI loop hit max_orai_steps cap). After signal_done,
        agent's pending LLM call returns "", agent emits default actions, and
        run() unwinds via natural max_steps_per_level termination.

        Without signal_done_first the thread typically deadlocks on the bridge's
        Event waiting for a response that will never come.
        """
        if signal_done_first and self._bridge is not None:
            self._bridge.signal_done()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if getattr(self, "_thread_error", None) is not None:
            raise self._thread_error
        return {
            "metrics": self.metrics,
            "n_orai_calls": len(self._records),
            "memory_dir": self.memory_dir,
        }

    # ─────────────────────────────────────────────────────────────────────

    def cleanup(self):
        """Explicit cleanup — call after consuming events to free tempdir."""
        # Ensure no zombie agent thread.
        if self._bridge is not None:
            self._bridge.signal_done()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._memory_cleanup is not None:
            try:
                self._memory_cleanup()
            except Exception:
                pass
            self._memory_cleanup = None
