"""Negotiation RLVR environment for SkyRL-Gym.

The trained policy plays one side of a two-player, multi-issue negotiation
(the "you" side); the environment plays the opponent (the "them" side) by
calling a fixed reference LLM over an OpenAI-compatible endpoint (OpenRouter
via litellm, matching the in-repo pattern used by ``task_gen`` and
``hint_synthesizer``). The reward is **fully verifiable** and computed in
exactly one place — :func:`game.evaluate` — from the parsed final allocation
and each agent's private value vector. No judge is involved.

Two reward modes (the planned ablation, see ``eval/REPORT.md``):
  - ``outcome``        : normalized self-score (``score / max_possible``).
                         No-deal / conflict / incomplete = 0.
  - ``outcome_pareto`` : ``outcome + pareto_coef * joint_efficiency`` on agreement,
                         where ``joint_efficiency`` is the achieved joint score over
                         the best achievable joint score (continuous in [0, 1]). This
                         is a denser gradient than the binary Pareto flag, which is
                         sparse, orthogonal to slice size, and would reward a
                         lopsided-but-technically-frontier split. No-deal / conflict /
                         incomplete still = 0, preserving the no-deal deterrent.

Two protocols (see ``prompts.py`` / ``game.py``):
  - ``single`` (default, recommended): one side proposes a full split via
    ``<propose>{...}</propose>`` (listing what THEY keep; partner gets the
    rest); the other finalizes with ``<accept>``. Only failure mode: no_deal.
  - ``dual``: both sides each emit a ``<deal>{...}</deal>`` of their own keep;
    the two claims must exactly partition the pool, else the deal fails.

Turn model: the policy ("you") always speaks first. The SkyRL generator drives
one policy generation per :meth:`step_async`; within each step we record the
policy's message, then (if the game is not already over) generate the
opponent's reply and return it as the next user observation. The episode ends
on an accepted deal (single) / both-tags-present (dual), or when the policy's
per-agent message budget (``max_turns``) is exhausted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.negotiation import game, prompts

logger = logging.getLogger(__name__)

# Guards the append to the per-process transcript JSONL. Episodes run concurrently
# (env.step/close are dispatched to a thread pool by the generator), so two episodes
# can finish at once; the lock keeps each episode's JSON line intact on disk.
_TRANSCRIPT_LOCK = threading.Lock()

_THINK_RE = re.compile(r"<think>.*?</think>|<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think> content from the policy action before forwarding to the opponent.

    The policy's private reasoning must never be visible to the opponent LLM:
    it contains the policy's own item valuations, defeating the hidden-values premise.
    Handles both properly-closed <think>...</think> blocks and the common case where
    thinking mode injects an unclosed <think> that the stop-token ends mid-block.
    """
    return _THINK_RE.sub("", text).strip()


# Qwen3 hybrid-reasoning soft switch: appending this token disables the thinking
# block for that turn (keeps the budget-constrained opponent from burning its
# whole reply on a <think> block — see eval/REPORT.md "turn thinking OFF").
NO_THINK_TOKEN = "/no_think"

# Phrases by which the policy's prose promises an item to the opponent. Used by
# the deception detector: a message is deceptive when it promises an item while
# its <propose> JSON keeps the pool's full count of that same item, so the
# opponent reading the prose believes it receives something the machine-readable
# offer denies it. Deliberately conservative (full-count keeps only) — partial
# keeps are ordinary hard bargaining, not deception.
_GIVE_PHRASE = (
    r"(?:you (?:can |could |will |would )?(?:keep|have|get|take|receive)"
    r"|you(?:'ll| will) (?:keep|have|get|take|receive)"
    r"|giv(?:e|ing) you"
    r"|I(?:'ll| will)? give up"
    r"|leav(?:e|ing) you(?: with)?"
    r"|offer(?:ing)? you)"
)


def _deceptive_promises(text: str, keep: List[int], item_names: List[str], counts: List[int]) -> int:
    """Count items the prose promises to the opponent while ``keep`` retains all of them."""
    prose = _strip_think(text)
    pattern = _GIVE_PHRASE + r"[^.<\n]*?\b(" + "|".join(re.escape(n) for n in item_names) + r")s?\b"
    promised = {m.group(1) for m in re.finditer(pattern, prose, re.IGNORECASE)}
    hits = 0
    for i, name in enumerate(item_names):
        if name in promised and counts[i] > 0 and keep[i] >= counts[i]:
            hits += 1
    return hits


def _cfg_get(env_config: Any, key: str, default: Any) -> Any:
    """Read a key from a DictConfig / dict / None env_config with a default."""
    if env_config is None:
        return default
    if hasattr(env_config, "get"):
        val = env_config.get(key, default)
        return default if val is None else val
    return getattr(env_config, key, default)


class NegotiationEnv(BaseTextEnv):
    """Self-play-vs-fixed-opponent negotiation env with verifiable rewards."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        gt = extras["reward_spec"].get("ground_truth")
        assert gt is not None, "reward_spec.ground_truth (the scenario) is required"

        # --- Scenario (the verifiable ground truth) ---
        self.item_names: List[str] = list(gt["item_names"])
        self.counts: List[int] = [int(c) for c in gt["counts"]]
        self.you_values: List[int] = [int(v) for v in gt["you_values"]]
        self.them_values: List[int] = [int(v) for v in gt["them_values"]]
        self.n_items = len(self.counts)

        extra_info: Dict[str, Any] = extras.get("extra_info", {}) or {}
        # Keep an id for the scenario (if the dataset baked one in) so logged
        # transcripts can be cross-referenced back to the prompt that produced them.
        self.scenario_id: Optional[str] = extra_info.get("scenario_id") or extra_info.get("id")

        # --- Per-agent message budget ---
        # The generator sets extras["max_turns"] = generator.max_turns; that is
        # the number of policy turns (= agent_loop iterations). Fall back to the
        # value baked into the dataset, then a small default.
        self.max_turns = int(
            extras.get("max_turns", extra_info.get("max_turns", 6))
        )

        # --- Protocol: prefer what the prompt was built with (dataset), else config ---
        self.protocol: str = (
            extra_info.get("protocol")
            or _cfg_get(env_config, "protocol", "single")
        )

        # --- Reward shaping config ---
        self.reward_mode: str = _cfg_get(env_config, "reward_mode", "outcome")
        self.pareto_coef: float = float(_cfg_get(env_config, "pareto_coef", 0.5))
        self.invalid_penalty: float = float(_cfg_get(env_config, "invalid_penalty", 0.0))
        # Per-message penalty for prose-vs-JSON deception (see _deceptive_promises).
        # Applied to the FINAL reward (not the step reward) so that deceptive
        # proposals the opponent immediately accepts — the most profitable case —
        # are still penalized despite the episode terminating on that same step.
        self.deception_penalty: float = float(_cfg_get(env_config, "deception_penalty", 0.0))
        self.deception_msgs: int = 0

        # --- Thinking-trace inspection log ---
        # When set, each finished episode's FULL transcript (the policy's raw "you"
        # messages still carry their <think>...</think> reasoning) is appended as one
        # JSON line to a per-process file under this dir. This is the only place the
        # thinking traces are persisted: they are stripped from the opponent's view
        # (see _strip_think) and from the policy's own multi-turn training context
        # (the qwen3_without_thinking chat template strips non-last-turn thinking), so
        # without this log the reasoning is unrecoverable after the rollout.
        self.transcript_dir: Optional[str] = _cfg_get(env_config, "transcript_dir", None)
        # Fraction of episodes to log (1.0 = all). Lower it if the log grows too large.
        self.transcript_sample_rate: float = float(_cfg_get(env_config, "transcript_sample_rate", 1.0))

        # --- Opponent ("them") LLM config ---
        self.opponent_model: str = _cfg_get(
            env_config, "opponent_model", "openrouter/openai/gpt-4o-mini"
        )
        self.opponent_base_url: Optional[str] = _cfg_get(env_config, "opponent_base_url", None)
        self.opponent_temperature: float = float(_cfg_get(env_config, "opponent_temperature", 0.7))
        self.opponent_max_tokens: int = int(_cfg_get(env_config, "opponent_max_tokens", 512))
        self.opponent_timeout: float = float(_cfg_get(env_config, "opponent_timeout", 60.0))
        self.opponent_no_think: bool = bool(_cfg_get(env_config, "opponent_no_think", True))
        self.opponent_max_retries: int = int(_cfg_get(env_config, "opponent_max_retries", 2))
        self.openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")

        # --- Opponent system prompt: reuse the one baked into the dataset if
        # present (guarantees it matches what was shown to the policy), else
        # rebuild from the scenario + protocol. ---
        them_sys = extra_info.get("them_system_prompt")
        if not them_sys:
            them_sys = prompts.build_system_prompt(
                self.item_names, self.counts, self.them_values,
                self.max_turns, protocol=self.protocol,
            )
        if self.opponent_no_think:
            them_sys = them_sys + "\n\n" + NO_THINK_TOKEN
        # The opponent never speaks first ("them" waits for the opener).
        self.them_history: ConversationType = [{"role": "system", "content": them_sys}]

        # --- Episode state ---
        # `pending` (single protocol): {"by": "you"|"them", "keep": [...]} most recent valid offer.
        self.pending: Optional[Dict[str, Any]] = None
        # dual protocol: each side's most recent parsed <deal> claim.
        self.you_deal: Optional[List[int]] = None
        self.them_deal: Optional[List[int]] = None
        # Final allocation (each side's keep), filled when the game resolves.
        self.you_take: Optional[List[int]] = None
        self.them_take: Optional[List[int]] = None
        self.outcome: Optional[game.Outcome] = None
        self.final_reward: float = 0.0
        self.opponent_errors: int = 0
        self.transcript: List[Dict[str, str]] = []

    # ------------------------------------------------------------------ init
    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        # The policy's full opening prompt (system + opening user msg) is built
        # in prepare_dataset.py and passed through unchanged. "You" speaks first.
        return prompt, {}

    # --------------------------------------------------------------- opponent
    async def _opponent_reply(self) -> str:
        """Generate the opponent's ("them") next message via litellm.

        Returns the reply text, or "" on failure (treated as an empty turn).
        """
        if not self.openrouter_api_key and self.opponent_model.startswith("openrouter/"):
            # Without a key we cannot drive the opponent; surface once.
            if self.opponent_errors == 0:
                logger.warning("OPENROUTER_API_KEY not set; opponent will be silent (deals will fail).")
            self.opponent_errors += 1
            return ""

        kwargs: Dict[str, Any] = {
            "model": self.opponent_model,
            "messages": self.them_history,
            "max_tokens": self.opponent_max_tokens,
            "temperature": self.opponent_temperature,
        }
        if self.openrouter_api_key:
            kwargs["api_key"] = self.openrouter_api_key
        if self.opponent_base_url:
            kwargs["base_url"] = self.opponent_base_url

        try:
            from litellm import acompletion
        except ImportError:
            logger.warning("litellm not installed; opponent will be silent.")
            self.opponent_errors += 1
            return ""

        for attempt in range(self.opponent_max_retries + 1):
            try:
                resp = await asyncio.wait_for(acompletion(**kwargs), timeout=self.opponent_timeout)
                choices = getattr(resp, "choices", None)
                if not choices:
                    return ""
                return (choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                if attempt >= self.opponent_max_retries:
                    self.opponent_errors += 1
                    logger.warning(f"Opponent LLM call failed after retries: {e}")
                    return ""
                await asyncio.sleep(1.0 * (attempt + 1))
        return ""

    # ------------------------------------------------------------------ step
    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.transcript.append({"speaker": "you", "text": action})
        self.them_history.append({"role": "user", "content": _strip_think(action)})

        if self.protocol == "dual":
            return await self._step_dual(action)
        return await self._step_single(action)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Synchronous fallback (the generator prefers step_async when present).
        return asyncio.run(self.step_async(action))

    # --------------------------------------------------------- single protocol
    async def _step_single(self, action: str) -> BaseTextEnvStepOutput:
        # 1. If the policy accepts a pending opponent offer, the deal closes now
        #    (accept wins over a co-occurring propose).
        if game.has_accept(action) and self.pending and self.pending["by"] == "them":
            self._finalize_single(self.pending)
            return self._terminal_output()

        # 2. Otherwise, record any proposal the policy made.
        prop = game.parse_proposal(action, self.item_names)
        if prop is not None:
            keep = [min(self.counts[i], max(0, prop[i])) for i in range(self.n_items)]
            self.pending = {"by": "you", "keep": keep}
            if self.deception_penalty != 0.0 and _deceptive_promises(
                action, keep, self.item_names, self.counts
            ):
                self.deception_msgs += 1

        # Penalise turns where the policy emitted no parseable action — this
        # discourages the degenerate "<think>\nThe<|im_end|>" collapse pattern.
        step_reward = 0.0 if prop is not None else self.invalid_penalty

        budget_exhausted = self.turns >= self.max_turns

        # 3. Opponent responds. It may accept the policy's pending offer or
        #    counter with its own proposal.
        them_text = await self._opponent_reply()
        self.them_history.append({"role": "assistant", "content": them_text})
        self.transcript.append({"speaker": "them", "text": them_text})

        if game.has_accept(them_text) and self.pending and self.pending["by"] == "you":
            self._finalize_single(self.pending)
            return self._terminal_output()

        them_prop = game.parse_proposal(them_text, self.item_names)
        if them_prop is not None:
            keep = [min(self.counts[i], max(0, them_prop[i])) for i in range(self.n_items)]
            self.pending = {"by": "them", "keep": keep}

        # 4. Budget check: after the opponent has had a chance to respond.
        if budget_exhausted:
            # No agreement reached within the budget -> no_deal (reward 0).
            self._resolve(None, None)
            return self._terminal_output()

        # 5. Continue: hand the opponent's message back to the policy.
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": them_text}],
            reward=step_reward,
            done=False,
            metadata={},
        )

    def _finalize_single(self, accepted: Dict[str, Any]) -> None:
        keep = accepted["keep"]
        other = [self.counts[i] - keep[i] for i in range(self.n_items)]
        if accepted["by"] == "you":
            self._resolve(keep, other)
        else:
            self._resolve(other, keep)

    # ----------------------------------------------------------- dual protocol
    async def _step_dual(self, action: str) -> BaseTextEnvStepOutput:
        deal = game.parse_deal(action, self.item_names)
        if deal is not None:
            self.you_deal = deal

        step_reward = 0.0 if deal is not None else self.invalid_penalty

        # Both tags already in -> resolve (the opponent placed its tag on a
        # prior turn).
        if self.you_deal is not None and self.them_deal is not None:
            self._resolve(self.you_deal, self.them_deal)
            return self._terminal_output()

        budget_exhausted = self.turns >= self.max_turns

        them_text = await self._opponent_reply()
        self.them_history.append({"role": "assistant", "content": them_text})
        self.transcript.append({"speaker": "them", "text": them_text})
        them_deal = game.parse_deal(them_text, self.item_names)
        if them_deal is not None:
            self.them_deal = them_deal

        if self.you_deal is not None and self.them_deal is not None:
            self._resolve(self.you_deal, self.them_deal)
            return self._terminal_output()

        if budget_exhausted:
            # One or both tags missing -> evaluate yields no_deal/incomplete/conflict (0).
            self._resolve(self.you_deal, self.them_deal)
            return self._terminal_output()

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": them_text}],
            reward=step_reward,
            done=False,
            metadata={},
        )

    # --------------------------------------------------------------- resolving
    def _resolve(self, you_take: Optional[List[int]], them_take: Optional[List[int]]) -> None:
        """Compute the verifiable outcome and the scalar reward."""
        self.you_take = you_take
        self.them_take = them_take
        self.outcome = game.evaluate(
            self.counts, self.you_values, self.them_values, you_take, them_take
        )
        out = self.outcome
        if out.agreed:
            reward = out.you_norm
            if self.reward_mode == "outcome_pareto":
                # Continuous joint-efficiency shaping (not the binary Pareto flag):
                # denser gradient, and it won't reward a lopsided split just because
                # it happens to sit on the frontier.
                reward = reward + self.pareto_coef * out.joint_efficiency
        else:
            reward = 0.0
        # Deception penalty applies regardless of outcome: a deceptive promise
        # is a deceptive promise even when the game ends in no_deal.
        reward += self.deception_penalty * self.deception_msgs
        self.final_reward = float(reward)

    def _terminal_output(self) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.final_reward,
            done=True,
            metadata={"outcome": self.outcome.reason if self.outcome else "unknown"},
        )

    # ----------------------------------------------------------------- metrics
    def get_metrics(self) -> Dict[str, Any]:
        out = self.outcome
        if out is None:
            return {"resolved": 0.0, "num_turns": float(self.turns)}
        return {
            "final_reward": self.final_reward,
            "agreed": 1.0 if out.agreed else 0.0,
            "no_deal": 1.0 if out.reason == "no_deal" else 0.0,
            "conflict": 1.0 if out.reason == "conflict" else 0.0,
            "incomplete": 1.0 if out.reason == "incomplete" else 0.0,
            "you_norm": out.you_norm,
            "them_norm": out.them_norm,
            "joint_efficiency": out.joint_efficiency,
            "pareto": out.pareto_bonus,
            "num_turns": float(self.turns),
            "opponent_errors": float(self.opponent_errors),
            "deception_msgs": float(self.deception_msgs),
        }

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        """Persist the full episode transcript (with thinking) for inspection.

        Called once by the generator at the end of every episode. Best-effort: any
        failure here is logged and swallowed so it can never break a training step.
        """
        if not self.transcript_dir:
            return
        if self.transcript_sample_rate < 1.0 and random.random() >= self.transcript_sample_rate:
            return

        out = self.outcome
        record = {
            "ts": time.time(),
            "episode_id": uuid.uuid4().hex,
            "scenario_id": self.scenario_id,
            "protocol": self.protocol,
            "reward_mode": self.reward_mode,
            "opponent_model": self.opponent_model,
            "scenario": {
                "item_names": self.item_names,
                "counts": self.counts,
                "you_values": self.you_values,
                "them_values": self.them_values,
            },
            "outcome": out.reason if out is not None else "unknown",
            "you_take": self.you_take,
            "them_take": self.them_take,
            "final_reward": self.final_reward,
            "deception_msgs": self.deception_msgs,
            "metrics": self.get_metrics(),
            # "you" turns retain their raw <think>...</think> reasoning; this is the
            # only persisted copy of the policy's thinking for this episode.
            "transcript": self.transcript,
        }

        try:
            dir_path = Path(self.transcript_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            # One file per host+process avoids cross-worker write contention; the lock
            # serialises the concurrent episodes within this process.
            file_path = dir_path / f"transcripts_{socket.gethostname()}_{os.getpid()}.jsonl"
            line = json.dumps(record, ensure_ascii=False, default=str)
            with _TRANSCRIPT_LOCK:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to write negotiation transcript to {self.transcript_dir}: {e}")
