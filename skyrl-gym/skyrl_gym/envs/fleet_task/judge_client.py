"""Async LLM-judge client for rubric / pure verifier modes.

Two modes:
  - "rubric": judge gets rubric_json + chat_history + task_spec; returns Likert 1-10
              (mapped to [0,1] reward). Default ensemble: Sonnet 4.6 + GPT-5-mini.
  - "pure":   judge gets task_spec + verifier_code + chat_history (+ optional env-state
              probe tools); returns continuous score in [0,1]. Default: GPT-5.5 single.

Per-rollout call lives in FleetTaskEnv.close_async(). Concurrency is governed by the
trainer (one rollout per env instance). Each judge call retries with exponential
backoff on transient errors and falls back to the binary verifier score if the
judge fails terminally.

System prompts are loaded lazily from the prompts/ directory under the
verifier-ablation thread. The prompt files live alongside this module's parent
docs; see SYSTEM_PROMPT_*_FALLBACK below for in-source defaults if the files are
unreachable on the cluster.

Score format ingested by the trainer:
  rubric mode → reward = (likert - 1) / 9, clamped to [0.05, 1.0] (avoid dead zone)
  pure mode   → reward = score_continuous_0_1, clamped to [0.0, 1.0]

Env-state probe tools (pure judge only) are wired in via the `env_probe_callbacks`
parameter on JudgeClient — the FleetTaskEnv passes a dict of callables that the
judge can invoke through the OpenAI / Anthropic tool-use API. v1 ships with stubs
that return "(not yet implemented)"; v2 will plumb actual API calls to the live
Fleet env instance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---- defaults — overrideable via env_config ----
DEFAULT_RUBRIC_JUDGES = [
    {"provider": "anthropic", "model": "claude-sonnet-4-6", "weight": 0.5, "max_tokens": 2048},
    {"provider": "openai",    "model": "gpt-5-mini",        "weight": 0.5, "max_tokens": 4096,
     "reasoning_effort": "low"},
]
DEFAULT_PURE_JUDGES = [
    {"provider": "openai",    "model": "gpt-5.5",           "weight": 1.0, "max_tokens": 4096,
     "reasoning_effort": "low"},
]
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0
TRAJ_HEAD_CAP = 60_000
TRAJ_TAIL_CAP = 30_000

SYSTEM_PROMPT_RUBRIC_FALLBACK = (
    "You are a strict evaluator scoring an AI agent's trajectory against a provided rubric. "
    "Apply the rubric holistically and return a single Likert score 1-10 in JSON. See the full "
    "system prompt at fleet-research/threads/verifier-ablation/rubric-verifier/prompts/rubric-judge.md."
)
SYSTEM_PROMPT_PURE_FALLBACK = (
    "You are a verifier agent for a tool-using AI assistant. Decide whether the assistant "
    "successfully completed the task. Return JSON with verdict + score_continuous_0_1 + "
    "reasoning. See full prompt at fleet-research/threads/verifier-ablation/rubric-verifier/"
    "prompts/pure-judge.md."
)


def _format_chat_history_compact(chat_history: List[Dict[str, str]]) -> str:
    """Render chat_history as a head + tail trajectory string (handles long agentic runs)."""
    parts = []
    full = "\n\n".join(
        f"[{m.get('role', '?')}]:\n{m.get('content', '')}" for m in chat_history
    )
    if len(full) <= TRAJ_HEAD_CAP + TRAJ_TAIL_CAP:
        return full
    head = full[:TRAJ_HEAD_CAP]
    tail = full[-TRAJ_TAIL_CAP:]
    omitted = len(full) - TRAJ_HEAD_CAP - TRAJ_TAIL_CAP
    return (
        f"{head}\n\n... [TRUNCATED {omitted} CHARS — final actions follow] ...\n\n{tail}"
    )


def _parse_json_blob(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", txt)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


class JudgeClient:
    """Async ensemble judge — calls 1+ models, returns averaged reward in [0,1].

    Args:
        mode: "rubric" or "pure"
        models: list of {provider: anthropic|openai, model: str, weight: float, max_tokens: int, ...}
        system_prompt: full system prompt text. If None, falls back to in-source default.
        env_probe_callbacks: dict of callable tools the pure judge can invoke. Pure mode only;
            v1 ignores this and runs trajectory-only judging.
    """

    def __init__(
        self,
        mode: str,
        models: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        env_probe_callbacks: Optional[Dict[str, Callable]] = None,
    ):
        assert mode in ("rubric", "pure"), f"unknown judge mode: {mode}"
        self.mode = mode
        self.models = models or (
            DEFAULT_RUBRIC_JUDGES if mode == "rubric" else DEFAULT_PURE_JUDGES
        )
        self.system_prompt = system_prompt or (
            SYSTEM_PROMPT_RUBRIC_FALLBACK if mode == "rubric" else SYSTEM_PROMPT_PURE_FALLBACK
        )
        self.env_probe_callbacks = env_probe_callbacks or {}
        self._anthropic_client = None
        self._openai_client = None

    def _ensure_anthropic(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic()
        return self._anthropic_client

    def _ensure_openai(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI()
        return self._openai_client

    def _build_user_message(
        self,
        task_spec: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        rubric_json: Optional[str],
    ) -> str:
        traj = _format_chat_history_compact(chat_history)
        prompt = (task_spec.get("prompt") or "")[:2500]
        env = task_spec.get("env_key") or task_spec.get("data_source") or "?"
        task_key = task_spec.get("key") or task_spec.get("task_key") or "?"
        if self.mode == "rubric":
            return f"""RUBRIC
======
{rubric_json or '(missing — fall back to verifier_code reasoning)'}

TASK SPEC (for context)
=======================
env: {env}
task_key: {task_key}
prompt: {prompt}

TRAJECTORY
==========
{traj}
"""
        else:  # pure
            verifier = (task_spec.get("verifier_code") or "(none)")[:3000]
            return f"""TASK SPEC
=========
env: {env}
task_key: {task_key}
prompt: {prompt}

VERIFIER FUNCTION (privileged context — the actual Python check the env runs):
{verifier}

TRAJECTORY (compact summary)
============================
{traj}
"""

    async def _call_one(
        self, model_cfg: Dict[str, Any], user_message: str
    ) -> Optional[Dict[str, Any]]:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        max_tokens = model_cfg.get("max_tokens", 2048)
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "anthropic":
                    client = self._ensure_anthropic()
                    msg = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=self.system_prompt,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    return _parse_json_blob(msg.content[0].text)
                elif provider == "openai":
                    client = self._ensure_openai()
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "max_completion_tokens": max_tokens,
                    }
                    if "reasoning_effort" in model_cfg:
                        kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
                    resp = await client.chat.completions.create(**kwargs)
                    return _parse_json_blob(resp.choices[0].message.content)
                else:
                    logger.error(f"unknown provider: {provider}")
                    return None
            except Exception as e:
                if attempt + 1 < MAX_RETRIES:
                    sleep = RETRY_BASE_SLEEP * (2 ** attempt)
                    logger.warning(
                        f"judge call to {provider}/{model} failed (attempt {attempt+1}): {e}; "
                        f"retrying in {sleep}s"
                    )
                    await asyncio.sleep(sleep)
                else:
                    logger.error(f"judge call to {provider}/{model} failed terminally: {e}")
        return None

    def _aggregate(self, parsed_outputs: List[Optional[Dict[str, Any]]]) -> Optional[float]:
        scores: List[float] = []
        weights: List[float] = []
        for parsed, cfg in zip(parsed_outputs, self.models):
            if parsed is None:
                continue
            w = cfg.get("weight", 1.0)
            if self.mode == "rubric":
                s = parsed.get("score_likert_1_10")
                if s is None:
                    continue
                # map Likert 1-10 → [0,1]
                reward = (float(s) - 1.0) / 9.0
            else:  # pure
                reward = parsed.get("score_continuous_0_1")
                if reward is None:
                    # also accept verdict mapping
                    verdict = (parsed.get("verdict") or "").lower()
                    reward = {"pass": 1.0, "partial": 0.5, "fail": 0.0}.get(verdict)
                    if reward is None:
                        continue
                reward = float(reward)
            reward = max(0.05 if self.mode == "rubric" else 0.0, min(1.0, reward))
            scores.append(reward)
            weights.append(w)
        if not scores:
            return None
        total_w = sum(weights) or 1.0
        return sum(s * w for s, w in zip(scores, weights)) / total_w

    async def score(
        self,
        task_spec: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        rubric_json: Optional[str] = None,
    ) -> Optional[float]:
        """Returns aggregated reward in [0,1], or None if all models failed."""
        if self.mode == "rubric" and not rubric_json:
            logger.warning("rubric mode but no rubric_json provided — judge will fall back")
        user_msg = self._build_user_message(task_spec, chat_history, rubric_json)
        # Fire all judges concurrently
        results = await asyncio.gather(
            *(self._call_one(cfg, user_msg) for cfg in self.models),
            return_exceptions=False,
        )
        return self._aggregate(results)

    async def score_with_audit(
        self,
        task_spec: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        rubric_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns dict with judge_score plus per-model breakdown (for audit logging)."""
        user_msg = self._build_user_message(task_spec, chat_history, rubric_json)
        t0 = time.time()
        results = await asyncio.gather(
            *(self._call_one(cfg, user_msg) for cfg in self.models),
            return_exceptions=False,
        )
        per_model: List[Dict[str, Any]] = []
        for cfg, parsed in zip(self.models, results):
            score: Optional[float] = None
            raw_likert: Optional[float] = None
            verdict: Optional[str] = None
            if parsed:
                if self.mode == "rubric":
                    raw_likert = parsed.get("score_likert_1_10")
                    if raw_likert is not None:
                        score = (float(raw_likert) - 1.0) / 9.0
                else:
                    score = parsed.get("score_continuous_0_1")
                    verdict = parsed.get("verdict")
                    if score is None and verdict:
                        score = {"pass": 1.0, "partial": 0.5, "fail": 0.0}.get(verdict.lower())
            per_model.append({
                "model": f"{cfg['provider']}/{cfg['model']}",
                "weight": cfg.get("weight", 1.0),
                "score": score,
                "likert": raw_likert,
                "verdict": verdict,
                "ok": parsed is not None,
            })
        agg = self._aggregate(results)
        return {
            "mode": self.mode,
            "judge_score": agg,
            "per_model": per_model,
            "elapsed_s": time.time() - t0,
        }
