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
    # Rubric externalizes criteria; judge just applies a checklist. Low reasoning is fine.
    # max_tokens needs to fit reasoning + Tier-A findings (~one per criterion) + JSON output;
    # tasks with 6+ Tier-A criteria push past 2K, so 4K minimum.
    {"provider": "anthropic", "model": "claude-sonnet-4-6", "weight": 0.5, "max_tokens": 4096},
    {"provider": "openai",    "model": "gpt-5-mini",        "weight": 0.5, "max_tokens": 4096,
     "reasoning_effort": "low"},
]
DEFAULT_PURE_JUDGES = [
    # Pure judge has no rubric — must derive success criteria itself + plan multi-step
    # tool calls. The whole point of GPT-5.5 here is strong reasoning, so default to high.
    {"provider": "openai",    "model": "gpt-5.5",           "weight": 1.0, "max_tokens": 8192,
     "reasoning_effort": "high",
     "use_tools": True},
]
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0
TRAJ_HEAD_CAP = 60_000
TRAJ_TAIL_CAP = 30_000
MAX_TOOL_CALLS_PER_JUDGE = 5  # cap per RaR / VAGEN — bound probe budget

# ---- tool schemas exposed to the pure judge ----
TOOL_SCHEMAS = [
    {
        "name": "read_env_state",
        "description": (
            "GET against the running Fleet env's read-only HTTP API. Returns JSON. "
            "Use to verify entities exist and fields are set correctly. "
            "Examples: '/api/bookings/4332', '/r/python/posts', '/users/me/inbox?unread=true'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "api_path": {"type": "string", "description": "Path under the env's API root"}
            },
            "required": ["api_path"],
        },
    },
    {
        "name": "inspect_trajectory_turn",
        "description": (
            "Re-read a specific trajectory turn verbatim. The summary in the user "
            "message is truncated; this returns the raw turn at the given index."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_index": {"type": "integer", "description": "0-based turn index"}
            },
            "required": ["turn_index"],
        },
    },
    {
        "name": "check_field",
        "description": (
            "Convenience: read_env_state(api_path), drill into json_path, compare to expected. "
            "Returns {actual, match}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "api_path": {"type": "string"},
                "json_path": {
                    "type": "string",
                    "description": "Dotted path with optional [N] indexing, e.g. 'data.items[0].id'",
                },
                "expected": {"description": "value to compare against"},
            },
            "required": ["api_path", "json_path", "expected"],
        },
    },
    {
        "name": "read_expected_outcome",
        "description": (
            "Returns the task's expected outcome (prose summary, optional structured fields, "
            "and the verifier_code excerpt) so the judge knows the success criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _to_openai_tool_schemas() -> List[Dict[str, Any]]:
    """Convert our tool schema list to OpenAI's chat.completions tool format."""
    out = []
    for t in TOOL_SCHEMAS:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


def _to_openai_responses_tool_schemas() -> List[Dict[str, Any]]:
    """Convert our tool schemas to OpenAI Responses API format (flat, no 'function' wrapper)."""
    out = []
    for t in TOOL_SCHEMAS:
        out.append({
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
            "strict": False,
        })
    return out


def _needs_responses_api(model: str, use_tools: bool, has_reasoning_effort: bool) -> bool:
    """OpenAI's chat.completions doesn't support tools+reasoning_effort for gpt-5.x reasoning
    models — must use /v1/responses for that combo. Use responses for all GPT-5.x with
    reasoning_effort to keep behavior uniform."""
    if model.startswith("gpt-5") and has_reasoning_effort:
        return True
    return False

SYSTEM_PROMPT_RUBRIC_FALLBACK = """You are a strict evaluator scoring an AI agent's trajectory against a
provided rubric. The agent operated in a multi-turn API-based environment.

You will receive:
  - The rubric (criteria + weights)
  - The agent's full trajectory (prompt + assistant turns + tool calls + tool
    responses, in order)
  - The task spec for context

Apply the rubric holistically. Read the trajectory once, identify how each
criterion fared (essential criteria carry the most weight, pitfall criteria
deduct), and produce a single Likert score from 1 to 10.

Scoring guide (Likert):
  10  All Essential criteria met, most Important met, no Pitfalls triggered.
   8  All Essential met, some Important missed, no Pitfalls.
   6  Most Essential met, mixed Important, minor Pitfalls.
   4  Some Essential missed OR major Pitfall triggered.
   2  Most Essential missed.
   1  Trajectory does not engage with the task at all.

Reasoning style:
  - Be terse. One short sentence per Tier A criterion, two short bullets for
    overall Tier B fit, then the score.
  - Refer to specific tool calls by index when needed.
  - If the trajectory is truncated by length cap mid-task, score what's there
    but note it.

Output STRICTLY this JSON, no extra prose:

{
  "task_key": "<from input>",
  "tier_a_findings": [
    {"id": "A1", "verdict": "met|partially_met|missed|n/a", "note": "<≤15 words>"}
  ],
  "tier_b_summary": "<≤30 words>",
  "pitfalls_triggered": ["<id of any matched pitfall>", ...],
  "score_likert_1_10": 7,
  "confidence": "high|medium|low"
}
"""

SYSTEM_PROMPT_PURE_FALLBACK = """You are a verifier agent for a tool-using AI assistant. Your job: decide
whether the assistant successfully completed a given task by inspecting BOTH
the trajectory of its actions AND the resulting environment state.

You have these tools (when available):

  read_env_state(api_path: str) -> JSON
    GET against the env's read-only API. Use this to verify expected entities
    exist, fields are set correctly, and side-effects landed.

  inspect_trajectory_turn(turn_index: int) -> {role, content}
    Re-read a specific turn of the trajectory verbatim (the trajectory you
    receive in the user message is summarized; this gives you the raw turn).

  check_field(api_path: str, json_path: str, expected: any) -> {actual, match: bool}
    Convenience: read_env_state(api_path), drill to json_path, compare to
    expected. Returns the actual value plus a boolean match.

  read_expected_outcome() -> {prose, structured, verifier_code}
    Returns the task's expected outcome and verifier function. Use once at
    the start to anchor what success means.

Verification protocol — follow in order:
  1. STATIC ASSESSMENT (no tools yet). Read the task spec, verifier function,
     and trajectory summary. If success/failure is obvious, decide and stop.
  2. RETROSPECTION (inspect_trajectory_turn). If something is ambiguous,
     re-read the relevant turn(s).
  3. PROACTIVE PROBING (read_env_state, check_field). Only if (1) and (2)
     leave real doubt. Use at most 5 probe calls per task.

Stop probing as soon as the verdict is unambiguous either way.

Output STRICTLY this JSON, no extra prose:

{
  "task_key": "<from input>",
  "verdict": "pass | partial | fail",
  "score_continuous_0_1": 0.0,
  "confidence": "high | medium | low",
  "evidence": {
    "trajectory_signals": ["<≤30 words: what the trajectory shows>"],
    "env_state_findings": ["<each tool call: api_path → key result>"],
    "decisive_factor": "<one sentence: what tipped the verdict>"
  },
  "tools_used": ["inspect_trajectory_turn", "read_env_state", ...]
}

Scoring scale:
  pass    = score 1.0   — task fully completed, no significant side-effects
  partial = score 0.3-0.7 — main objective hit but with caveats
  fail    = score 0.0   — task not completed, or completed-then-broken

Be a tough grader. Errors that the agent never noticed but left in the env
state count against it.
"""


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
        # Wrap trajectory in a fence + explicit "do not continue" framing so models don't
        # mistake the embedded <tool_call> / <|im_end|> tokens for a live conversation
        # they should continue. (Observed Claude continuing trajectories without this.)
        traj = _format_chat_history_compact(chat_history)
        prompt = (task_spec.get("prompt") or "")[:2500]
        env = task_spec.get("env_key") or task_spec.get("data_source") or "?"
        task_key = task_spec.get("key") or task_spec.get("task_key") or "?"
        traj_block = (
            "BEGIN_TRAJECTORY_TO_SCORE — this is INPUT DATA showing what an agent\n"
            "already did. Do NOT continue or extend the trajectory. Read it as evidence,\n"
            "then output your scoring JSON.\n"
            "------------------------------------------------------------------------\n"
            f"{traj}\n"
            "------------------------------------------------------------------------\n"
            "END_TRAJECTORY_TO_SCORE.\n"
            "Now produce your scoring JSON exactly as specified in the system prompt — "
            "no prose outside the JSON, no continuation of the trajectory."
        )
        if self.mode == "rubric":
            return f"""RUBRIC
======
{rubric_json or '(missing — fall back to verifier_code reasoning)'}

TASK SPEC (for context)
=======================
env: {env}
task_key: {task_key}
prompt: {prompt}

{traj_block}
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

{traj_block}
"""

    async def _dispatch_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Call one of the env_probe_callbacks the wrapper handed us."""
        cb = self.env_probe_callbacks.get(tool_name)
        if cb is None:
            return {"error": f"tool {tool_name} not implemented in this run"}
        try:
            return await cb(**tool_args)
        except TypeError as e:
            return {"error": f"bad tool args for {tool_name}: {e}"}
        except Exception as e:
            return {"error": f"{tool_name} raised {type(e).__name__}: {e}"}

    async def _call_one_anthropic(
        self, model_cfg: Dict[str, Any], user_message: str
    ) -> Optional[Dict[str, Any]]:
        """Anthropic tool-use loop. Returns the parsed final JSON, or None on failure."""
        client = self._ensure_anthropic()
        model = model_cfg["model"]
        max_tokens = model_cfg.get("max_tokens", 2048)
        use_tools = model_cfg.get("use_tools", False) and bool(self.env_probe_callbacks)
        messages = [{"role": "user", "content": user_message}]
        n_tool_calls = 0
        for _ in range(MAX_TOOL_CALLS_PER_JUDGE + 2):  # rough loop bound
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": self.system_prompt,
                "messages": messages,
            }
            if use_tools:
                kwargs["tools"] = TOOL_SCHEMAS
            msg = await client.messages.create(**kwargs)
            if msg.stop_reason == "tool_use":
                # Extract tool_use blocks, dispatch each, append results
                tool_results = []
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        if n_tool_calls >= MAX_TOOL_CALLS_PER_JUDGE:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": "tool budget exhausted; finalize the verdict"}),
                            })
                        else:
                            n_tool_calls += 1
                            result = await self._dispatch_tool(block.name, block.input or {})
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result)[:6000],
                            })
                # Append assistant turn (with tool_use blocks) and user turn (with results)
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": tool_results})
                continue
            # End — extract text and parse
            text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            return _parse_json_blob("".join(text_parts))
        logger.warning(f"anthropic tool-use loop exceeded bounds for {model}")
        return None

    async def _call_one_openai_responses(
        self, model_cfg: Dict[str, Any], user_message: str
    ) -> Optional[Dict[str, Any]]:
        """OpenAI Responses API tool-use loop (required for GPT-5.x + tools + reasoning_effort)."""
        client = self._ensure_openai()
        model = model_cfg["model"]
        max_tokens = model_cfg.get("max_tokens", 2048)
        use_tools = model_cfg.get("use_tools", False) and bool(self.env_probe_callbacks)
        # Responses API uses `input=[messages]`. After a function_call, append a
        # function_call_output item with the same call_id and the result string.
        input_items: List[Dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]
        n_tool_calls = 0
        for _ in range(MAX_TOOL_CALLS_PER_JUDGE + 2):
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": input_items,
                "instructions": self.system_prompt,
                "max_output_tokens": max_tokens,
            }
            if "reasoning_effort" in model_cfg:
                kwargs["reasoning"] = {"effort": model_cfg["reasoning_effort"]}
            if use_tools:
                kwargs["tools"] = _to_openai_responses_tool_schemas()
            resp = await client.responses.create(**kwargs)
            # Find function_call items in resp.output
            fn_calls = []
            text_parts: List[str] = []
            assistant_items: List[Dict[str, Any]] = []
            for item in resp.output:
                t = getattr(item, "type", None)
                if t == "function_call":
                    fn_calls.append(item)
                    # Only the fields the API accepts on input — `status` etc.
                    # come back on output but are rejected on input.
                    assistant_items.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })
                elif t == "message":
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            text_parts.append(getattr(c, "text", "") or "")
                # reasoning items are NOT carried forward — they were on the
                # output side; the Responses API doesn't accept them as input
                # without an encrypted_content blob, so we drop them and let
                # the model re-derive context from the function_call_output.
            if fn_calls:
                input_items.extend(assistant_items)
                for fc in fn_calls:
                    if n_tool_calls >= MAX_TOOL_CALLS_PER_JUDGE:
                        result = {"error": "tool budget exhausted; finalize the verdict"}
                    else:
                        n_tool_calls += 1
                        try:
                            tool_args = json.loads(fc.arguments or "{}")
                        except json.JSONDecodeError:
                            tool_args = {}
                        result = await self._dispatch_tool(fc.name, tool_args)
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": json.dumps(result)[:6000],
                    })
                continue
            return _parse_json_blob("".join(text_parts))
        logger.warning(f"openai responses tool-use loop exceeded bounds for {model}")
        return None

    async def _call_one_openai(
        self, model_cfg: Dict[str, Any], user_message: str
    ) -> Optional[Dict[str, Any]]:
        """OpenAI tool-use loop. Routes GPT-5.x + reasoning_effort to Responses API,
        everything else to chat.completions."""
        model = model_cfg["model"]
        use_tools = model_cfg.get("use_tools", False) and bool(self.env_probe_callbacks)
        has_reasoning = "reasoning_effort" in model_cfg
        if _needs_responses_api(model, use_tools, has_reasoning):
            return await self._call_one_openai_responses(model_cfg, user_message)

        client = self._ensure_openai()
        max_tokens = model_cfg.get("max_tokens", 2048)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        n_tool_calls = 0
        for _ in range(MAX_TOOL_CALLS_PER_JUDGE + 2):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
            if has_reasoning:
                kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
            if use_tools:
                kwargs["tools"] = _to_openai_tool_schemas()
            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                messages.append(choice.message.model_dump(exclude_unset=True))
                for tc in tool_calls:
                    if n_tool_calls >= MAX_TOOL_CALLS_PER_JUDGE:
                        result = {"error": "tool budget exhausted; finalize the verdict"}
                    else:
                        n_tool_calls += 1
                        try:
                            tool_args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            tool_args = {}
                        result = await self._dispatch_tool(tc.function.name, tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)[:6000],
                    })
                continue
            return _parse_json_blob(choice.message.content)
        logger.warning(f"openai tool-use loop exceeded bounds for {model}")
        return None

    async def _call_one(
        self, model_cfg: Dict[str, Any], user_message: str
    ) -> Optional[Dict[str, Any]]:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        for attempt in range(MAX_RETRIES):
            try:
                if provider == "anthropic":
                    return await self._call_one_anthropic(model_cfg, user_message)
                elif provider == "openai":
                    return await self._call_one_openai(model_cfg, user_message)
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
                # Surface tools-used and decisive_factor for audit / debugging.
                # Pure judge: tools_used list. Rubric judge: pitfalls_triggered.
                "tools_used": (parsed or {}).get("tools_used") if parsed else None,
                "pitfalls_triggered": (parsed or {}).get("pitfalls_triggered") if parsed else None,
                "decisive_factor": (
                    ((parsed or {}).get("evidence") or {}).get("decisive_factor")
                    if parsed else None
                ),
            })
        agg = self._aggregate(results)
        return {
            "mode": self.mode,
            "judge_score": agg,
            "per_model": per_model,
            "elapsed_s": time.time() - t0,
        }
