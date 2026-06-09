"""IFEval LLM-as-Judge environment — Direction 3: in-context reward adaptation.

Drop-in variant of IFEvalEnv that replaces the programmatic scorer with an
LLM judge. The judge prompt optionally includes a FeedbackBuffer of recent
critiques to adapt its scoring in-context without any weight updates.

The judge is asked to score each constraint independently (0 or 1) and return
a JSON object. The final reward is the mean over constraints — same aggregation
as the programmatic scorer.

Design notes:
  - Uses claude-haiku-4-5 by default (cheap, fast enough for per-step calls).
  - Falls back to programmatic scoring if the API call fails, so training
    never stalls due to a transient API error.
  - Caching: identical (response, ground_truth, feedback_hash) triples are
    cached in-process to avoid redundant API calls during evaluation loops.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.ifeval.feedback_buffer import FeedbackBuffer, _INSTRUCTION_DESCRIPTIONS
from skyrl_gym.envs.ifeval.ifeval_utils import compute_score


# ── Judge prompt templates ────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """You are a precise instruction-following evaluator.

You will be given a user prompt and a model response. The response was generated
under a set of verifiable constraints. Your job is to score each constraint
independently: 1 if satisfied, 0 if not.

## Scoring rules
- Score each constraint in isolation. Do not let other constraints affect your judgment.
- A constraint is satisfied if and only if the response fully meets it. Partial credit = 0.
- Be literal: "at least 10 sentences" means ≥10 sentences. Count carefully.
- For case constraints (ALL CAPS / all lowercase): every alphabetic character must conform.
- For keyword constraints: the exact word must appear (case-insensitive). Synonyms do not count.
- For format constraints (JSON, bullet points, sections): check the structure, not just the content.
{feedback_section}
## Output format
Return ONLY a JSON object with this schema:
{{
  "scores": {{
    "<instruction_id>": 0 or 1,
    ...
  }},
  "rationale": "<one sentence per constraint explaining your score>"
}}
Do not include any text outside the JSON object."""

_USER_TEMPLATE = """## User prompt
{prompt}

## Model response
{response}

## Constraints to evaluate
{constraints}"""


def _build_constraint_list(instruction_ids: list, kwargs_list: list) -> str:
    lines = []
    for i, iid in enumerate(instruction_ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        desc = _INSTRUCTION_DESCRIPTIONS.get(iid, iid)
        kw_str = f" (params: {json.dumps(kw)})" if kw else ""
        lines.append(f"- {iid}: {desc}{kw_str}")
    return "\n".join(lines)


# ── In-process cache ──────────────────────────────────────────────────────────

_JUDGE_CACHE: Dict[str, float] = {}


def _cache_key(response: str, ground_truth_json: str, feedback_hash: str) -> str:
    raw = f"{response}|||{ground_truth_json}|||{feedback_hash}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Core judge function ───────────────────────────────────────────────────────

def judge_score(
    response: str,
    ground_truth_json: str,
    prompt_text: str = "",
    feedback_buffer: Optional[FeedbackBuffer] = None,
    model: str = "claude-haiku-4-5-20251001",
    api_key: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[float, dict]:
    """Score a response using an LLM judge.

    Returns:
        (score, detail_dict) where score is the mean over constraints [0, 1]
        and detail_dict has per-constraint scores and the judge's rationale.
    """
    try:
        spec = json.loads(ground_truth_json)
    except Exception:
        return 0.0, {}

    instruction_ids = spec.get("instruction_id_list", []) or []
    kwargs_list = spec.get("kwargs", []) or []

    if not instruction_ids:
        return 0.0, {}

    # Cache lookup.
    fb_text = feedback_buffer.format_for_prompt() if feedback_buffer else ""
    fb_hash = hashlib.md5(fb_text.encode()).hexdigest()
    key = _cache_key(response, ground_truth_json, fb_hash)
    if use_cache and key in _JUDGE_CACHE:
        cached = _JUDGE_CACHE[key]
        return cached, {"cached": True}

    # Build prompt.
    feedback_section = f"\n{fb_text}\n" if fb_text else ""
    system = _SYSTEM_TEMPLATE.format(feedback_section=feedback_section)
    constraint_list = _build_constraint_list(instruction_ids, kwargs_list)
    user = _USER_TEMPLATE.format(
        prompt=prompt_text or "(prompt not provided)",
        response=response,
        constraints=constraint_list,
    )

    # Call API.
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text.strip()
        if verbose:
            print(f"[judge] raw output: {raw[:200]}")
        # Strip markdown fences if present.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        scores = parsed.get("scores", {})
        per_constraint = {iid: int(bool(scores.get(iid, 0))) for iid in instruction_ids}
        score = sum(per_constraint.values()) / len(instruction_ids)
        detail = {
            "per_constraint": per_constraint,
            "rationale": parsed.get("rationale", ""),
            "feedback_used": bool(fb_text),
        }
    except Exception as e:
        if verbose:
            print(f"[judge] API error, falling back to programmatic: {e}")
        # Fallback: use programmatic scorer.
        score = compute_score(response, ground_truth_json)
        detail = {"fallback": True, "error": str(e)}

    if use_cache:
        _JUDGE_CACHE[key] = score

    return score, detail


# ── Environment ───────────────────────────────────────────────────────────────

class IFEvalJudgeEnv(BaseTextEnv):
    """IFEval environment using an LLM judge with optional in-context feedback.

    Compatible with IFEvalEnv's interface. Set feedback_buffer on the env
    (or pass via extras) to enable Direction 3 in-context adaptation.

    extras keys (beyond base IFEvalEnv):
        judge_model: str — Anthropic model ID (default: claude-haiku-4-5-20251001)
        judge_prompt_text: str — original user prompt, passed to judge for context
        feedback_buffer: FeedbackBuffer — in-context feedback buffer (optional)
        fallback_to_programmatic: bool — use programmatic scorer on API error (default True)
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.prompt_text = extras.get("judge_prompt_text", "")
        self.model = extras.get("judge_model", "claude-haiku-4-5-20251001")
        self.fallback = extras.get("fallback_to_programmatic", True)
        self.verbose = extras.get("verbose", False)

        # feedback_buffer can be set after construction too.
        self.feedback_buffer: Optional[FeedbackBuffer] = extras.get("feedback_buffer", None)

        self._last_detail: dict = {}

    def _get_reward(self, action: str) -> float:
        score, detail = judge_score(
            response=action,
            ground_truth_json=self.ground_truth,
            prompt_text=self.prompt_text,
            feedback_buffer=self.feedback_buffer,
            model=self.model,
            verbose=self.verbose,
        )
        self._last_detail = detail
        return score

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"judge_detail": self._last_detail},
        )
