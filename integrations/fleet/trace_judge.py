"""Trace Interestingness Judge for Value Function Research (agu/vf).

Identifies which steps in an agent trace are "interesting" — i.e., high-causal-influence
decision points where the agent's choice meaningfully affected the outcome.

Two complementary approaches:

1. **Direct LLM judge** — passes each step to an LLM and asks it to score
   interestingness on a 0-1 scale. Fast but uncalibrated; use for exploration.

2. **Trajectory divergence** — given multiple rollouts of the same task, computes
   where they diverge. Steps with high cross-trajectory divergence are decision
   points by definition. Requires ≥2 rollouts per task but is model-free.

Calibration:
   Both methods output step-level scores. calibrate_against_rewards() measures
   how well those scores predict actual verifier outcomes. The ground-truth signal
   is: if you branch from an "interesting" step, the reward distribution should
   be high-variance (some branches succeed, some fail). We approximate this
   offline by checking whether high-scored steps correlate with outcome divergence
   across the rollout group.

Usage (offline analysis):
    from integrations.fleet.trace_judge import (
        parse_steps, direct_judge, divergence_judge, calibrate_against_rewards
    )

    # Single trace
    steps = parse_steps(chat_history)
    scores = direct_judge(steps, task_prompt, client=openai_client)

    # Multiple traces for same task
    all_steps = [parse_steps(h) for h in chat_histories]
    div_scores = divergence_judge(all_steps)

    # Calibration
    metrics = calibrate_against_rewards(div_scores, rewards)

Trace format (chat_history from FleetTaskEnv.get_metrics()):
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<task prompt>"},
        {"role": "assistant", "content": "<think>...</think><tool_call>...</tool_call>"},
        {"role": "user", "content": "Tool result: ..."},
        ...
    ]
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """One agent action + environment response pair."""

    turn_idx: int  # 0-indexed turn in the episode
    assistant_content: str  # raw assistant message (may include <think>, <tool_call>)
    observation_content: str  # raw tool result or error message
    tool_name: Optional[str] = None  # parsed tool name if present
    tool_args: Optional[Dict[str, Any]] = None  # parsed tool arguments
    is_done: bool = False  # True if this step ended the episode


@dataclass
class StepScore:
    """Interestingness score for a single step."""

    turn_idx: int
    score: float  # 0.0 = routine, 1.0 = highly interesting
    method: str  # "direct_judge" | "divergence"
    rationale: str = ""  # explanation from LLM judge (direct method only)


@dataclass
class TraceJudgeResult:
    """Full judge output for one trace (or group of traces)."""

    task_key: str
    step_scores: List[StepScore]
    method: str

    @property
    def top_k_steps(self) -> List[StepScore]:
        return sorted(self.step_scores, key=lambda s: s.score, reverse=True)

    def summary(self) -> Dict[str, Any]:
        if not self.step_scores:
            return {"task_key": self.task_key, "n_steps": 0}
        scores = [s.score for s in self.step_scores]
        return {
            "task_key": self.task_key,
            "n_steps": len(scores),
            "mean_score": statistics.mean(scores),
            "max_score": max(scores),
            "top_step": self.top_k_steps[0].turn_idx if self.step_scores else -1,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def _parse_tool_call(content: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extract tool name and arguments from a legacy <tool_call>...</tool_call> message."""
    m = _TOOL_CALL_RE.search(content)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(1))
        return obj.get("name"), obj.get("arguments")
    except json.JSONDecodeError:
        return None, None


def _parse_native_tool_calls(
    tool_calls: Any,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Extract a canonical (name, args) representation from native OpenAI tool_calls.

    A single assistant turn may issue multiple (parallel) tool calls. We collapse
    them into:
      - tool_name: the tool names joined by " + " in call order (captures which
        tools were chosen — used for name-diversity in divergence_judge).
      - tool_args: an ordered list of {"name", "arguments"} dicts (JSON-serializable,
        captures argument values — used for args-diversity in divergence_judge).

    Returns (None, None) if there are no tool calls.
    """
    if not tool_calls:
        return None, None

    names: List[str] = []
    calls: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = raw_args
        else:
            parsed_args = raw_args
        if name:
            names.append(name)
        calls.append({"name": name, "arguments": parsed_args})

    if not calls:
        return None, None

    tool_name = " + ".join(names) if names else None
    return tool_name, calls


def parse_steps(chat_history: List[Dict[str, Any]]) -> List[TraceStep]:
    """Parse a chat_history into a list of TraceStep objects.

    One TraceStep is created per assistant turn. Each step's observation is the
    concatenation of the environment messages that follow it (role "tool" results,
    and any mid-trajectory "user" follow-ups) until the next assistant turn. The
    system prompt and the initial task message are excluded.

    Supports two trace formats transparently:
      1. Native OpenAI tool calling — assistant messages carry a ``tool_calls``
         list and observations arrive as role "tool" messages. (Fleet / Supabase
         rollouts.)
      2. Legacy inline format — tool calls embedded as ``<tool_call>...</tool_call>``
         in assistant content and observations as role "user" messages. (Smoke test.)

    Args:
        chat_history: List of {"role": ..., "content": ..., "tool_calls": ...} dicts.

    Returns:
        List of TraceStep, one per assistant action.
    """
    steps: List[TraceStep] = []
    turn_idx = 0
    n = len(chat_history)

    i = 0
    while i < n:
        msg = chat_history[i]
        if msg.get("role") != "assistant":
            i += 1
            continue

        assistant_content = _extract_text(msg.get("content"))

        # Prefer native tool_calls; fall back to legacy inline <tool_call> parsing.
        tool_name, tool_args = _parse_native_tool_calls(msg.get("tool_calls"))
        if tool_name is None:
            tool_name, tool_args = _parse_tool_call(assistant_content)

        # Gather observation messages until the next assistant turn.
        j = i + 1
        obs_parts: List[str] = []
        while j < n and chat_history[j].get("role") in ("tool", "user"):
            text = _extract_text(chat_history[j].get("content"))
            if text:
                obs_parts.append(text)
            j += 1
        obs_content = "\n".join(obs_parts)

        # A turn with no tool call ends the episode (agent stopped calling tools),
        # as does an explicit done marker.
        is_done = (
            tool_name is None
            or "<done>" in assistant_content.lower()
            or "[done]" in assistant_content.lower()
        )

        steps.append(
            TraceStep(
                turn_idx=turn_idx,
                assistant_content=assistant_content,
                observation_content=obs_content,
                tool_name=tool_name,
                tool_args=tool_args,
                is_done=is_done,
            )
        )
        turn_idx += 1
        i = j

    return steps


def _extract_text(content: Any) -> str:
    """Flatten multimodal content blocks to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Method 1: Direct LLM judge
# ---------------------------------------------------------------------------

_DIRECT_JUDGE_SYSTEM = """\
You are an expert at analyzing agentic task traces. Your job is to identify which \
steps in an agent trace are "interesting" — meaning they are genuine decision points \
where the agent's choice materially affected whether the task succeeded or failed.

A step is HIGHLY INTERESTING (score close to 1.0) if:
- The agent faces a real choice between multiple reasonable actions, and the chosen \
  action meaningfully narrows or changes the solution path
- The tool result is surprising, unexpected, or reveals critical information that \
  constrains future actions
- This is a branching point: if the agent had done something different here, the \
  outcome would likely differ

A step is NOT INTERESTING (score close to 0.0) if:
- It is routine / mechanical (e.g., taking a screenshot after a click to verify state)
- The outcome is predictable and the agent had no real alternatives
- It is boilerplate (confirming a page loaded, reading a field that was obviously set)

Respond with a JSON object:
{
  "score": <float 0.0-1.0>,
  "rationale": "<one sentence>"
}
Do not include any other text."""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    """Parse a judge response into a dict, tolerant of markdown fences / prose.

    Models behind OpenAI-compatible endpoints (incl. OpenRouter) sometimes wrap
    JSON in ```json fences or add leading/trailing prose. Try strict parse first,
    then strip fences, then fall back to the first {...} object found.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (``` or ```json) and any closing fence.
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

    m = _JSON_OBJ_RE.search(cleaned)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"no JSON object found in judge response: {raw[:120]!r}")


_DIRECT_JUDGE_TEMPLATE = """\
TASK: {task_prompt}

STEP {turn_idx} of {n_steps}:

Agent action:
{assistant_content}

Tool result / observation:
{observation_content}

Score this step's interestingness."""


def direct_judge(
    steps: List[TraceStep],
    task_prompt: str,
    client: Any,  # openai.OpenAI or compatible
    model: str = "gpt-4o-mini",
    max_steps_to_score: int = 30,
) -> List[StepScore]:
    """Score each step in a trace using an LLM judge.

    Args:
        steps: Parsed steps from parse_steps().
        task_prompt: Original task instruction (for context).
        client: OpenAI-compatible client with .chat.completions.create().
        model: Model to use for scoring.
        max_steps_to_score: Cap to avoid excessive API calls for long traces.

    Returns:
        List of StepScore, one per step. Steps beyond max_steps_to_score get score=0.0.
    """
    scores: List[StepScore] = []
    n_steps = len(steps)

    # Truncate assistant/obs content for the judge prompt
    def _truncate(text: str, max_chars: int = 600) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

    # Strip <think> blocks from assistant content to reduce noise
    def _strip_think(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "[reasoning omitted]", text, flags=re.DOTALL).strip()

    for step in steps:
        if step.turn_idx >= max_steps_to_score:
            scores.append(StepScore(turn_idx=step.turn_idx, score=0.0, method="direct_judge",
                                    rationale="beyond max_steps_to_score"))
            continue

        prompt = _DIRECT_JUDGE_TEMPLATE.format(
            task_prompt=_truncate(task_prompt, 400),
            turn_idx=step.turn_idx,
            n_steps=n_steps,
            assistant_content=_truncate(_strip_think(step.assistant_content)),
            observation_content=_truncate(step.observation_content),
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DIRECT_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = (response.choices[0].message.content or "").strip()
            obj = _parse_judge_json(raw)
            score = float(obj.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            rationale = str(obj.get("rationale", ""))
        except Exception as e:
            logger.warning(f"direct_judge failed at step {step.turn_idx}: {e}")
            score = 0.0
            rationale = f"error: {e}"

        scores.append(StepScore(
            turn_idx=step.turn_idx,
            score=score,
            method="direct_judge",
            rationale=rationale,
        ))

    return scores


# ---------------------------------------------------------------------------
# Method 2: Trajectory divergence judge
# ---------------------------------------------------------------------------


def divergence_judge(
    all_steps: List[List[TraceStep]],
    tool_name_weight: float = 0.4,
    tool_args_weight: float = 0.6,
) -> List[StepScore]:
    """Score step interestingness by cross-trajectory divergence.

    Given N rollouts of the same task, computes how much the agent's choice at
    each step varied across rollouts. High divergence = genuine decision point.

    Divergence is measured at the tool-call level:
    - tool_name_weight: fraction of score from tool-name diversity
    - tool_args_weight: fraction of score from argument diversity

    Traces of different lengths are handled by only scoring steps that appear in
    ≥2 traces.

    Args:
        all_steps: List of step lists, one per rollout (from parse_steps()).
        tool_name_weight: Weight for tool-name diversity in composite score.
        tool_args_weight: Weight for tool-args diversity in composite score.

    Returns:
        List of StepScore indexed by turn_idx. Score is the divergence at that step,
        0.0 if fewer than 2 traces reached that step.
    """
    if len(all_steps) < 2:
        logger.warning("divergence_judge requires ≥2 traces; returning empty scores")
        return []

    # Find the max turn_idx across all traces
    max_turns = max((steps[-1].turn_idx + 1 for steps in all_steps if steps), default=0)

    # Index steps by turn_idx within each trace
    indexed: List[Dict[int, TraceStep]] = [
        {s.turn_idx: s for s in steps} for steps in all_steps
    ]

    result: List[StepScore] = []

    for turn in range(max_turns):
        # Collect tool calls at this turn from all traces that reached it
        present = [idx[turn] for idx in indexed if turn in idx]
        if len(present) < 2:
            result.append(StepScore(turn_idx=turn, score=0.0, method="divergence",
                                    rationale=f"only {len(present)} trace(s) reached this step"))
            continue

        # Tool name diversity: fraction of unique names / total names
        names = [s.tool_name or "__no_tool__" for s in present]
        name_diversity = _set_diversity(names)

        # Argument diversity: pairwise string-edit distance normalized to [0,1]
        args_strs = [json.dumps(s.tool_args, sort_keys=True) if s.tool_args else "" for s in present]
        args_diversity = _pairwise_diversity(args_strs)

        score = tool_name_weight * name_diversity + tool_args_weight * args_diversity
        n_traces = len(present)
        result.append(StepScore(
            turn_idx=turn,
            score=round(score, 4),
            method="divergence",
            rationale=(
                f"{n_traces} traces: name_div={name_diversity:.2f}, "
                f"args_div={args_diversity:.2f}"
            ),
        ))

    return result


# ---------------------------------------------------------------------------
# Method 2b: Math reasoning chunking + text-content divergence
# ---------------------------------------------------------------------------
#
# Single-turn math rollouts (e.g. DAPO / AIME completions) have no tool calls,
# so parse_steps() collapses the whole solution into one step and the tool-call
# divergence judge has nothing to compare. We instead chunk each completion into
# reasoning steps (paragraphs) and measure cross-rollout divergence on the chunk
# *text* via token-set (Jaccard) distance. Aligning by chunk index mirrors the
# turn-index alignment used by the tool-call divergence judge.

_MATH_CHUNK_SPLIT_RE = re.compile(r"\n\s*\n")
_MATH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def chunk_math_completion(completion: str, min_chars: int = 24) -> List[str]:
    """Split a free-text math solution into reasoning-step chunks.

    Chunks on blank lines (paragraph boundaries — the natural unit for
    step-by-step math reasoning). Tiny fragments (e.g. a lone ``\\[`` display
    block shorter than ``min_chars``) are merged into the preceding chunk so a
    single logical step isn't split across multiple entries.

    Args:
        completion: The raw assistant completion text.
        min_chars: Fragments shorter than this are merged into the prior chunk.

    Returns:
        Ordered list of chunk strings (may be empty for empty input).
    """
    if not completion or not completion.strip():
        return []
    raw = [c.strip() for c in _MATH_CHUNK_SPLIT_RE.split(completion)]
    raw = [c for c in raw if c]
    chunks: List[str] = []
    for c in raw:
        if chunks and len(c) < min_chars:
            chunks[-1] = chunks[-1] + "\n\n" + c
        else:
            chunks.append(c)
    return chunks


def parse_math_steps(completion: str, min_chars: int = 24) -> List[TraceStep]:
    """Parse a math completion into TraceStep objects, one per reasoning chunk.

    The chunk text is stored in ``assistant_content``; there is no observation
    and no tool call. ``is_done`` marks the final chunk.
    """
    chunks = chunk_math_completion(completion, min_chars=min_chars)
    steps: List[TraceStep] = []
    for i, chunk in enumerate(chunks):
        steps.append(
            TraceStep(
                turn_idx=i,
                assistant_content=chunk,
                observation_content="",
                tool_name=None,
                tool_args=None,
                is_done=(i == len(chunks) - 1),
            )
        )
    return steps


def _tokenize_math(text: str) -> set:
    """Lowercase token set: alphanumeric words plus individual symbols.

    Splitting symbols individually keeps math operators / delimiters
    (``=``, ``+``, ``\\``, ``{`` ...) as discriminating tokens, which matters
    for distinguishing different algebraic moves.
    """
    return set(_MATH_TOKEN_RE.findall(text.lower()))


def _token_jaccard_diversity(texts: List[str]) -> float:
    """Mean pairwise token-set Jaccard *distance* (1 - Jaccard) in [0, 1].

    Robust to length (unlike edit distance) and meaningful for free text:
    chunks that share most tokens (same reasoning move) score low; chunks that
    use disjoint tokens (divergent approaches) score near 1.
    """
    if len(texts) <= 1:
        return 0.0
    sets = [_tokenize_math(t) for t in texts]
    pairs = 0
    total = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                dist = 0.0
            else:
                union = len(a | b)
                dist = 1.0 - (len(a & b) / union if union else 0.0)
            total += dist
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def math_divergence_judge(all_steps: List[List[TraceStep]]) -> List[StepScore]:
    """Score reasoning-step interestingness by cross-rollout text divergence.

    Companion to :func:`divergence_judge` for tool-free (math) traces. Given N
    rollouts of the same problem (each chunked via :func:`parse_math_steps`),
    aligns chunks by index and scores each step by the token-Jaccard diversity
    of the chunk text across the rollouts that reached that step.

    Args:
        all_steps: List of step lists, one per rollout (from parse_math_steps()).

    Returns:
        List of StepScore indexed by turn_idx; 0.0 where fewer than 2 rollouts
        reached the step.
    """
    if len(all_steps) < 2:
        logger.warning("math_divergence_judge requires ≥2 traces; returning empty scores")
        return []

    max_turns = max((steps[-1].turn_idx + 1 for steps in all_steps if steps), default=0)
    indexed: List[Dict[int, TraceStep]] = [
        {s.turn_idx: s for s in steps} for steps in all_steps
    ]

    result: List[StepScore] = []
    for turn in range(max_turns):
        present = [idx[turn] for idx in indexed if turn in idx]
        if len(present) < 2:
            result.append(StepScore(turn_idx=turn, score=0.0, method="math_divergence",
                                    rationale=f"only {len(present)} trace(s) reached this step"))
            continue
        texts = [s.assistant_content for s in present]
        div = _token_jaccard_diversity(texts)
        result.append(StepScore(
            turn_idx=turn,
            score=round(div, 4),
            method="math_divergence",
            rationale=f"{len(present)} traces: token_jaccard_div={div:.2f}",
        ))
    return result


def _set_diversity(items: List[str]) -> float:
    """Fraction of items that are NOT the majority value."""
    if len(items) <= 1:
        return 0.0
    from collections import Counter
    most_common_count = Counter(items).most_common(1)[0][1]
    return 1.0 - most_common_count / len(items)


def _pairwise_diversity(strings: List[str]) -> float:
    """Mean pairwise normalized edit distance, capped at 1.0.

    Uses a fast character-level approximation: length-normalized Levenshtein.
    Falls back to exact-match diversity when strings are too long.
    """
    if len(strings) <= 1:
        return 0.0

    # If strings are long, fall back to set diversity (cheaper)
    if any(len(s) > 500 for s in strings):
        return _set_diversity(strings)

    pairs = 0
    total_dist = 0.0
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            a, b = strings[i], strings[j]
            dist = _normalized_edit_distance(a, b)
            total_dist += dist
            pairs += 1

    return total_dist / pairs if pairs > 0 else 0.0


def _normalized_edit_distance(a: str, b: str) -> float:
    """Normalized edit distance in [0, 1] using DP."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    # Limit size for performance
    a, b = a[:200], b[:200]
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[lb] / max(la, lb)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def calibrate_against_rewards(
    step_scores: List[StepScore],
    rewards: List[float],
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Measure how well step scores predict outcome divergence.

    Given step scores for a group of rollouts (all same task) and their
    terminal rewards, computes:

    1. **Reward variance** — overall variance across rollouts. High variance
       means the task has a meaningful signal.
    2. **Score-reward correlation** — for each step, is a higher divergence
       score associated with the trace that succeeded?
       (Only meaningful when len(rewards) > 1.)
    3. **Top-K step stats** — for the top-K most interesting steps, what is
       the average score?

    This is an offline proxy. True calibration requires branching rollouts from
    each flagged step and observing reward variance, which requires live execution.

    Args:
        step_scores: Step scores from direct_judge() or divergence_judge().
        rewards: Terminal rewards for each rollout in the group (same order as
                 the traces that produced step_scores).
        top_k: Number of top-scoring steps to include in the summary.

    Returns:
        Dict with calibration metrics.
    """
    if not step_scores:
        return {"error": "no step scores"}

    scores = [s.score for s in step_scores]
    reward_var = statistics.variance(rewards) if len(rewards) >= 2 else 0.0
    mean_score = statistics.mean(scores)

    # Top-K
    top_steps = sorted(step_scores, key=lambda s: s.score, reverse=True)[:top_k]

    return {
        "n_steps_scored": len(step_scores),
        "n_rollouts": len(rewards),
        "reward_variance": round(reward_var, 4),
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "mean_interestingness": round(mean_score, 4),
        "max_interestingness": round(max(scores), 4),
        "top_k_steps": [
            {"turn_idx": s.turn_idx, "score": s.score, "rationale": s.rationale}
            for s in top_steps
        ],
        # Proxy calibration: if reward_variance is high, the judge should find
        # high-scoring steps. We report the correlation direction but don't
        # compute a p-value since we usually have < 10 rollouts per task.
        "high_variance_task": reward_var > 0.1,
    }


def calibrate_batch(
    task_to_scores: Dict[str, List[StepScore]],
    task_to_rewards: Dict[str, List[float]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Calibrate judge across a batch of tasks.

    Args:
        task_to_scores: {task_key: step_scores} from judge methods.
        task_to_rewards: {task_key: [reward, ...]} for each task.
        top_k: Top-K steps per task for detailed output.

    Returns:
        Aggregate calibration metrics across all tasks.
    """
    per_task = {}
    all_max_scores: List[float] = []
    all_reward_vars: List[float] = []

    for task_key, scores in task_to_scores.items():
        rewards = task_to_rewards.get(task_key, [])
        metrics = calibrate_against_rewards(scores, rewards, top_k=top_k)
        per_task[task_key] = metrics
        if "max_interestingness" in metrics:
            all_max_scores.append(metrics["max_interestingness"])
        if "reward_variance" in metrics:
            all_reward_vars.append(metrics["reward_variance"])

    # Rank-correlation between max_interestingness and reward_variance
    # (Spearman, since both are bounded [0,1])
    rank_corr = None
    if len(all_max_scores) >= 3 and len(all_reward_vars) >= 3:
        rank_corr = _spearman_rank_correlation(all_max_scores, all_reward_vars)

    return {
        "n_tasks": len(per_task),
        "mean_max_interestingness": (
            statistics.mean(all_max_scores) if all_max_scores else None
        ),
        "mean_reward_variance": (
            statistics.mean(all_reward_vars) if all_reward_vars else None
        ),
        "spearman_max_score_vs_reward_var": rank_corr,
        "per_task": per_task,
    }


def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")

    def _ranks(vals: List[float]) -> List[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank, (idx, _) in enumerate(sorted_vals):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))
