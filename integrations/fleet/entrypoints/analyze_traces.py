"""Offline trace interestingness analysis (agu/vf).

Loads saved trajectories (JSONL of {task_key, chat_history, reward, env_key}),
runs both judge methods, reports calibration metrics, and writes results.

Usage:
    # With direct LLM judge (requires OPENAI_API_KEY or compatible endpoint)
    python -m integrations.fleet.entrypoints.analyze_traces \\
        --traces ~/data/fleet/traces/booking_traces.jsonl \\
        --method both \\
        --model gpt-4o-mini \\
        --output ~/data/fleet/analysis/judge_results.json

    # Divergence only (no API key needed)
    python -m integrations.fleet.entrypoints.analyze_traces \\
        --traces ~/data/fleet/traces/booking_traces.jsonl \\
        --method divergence \\
        --output ~/data/fleet/analysis/judge_results.json

    # Generate synthetic traces for smoke-testing without real rollouts
    python -m integrations.fleet.entrypoints.analyze_traces --smoke-test

Trace JSONL format (one JSON object per line):
    {
        "task_key": "task_abc123",
        "env_key": "booking",
        "chat_history": [...],   // FleetTaskEnv format
        "reward": 0.0            // terminal verifier reward
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow running as script from repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrations.fleet.trace_judge import (
    TraceJudgeResult,
    StepScore,
    calibrate_batch,
    direct_judge,
    divergence_judge,
    parse_steps,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_traces(path: str) -> List[Dict[str, Any]]:
    """Load traces from a JSONL file."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Traces file not found: {expanded}")
    records = []
    with open(expanded) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {lineno}: JSON decode error — {e}")
    logger.info(f"Loaded {len(records)} traces from {expanded}")
    return records


def group_by_task(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group trace records by task_key."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["task_key"]].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_divergence_analysis(
    task_groups: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[StepScore]]:
    """Run divergence judge for all tasks that have ≥2 rollouts."""
    task_to_scores: Dict[str, List[StepScore]] = {}
    skipped = 0

    for task_key, records in task_groups.items():
        if len(records) < 2:
            skipped += 1
            continue
        all_steps = [parse_steps(r["chat_history"]) for r in records]
        scores = divergence_judge(all_steps)
        task_to_scores[task_key] = scores

    if skipped:
        logger.info(f"Divergence: skipped {skipped} tasks with only 1 rollout")
    logger.info(f"Divergence: scored {len(task_to_scores)} tasks")
    return task_to_scores


def run_direct_analysis(
    task_groups: Dict[str, List[Dict[str, Any]]],
    model: str,
    base_url: Optional[str],
    max_tasks: int = 50,
) -> Dict[str, List[StepScore]]:
    """Run direct LLM judge for all tasks (first rollout per task only)."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    task_to_scores: Dict[str, List[StepScore]] = {}
    tasks = list(task_groups.items())[:max_tasks]

    logger.info(f"Direct judge: scoring {len(tasks)} tasks with {model}")
    for i, (task_key, records) in enumerate(tasks):
        rec = records[0]  # first rollout per task
        steps = parse_steps(rec["chat_history"])
        # Extract task prompt from chat_history (second message = initial user msg)
        task_prompt = ""
        hist = rec["chat_history"]
        for msg in hist:
            if msg["role"] == "user":
                content = msg["content"]
                task_prompt = content if isinstance(content, str) else str(content)[:500]
                break

        scores = direct_judge(steps, task_prompt, client=client, model=model)
        task_to_scores[task_key] = scores

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{len(tasks)} tasks scored")

    logger.info(f"Direct judge: completed {len(task_to_scores)} tasks")
    return task_to_scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_top_findings(
    task_to_scores: Dict[str, List[StepScore]],
    task_to_rewards: Dict[str, List[float]],
    task_groups: Dict[str, List[Dict[str, Any]]],
    method: str,
    top_n_tasks: int = 5,
) -> None:
    """Print a human-readable summary of the most interesting findings."""

    print(f"\n{'='*60}")
    print(f"METHOD: {method.upper()}")
    print(f"{'='*60}")

    # Sort tasks by max interestingness score
    ranked = sorted(
        task_to_scores.items(),
        key=lambda kv: max((s.score for s in kv[1]), default=0.0),
        reverse=True,
    )

    for task_key, scores in ranked[:top_n_tasks]:
        rewards = task_to_rewards.get(task_key, [])
        env_key = task_groups[task_key][0].get("env_key", "?")
        top = sorted(scores, key=lambda s: s.score, reverse=True)[:3]

        print(f"\nTask: {task_key} [{env_key}]")
        print(f"  Rollouts: {len(rewards)}, rewards: {[round(r, 2) for r in rewards]}")
        print(f"  Top interesting steps:")
        for s in top:
            print(f"    turn {s.turn_idx:>2d}  score={s.score:.3f}  {s.rationale[:80]}")


def write_results(
    output_path: str,
    divergence_results: Dict[str, Any],
    direct_results: Optional[Dict[str, Any]],
) -> None:
    expanded = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
    out = {"divergence": divergence_results}
    if direct_results:
        out["direct_judge"] = direct_results
    with open(expanded, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results written to {expanded}")


# ---------------------------------------------------------------------------
# Smoke test (synthetic traces, no API key needed)
# ---------------------------------------------------------------------------


def _make_synthetic_trace(
    task_key: str,
    reward: float,
    variant: int = 0,
) -> Dict[str, Any]:
    """Build a synthetic chat_history for smoke-testing."""
    tool_options = [
        ("search_hotels", {"city": "Austin", "stars": 5}),
        ("search_hotels", {"city": "Austin", "stars": 4}),  # variant: different args
        ("get_hotel_details", {"hotel_id": 9992}),
        ("book_hotel", {"hotel_id": 9992, "check_in": "2026-04-13"}),
        ("confirm_booking", {"booking_id": 25595}),
    ]
    # Variant 0: takes booking path. Variant 1: different search args.
    steps_for_variant = tool_options if variant == 0 else [
        ("search_hotels", {"city": "Austin", "stars": 3}),  # diverges here
        ("get_hotel_details", {"hotel_id": 1234}),
        ("book_hotel", {"hotel_id": 1234, "check_in": "2026-04-13"}),
        ("confirm_booking", {"booking_id": 99999}),
    ]

    history = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Book a 5-star hotel in Austin for tonight."},
    ]
    for tool_name, args in steps_for_variant:
        history.append({
            "role": "assistant",
            "content": f"<tool_call>{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(args)}}}</tool_call>",
        })
        history.append({
            "role": "user",
            "content": f"Tool result: {{'status': 'ok', 'data': {json.dumps(args)}}}",
        })
    history.append({"role": "assistant", "content": "Task complete. <done>"})

    return {
        "task_key": task_key,
        "env_key": "booking",
        "chat_history": history,
        "reward": reward,
    }


def run_smoke_test() -> None:
    print("Running smoke test with synthetic traces...\n")

    # 3 tasks, 2 rollouts each, varying rewards
    records = [
        _make_synthetic_trace("task_001", reward=1.0, variant=0),
        _make_synthetic_trace("task_001", reward=0.0, variant=1),  # diverges at step 0
        _make_synthetic_trace("task_002", reward=1.0, variant=0),
        _make_synthetic_trace("task_002", reward=1.0, variant=0),  # identical → no divergence
        _make_synthetic_trace("task_003", reward=0.0, variant=0),
        _make_synthetic_trace("task_003", reward=0.5, variant=1),
    ]

    task_groups = group_by_task(records)
    task_to_rewards = {
        k: [r["reward"] for r in v] for k, v in task_groups.items()
    }

    # Divergence judge
    div_scores = run_divergence_analysis(task_groups)
    div_cal = calibrate_batch(div_scores, task_to_rewards)

    print("=== Divergence Judge Results ===")
    for task_key, scores in div_scores.items():
        rewards = task_to_rewards[task_key]
        print(f"\n{task_key}  rewards={rewards}")
        for s in sorted(scores, key=lambda x: x.score, reverse=True)[:3]:
            print(f"  turn {s.turn_idx:>2d}  score={s.score:.3f}  {s.rationale}")

    print(f"\n=== Aggregate Calibration ===")
    print(f"  tasks scored:        {div_cal['n_tasks']}")
    print(f"  mean max score:      {div_cal['mean_max_interestingness']:.3f}")
    print(f"  mean reward var:     {div_cal['mean_reward_variance']:.3f}")
    print(f"  spearman(score,var): {div_cal['spearman_max_score_vs_reward_var']}")

    print("\nExpected: task_001 and task_003 have divergence at step 0 (search args differ).")
    print("task_002 should have near-zero divergence (identical rollouts).")
    print("\nSmoke test complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace interestingness analysis")
    parser.add_argument("--traces", help="Path to JSONL trace file")
    parser.add_argument(
        "--method",
        choices=["divergence", "direct", "both"],
        default="divergence",
        help="Which judge method to use (default: divergence)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model for direct judge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL for OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=50,
        help="Max tasks to score with direct judge (default: 50)",
    )
    parser.add_argument(
        "--output",
        default="~/data/fleet/analysis/judge_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with synthetic traces (no real data or API key needed)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if not args.traces:
        parser.error("--traces is required unless --smoke-test is set")

    records = load_traces(args.traces)
    task_groups = group_by_task(records)
    task_to_rewards = {
        k: [r["reward"] for r in v] for k, v in task_groups.items()
    }

    logger.info(
        f"Loaded {len(records)} traces across {len(task_groups)} tasks "
        f"({sum(1 for v in task_groups.values() if len(v) >= 2)} with ≥2 rollouts)"
    )

    div_scores: Dict[str, List[StepScore]] = {}
    direct_scores: Dict[str, List[StepScore]] = {}

    if args.method in ("divergence", "both"):
        div_scores = run_divergence_analysis(task_groups)
        div_cal = calibrate_batch(div_scores, task_to_rewards)
        print_top_findings(div_scores, task_to_rewards, task_groups, "divergence")
        print(f"\nAggregate (divergence):")
        print(f"  tasks scored:        {div_cal['n_tasks']}")
        print(f"  mean max score:      {div_cal.get('mean_max_interestingness', 'N/A')}")
        print(f"  mean reward var:     {div_cal.get('mean_reward_variance', 'N/A')}")
        print(f"  spearman(score,var): {div_cal.get('spearman_max_score_vs_reward_var', 'N/A')}")

    if args.method in ("direct", "both"):
        direct_scores = run_direct_analysis(
            task_groups,
            model=args.model,
            base_url=args.base_url,
            max_tasks=args.max_tasks,
        )
        if direct_scores:
            dir_cal = calibrate_batch(direct_scores, task_to_rewards)
            print_top_findings(direct_scores, task_to_rewards, task_groups, "direct_judge")
            print(f"\nAggregate (direct_judge):")
            print(f"  tasks scored:        {dir_cal['n_tasks']}")
            print(f"  mean max score:      {dir_cal.get('mean_max_interestingness', 'N/A')}")
            print(f"  mean reward var:     {dir_cal.get('mean_reward_variance', 'N/A')}")
            print(f"  spearman(score,var): {dir_cal.get('spearman_max_score_vs_reward_var', 'N/A')}")

    # Serialize for output (convert StepScore dataclasses)
    def _scores_to_dict(scores_map):
        return {
            tk: [{"turn_idx": s.turn_idx, "score": s.score, "rationale": s.rationale}
                 for s in scores]
            for tk, scores in scores_map.items()
        }

    write_results(
        args.output,
        divergence_results=_scores_to_dict(div_scores) if div_scores else {},
        direct_results=_scores_to_dict(direct_scores) if direct_scores else None,
    )


if __name__ == "__main__":
    main()
