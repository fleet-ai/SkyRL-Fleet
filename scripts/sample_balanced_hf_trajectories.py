#!/usr/bin/env python3
"""Sample or normalize trajectories from the Claude HF-style dump.

The output is normalized to the eval JSONL schema consumed by
scripts/run_taste_judge_local.py:
  - input_prompt
  - output_response
  - score
  - data_source
  - env_extras
  - image_paths
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def score_value(row: dict[str, Any]) -> float:
    raw = row.get("score", 0.0)
    if isinstance(raw, list):
        vals = []
        for item in raw:
            try:
                vals.append(float(item))
            except Exception:
                pass
        return max(vals) if vals else 0.0
    try:
        return float(raw or 0.0)
    except Exception:
        return 0.0


def first_user_text(conversation: list[dict[str, Any]]) -> str:
    for msg in conversation:
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return ""


def format_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=True)
    except TypeError:
        return str(tool_calls)


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=True)
    except TypeError:
        return str(content)


def format_conversation(conversation: list[dict[str, Any]]) -> str:
    parts = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = normalize_content(msg.get("content"))
        pos = msg.get("position")
        prefix = f"[{pos}] {role}" if pos is not None else str(role)
        if content:
            parts.append(f"{prefix}: {content}")
        tool_calls = format_tool_calls(msg.get("tool_calls"))
        if tool_calls:
            parts.append(f"{prefix} tool_calls: {tool_calls}")
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            parts.append(f"{prefix} tool_call_id: {tool_call_id}")
    return "\n\n".join(parts)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    conversation = row.get("conversation") or []
    env_key = row.get("env_key") or row.get("data_source") or ""
    task_key = row.get("task_key") or row.get("session_id") or ""
    return {
        "input_prompt": first_user_text(conversation),
        "output_response": format_conversation(conversation),
        "score": row.get("score", 0.0),
        "stop_reason": row.get("outcome", ""),
        "env_class": "fleet_task",
        "env_extras": {
            "task_key": task_key,
            "session_id": row.get("session_id", ""),
            "data_source": env_key,
            "model": row.get("model", ""),
            "outcome": row.get("outcome", ""),
            "num_turns": row.get("num_turns"),
            "num_messages": row.get("num_messages"),
        },
        "data_source": env_key,
        "image_paths": row.get("image_paths") or [],
        "num_screenshots": row.get("num_screenshots", len(row.get("image_paths") or [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF-style Claude trajectories JSONL")
    parser.add_argument("--out", required=True, help="Normalized balanced output JSONL")
    parser.add_argument("--n-success", type=int, default=50)
    parser.add_argument("--n-failure", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--env-key", action="append", help="Optional env_key filter; can be repeated.")
    parser.add_argument("--all", action="store_true", help="Normalize all rows after optional env filtering instead of balanced sampling.")
    args = parser.parse_args()

    successes = []
    failures = []
    all_rows = []
    env_filter = set(args.env_key or [])
    with Path(args.input).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if env_filter and row.get("env_key") not in env_filter:
                continue
            all_rows.append(row)
            if score_value(row) >= args.success_threshold:
                successes.append(row)
            else:
                failures.append(row)

    rng = random.Random(args.seed)
    if args.all:
        sampled = [(1 if score_value(row) >= args.success_threshold else 0, row) for row in all_rows]
        n_success_written = sum(label for label, _ in sampled)
        n_failure_written = len(sampled) - n_success_written
    else:
        if len(successes) < args.n_success:
            raise SystemExit(f"Need {args.n_success} successes, found {len(successes)}")
        if len(failures) < args.n_failure:
            raise SystemExit(f"Need {args.n_failure} failures, found {len(failures)}")
        sampled_successes = rng.sample(successes, args.n_success)
        sampled_failures = rng.sample(failures, args.n_failure)
        sampled = [(1, row) for row in sampled_successes] + [(0, row) for row in sampled_failures]
        rng.shuffle(sampled)
        n_success_written = args.n_success
        n_failure_written = args.n_failure

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for _, row in sampled:
            out.write(json.dumps(normalize_row(row), ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "input": args.input,
                "out": str(out_path),
                "seed": args.seed,
                "n_success_available": len(successes),
                "n_failure_available": len(failures),
                "n_success_written": n_success_written,
                "n_failure_written": n_failure_written,
                "n_total_written": len(sampled),
                "mode": "all" if args.all else "balanced",
                "success_threshold": args.success_threshold,
                "env_filter": sorted(env_filter),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
