#!/usr/bin/env python3
"""Prepare gym-anything task index as SkyRL parquet dataset.

Converts the task index JSON (from build_task_index.py) into train/validation
parquet files compatible with SkyRL's data pipeline.

Usage:
    python -m integrations.fleet.prepare_gym_anything_dataset \
        --tasks-json tasks_gym_anything_train.json \
        --output-dir data/fleet/gym_anything
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def prepare_dataset(
    tasks_json: str,
    output_dir: str,
    eval_fraction: float = 0.1,
    max_tasks: Optional[int] = None,
    seed: int = 42,
):
    """Convert gym-anything task index to SkyRL parquet dataset."""
    with open(tasks_json) as f:
        tasks = json.load(f)

    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    print(f"Loaded {len(tasks)} tasks from {tasks_json}")

    if max_tasks and len(tasks) > max_tasks:
        random.seed(seed)
        tasks = random.sample(tasks, max_tasks)
        print(f"Sampled {max_tasks} tasks")

    # Group by environment for balanced splitting
    env_tasks: Dict[str, List[Dict]] = {}
    for t in tasks:
        env_name = t.get("env_name", "unknown")
        env_tasks.setdefault(env_name, []).append(t)

    train_records = []
    eval_records = []

    random.seed(seed)
    for env_name, etasks in sorted(env_tasks.items()):
        random.shuffle(etasks)
        n_eval = max(1, int(len(etasks) * eval_fraction))
        if len(etasks) <= 2:
            # Too few tasks — all go to train
            for t in etasks:
                train_records.append(_task_to_record(t))
        else:
            for t in etasks[:n_eval]:
                eval_records.append(_task_to_record(t))
            for t in etasks[n_eval:]:
                train_records.append(_task_to_record(t))

    print(f"Train: {len(train_records)} records, Eval: {len(eval_records)} records")
    print(f"Environments: {len(env_tasks)}")

    os.makedirs(output_dir, exist_ok=True)
    _write_parquet(train_records, os.path.join(output_dir, "train.parquet"))
    _write_parquet(eval_records, os.path.join(output_dir, "validation.parquet"))
    print(f"Written to {output_dir}/")


def _task_to_record(task: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a gym-anything task dict to a SkyRL dataset record."""
    description = task.get("description", "Complete the task.")
    return {
        "prompt": [{"role": "user", "content": description}],
        "env_class": "gym_anything",
        "task_key": task["task_key"],
        "env_dir": task["env_dir"],
        "task_id": task["task_id"],
        "env_name": task.get("env_name", ""),
        "max_turns": task.get("max_turns", 50),
        "data_source": task.get("env_name", "unknown"),
    }


def _write_parquet(records: List[Dict[str, Any]], path: str):
    """Write records to parquet with SkyRL-compatible schema."""
    if not records:
        # Write empty parquet with correct schema
        records = []

    # Serialize prompt as JSON string (SkyRL convention)
    rows = []
    for r in records:
        row = dict(r)
        row["prompt"] = json.dumps(row["prompt"])
        rows.append(row)

    table = pa.table({k: [r.get(k) for r in rows] for k in rows[0].keys()}) if rows else pa.table({})
    pq.write_table(table, path)
    print(f"  {path}: {len(rows)} records")


def main():
    parser = argparse.ArgumentParser(description="Prepare gym-anything dataset for SkyRL")
    parser.add_argument("--tasks-json", required=True, help="Path to gym-anything task index JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Fraction of tasks for eval")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    prepare_dataset(args.tasks_json, args.output_dir, args.eval_fraction, args.max_tasks, args.seed)


if __name__ == "__main__":
    main()
