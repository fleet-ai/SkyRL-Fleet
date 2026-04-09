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

from datasets import Dataset


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

    if train_records:
        train_ds = Dataset.from_list(train_records)
        train_path = os.path.join(output_dir, "train.parquet")
        train_ds.to_parquet(train_path)
        print(f"  {train_path}: {len(train_records)} records")

    if eval_records:
        eval_ds = Dataset.from_list(eval_records)
        eval_path = os.path.join(output_dir, "validation.parquet")
        eval_ds.to_parquet(eval_path)
        print(f"  {eval_path}: {len(eval_records)} records")

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
