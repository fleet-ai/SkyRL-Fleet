#!/usr/bin/env python3
"""Build SkyRL task index JSON from gym-anything's CUA-World environments.

Scans benchmarks/cua_world/environments/ and generates a flat JSON list
of {task_key, env_dir, task_id, description, difficulty, max_steps, ...}
that GymAnythingTaskEnv can load.

Usage:
    python -m skyrl_gym.envs.gym_anything.build_task_index \
        --gym-anything-root /path/to/gym-anything \
        --output tasks_gym_anything.json \
        [--split train]  # optional: filter by split
"""

import argparse
import json
import sys
from pathlib import Path


def _load_split_filter(ga_root: Path, split: str) -> set:
    """Load train/test split from per-environment split files.

    Returns a set of task_keys (env_name/task_id) in the requested split.
    """
    splits_dir = ga_root / "benchmarks" / "cua_world" / "splits"
    split_key = f"{split}_tasks"
    task_keys = set()

    for split_file in sorted(splits_dir.glob("*_split.json")):
        try:
            with open(split_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        # Extract env_name from filename: blender3d_split.json -> blender3d_env
        env_stem = split_file.stem.replace("_split", "")
        # Try both with and without _env suffix
        env_names = [f"{env_stem}_env", env_stem]

        task_ids = data.get(split_key, [])
        for env_name in env_names:
            for tid in task_ids:
                task_keys.add(f"{env_name}/{tid}")

    return task_keys


def _load_verified_tasks(ga_root: Path) -> set:
    """Load verified task list from verified.json."""
    verified_path = ga_root / "benchmarks" / "cua_world" / "splits" / "verified.json"
    if not verified_path.exists():
        return set()
    with open(verified_path) as f:
        data = json.load(f)
    return set(data.get("tasks", []))


def build_index(ga_root: Path, split: str = None, verified_only: bool = True) -> list:
    envs_dir = ga_root / "benchmarks" / "cua_world" / "environments"
    if not envs_dir.exists():
        print(f"Error: {envs_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load filters
    split_tasks = _load_split_filter(ga_root, split) if split else None
    verified_tasks = _load_verified_tasks(ga_root) if verified_only else None

    if split_tasks is not None:
        print(f"Split '{split}': {len(split_tasks)} tasks", file=sys.stderr)
    if verified_tasks is not None:
        print(f"Verified tasks: {len(verified_tasks)}", file=sys.stderr)

    tasks = []
    for env_dir in sorted(envs_dir.iterdir()):
        if not env_dir.is_dir():
            continue

        # Load env.json
        env_json = None
        for name in ("env.json", "env.yaml", "env.yml"):
            p = env_dir / name
            if p.exists():
                if name.endswith(".json"):
                    with open(p) as f:
                        env_json = json.load(f)
                break

        env_id = env_json.get("id", env_dir.name) if env_json else env_dir.name

        # Skip non-Linux environments (Android, Windows need special runners)
        if env_json:
            os_type = env_json.get("os_type", "")
            runner = env_json.get("runner", "")
            tags = env_json.get("tags", [])
            if os_type in ("windows", "android"):
                continue
            if runner in ("avd", "avd_native", "qemu"):
                continue
            if "android" in tags or "windows" in tags:
                continue

        # Scan tasks
        tasks_dir = env_dir / "tasks"
        if not tasks_dir.exists():
            continue

        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            # Load task.json
            task_json = None
            for name in ("task.json", "task.yaml", "task.yml"):
                p = task_dir / name
                if p.exists():
                    if name.endswith(".json"):
                        try:
                            with open(p) as f:
                                task_json = json.load(f)
                        except json.JSONDecodeError:
                            pass  # skip malformed
                    break

            task_id = task_dir.name
            task_key = f"{env_dir.name}/{task_id}"

            if split_tasks is not None and task_key not in split_tasks:
                continue
            if verified_tasks is not None and task_key not in verified_tasks:
                continue

            description = ""
            difficulty = "medium"
            max_steps = 50
            if task_json:
                description = task_json.get("description", "")
                difficulty = task_json.get("difficulty", "medium")
                init = task_json.get("init", {})
                max_steps = init.get("max_steps", 50)

            tasks.append({
                "task_key": task_key,
                "env_dir": str(env_dir),
                "task_id": task_id,
                "env_id": env_id,
                "env_name": env_dir.name,
                "description": description,
                "difficulty": difficulty,
                "max_turns": max_steps,
                "env_class": "gym_anything",
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Build gym-anything task index for SkyRL")
    parser.add_argument("--gym-anything-root", required=True, help="Path to gym-anything repo")
    parser.add_argument("--output", default="tasks_gym_anything.json", help="Output JSON path")
    parser.add_argument("--split", default=None, help="Filter by split (train/test)")
    parser.add_argument("--no-verified-filter", action="store_true", help="Include unverified tasks")
    args = parser.parse_args()

    ga_root = Path(args.gym_anything_root)
    tasks = build_index(ga_root, args.split, verified_only=not args.no_verified_filter)

    with open(args.output, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Generated {len(tasks)} tasks from {len(set(t['env_name'] for t in tasks))} environments → {args.output}")


if __name__ == "__main__":
    main()
