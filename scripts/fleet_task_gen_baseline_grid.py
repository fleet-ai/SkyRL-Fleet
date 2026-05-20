#!/usr/bin/env python3
"""Build and optionally run the Fleet task-generation baseline grid.

Default grid:
  * 4 sampled Fleet task keys per environment.
  * 3 generator models x 3 solver models.
  * k/max-step sweep only for zillow + Opus 4.7 generator + Opus 4.7 solver.

Example dry run:
  python3 scripts/fleet_task_gen_baseline_grid.py --dry-run

Example execution from the repo root:
  uv run python scripts/fleet_task_gen_baseline_grid.py --run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_ENVIRONMENTS = [
    "booking",
    "budget",
    "carlisle",
    "dropbox",
    "fira",
    "forums-homes",
    "hubspot",
    "outlook",
    "reddit",
    "rops-mail",
    "ticketmaster",
    "wallst",
    "zillow",
]
DEFAULT_MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "google/gemini-3.5-flash",
]
DEFAULT_SWEEP_MODEL = "anthropic/claude-opus-4.7"
WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class SelectedTask:
    env: str
    task_index: int
    task_key: str
    candidate_count: int
    source: str
    source_sha256: str


@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    env: str
    task_index: int
    task_key: str
    generator_model: str
    output: str
    command: list[str]


@dataclass(frozen=True)
class SolveJob:
    job_id: str
    generation_job_id: str
    env: str
    task_index: int
    task_key: str
    generator_model: str
    solver_model: str
    eval_k_rollouts: int
    max_eval_steps: int
    generated_file: str
    rollout_dir: str
    output: str
    command: list[str]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    parsed = []
    for item in parse_csv(value):
        parsed.append(int(item))
    return parsed


def default_path(path: Path, fallback: Path | None = None) -> str:
    if path.exists():
        return str(path)
    if fallback is not None:
        return str(fallback)
    return str(path)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_label(path: Path | None) -> str:
    return path.name if path else "embedded-candidate-snapshot"


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a") as handle:
            handle.write(text)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a") as handle:
            handle.write(json.dumps(data, sort_keys=True) + "\n")


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def git_output(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_output_raw(repo_root: Path, args: list[str], *, allow_codes: set[int] | None = None) -> str:
    allow_codes = allow_codes or {0}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    if result.returncode not in allow_codes:
        return result.stdout + result.stderr
    return result.stdout


def git_metadata(repo_root: Path) -> dict[str, Any]:
    branch = git_output(repo_root, ["branch", "--show-current"])
    commit = git_output(repo_root, ["rev-parse", "HEAD"])
    status_short = git_output(repo_root, ["status", "--short"])
    untracked = git_output(repo_root, ["ls-files", "--others", "--exclude-standard"])
    return {
        "branch": branch,
        "commit": commit,
        "commit_short": commit[:8] if commit else "",
        "dirty": bool(status_short),
        "status_short": status_short.splitlines() if status_short else [],
        "untracked_files": untracked.splitlines() if untracked else [],
    }


def write_reproducibility_snapshots(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    status_path = repro_dir / "git_status.txt"
    tracked_diff_path = repro_dir / "tracked_worktree.diff"
    untracked_diff_path = repro_dir / "untracked_files.diff"
    combined_diff_path = repro_dir / "working_tree.diff"

    status_text = git_output_raw(repo_root, ["status", "--short"])
    tracked_diff = git_output_raw(repo_root, ["diff", "--binary"])
    cached_diff = git_output_raw(repo_root, ["diff", "--cached", "--binary"])
    untracked = [line for line in git_output(repo_root, ["ls-files", "--others", "--exclude-standard"]).splitlines() if line]

    untracked_chunks = []
    for rel_path in untracked:
        path = repo_root / rel_path
        if not path.is_file():
            continue
        untracked_chunks.append(
            git_output_raw(
                repo_root,
                ["diff", "--no-index", "--binary", "--", "/dev/null", rel_path],
                allow_codes={0, 1},
            )
        )

    status_path.write_text(status_text)
    tracked_diff_path.write_text(cached_diff + tracked_diff)
    untracked_diff_path.write_text("\n".join(chunk for chunk in untracked_chunks if chunk))
    combined = "\n".join(
        chunk
        for chunk in [
            "# git diff --cached --binary\n",
            cached_diff,
            "# git diff --binary\n",
            tracked_diff,
            "# untracked files as git diff --no-index --binary\n",
            untracked_diff_path.read_text(),
        ]
        if chunk
    )
    combined_diff_path.write_text(combined)

    return {
        "git_status": str(status_path),
        "tracked_worktree_diff": str(tracked_diff_path),
        "untracked_files_diff": str(untracked_diff_path),
        "working_tree_diff": str(combined_diff_path),
        "working_tree_diff_sha256": file_sha256(combined_diff_path),
        "untracked_files": untracked,
    }


def slug(value: str) -> str:
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "value"


def load_environments(args: argparse.Namespace, snapshot: dict[str, list[str]] | None = None) -> list[str]:
    if args.envs:
        envs = parse_csv(args.envs)
    elif args.env_file and Path(args.env_file).exists():
        envs = load_json(Path(args.env_file))
    elif snapshot:
        envs = sorted(snapshot)
    else:
        envs = DEFAULT_ENVIRONMENTS

    if not isinstance(envs, list) or not all(isinstance(env, str) and env for env in envs):
        raise ValueError("Environment list must be a JSON array of non-empty strings.")
    return list(dict.fromkeys(envs))


def normalize_metadata_records(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("tasks"), list):
            records = raw["tasks"]
        else:
            records = []
            for value in raw.values():
                if isinstance(value, list):
                    records.extend(value)
    else:
        records = []
    return [record for record in records if isinstance(record, dict)]


def task_key_from_record(record: dict[str, Any]) -> str:
    value = record.get("key") or record.get("task_key")
    return value if isinstance(value, str) else ""


def env_from_record(record: dict[str, Any]) -> str:
    value = record.get("environment_id") or record.get("env_id") or record.get("env_key")
    return value if isinstance(value, str) else ""


def load_metadata_task_keys(metadata_json: Path | None) -> dict[str, list[str]]:
    if not metadata_json or not metadata_json.exists():
        return {}
    grouped: dict[str, set[str]] = {}
    for record in normalize_metadata_records(load_json(metadata_json)):
        env = env_from_record(record)
        task_key = task_key_from_record(record)
        if env and task_key:
            grouped.setdefault(env, set()).add(task_key)
    return {env: sorted(keys) for env, keys in grouped.items()}


def load_metadata_contexts(metadata_json: Path | None) -> dict[str, dict[str, Any]]:
    if not metadata_json or not metadata_json.exists():
        return {}
    contexts: dict[str, dict[str, Any]] = {}
    for record in normalize_metadata_records(load_json(metadata_json)):
        task_key = task_key_from_record(record)
        if not task_key:
            continue
        contexts[task_key] = {
            "task_key": task_key,
            "env_key": env_from_record(record),
            "env_version": record.get("env_version") or record.get("version") or "",
            "data_key": record.get("data_key") or record.get("data_id") or "",
            "data_version": record.get("data_version") or "",
            "env_variables": record.get("env_variables") or {},
        }
    return contexts


def load_task_key_file(path: Path) -> list[str]:
    data = load_json(path)
    if not isinstance(data, list) or not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError(f"{path} must contain a JSON array of non-empty task-key strings.")
    return sorted(set(item.strip() for item in data))


def load_candidate_snapshot(path: Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("candidates"), dict):
        data = data["candidates"]
    if not isinstance(data, dict):
        raise ValueError("--candidate-snapshot-json must contain an env -> task-key-list object.")

    candidates: dict[str, list[str]] = {}
    for env, keys in data.items():
        if not isinstance(env, str) or not env:
            raise ValueError("--candidate-snapshot-json contains an invalid environment key.")
        if not isinstance(keys, list) or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError(f"--candidate-snapshot-json env={env!r} must contain non-empty task-key strings.")
        candidates[env] = sorted(set(key.strip() for key in keys))
    return candidates


def load_candidate_task_keys(
    envs: list[str],
    candidate_snapshot: dict[str, list[str]],
    task_key_dir: Path | None,
    metadata_json: Path | None,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    metadata_keys = load_metadata_task_keys(metadata_json)
    candidates: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    metadata_hash = file_sha256(metadata_json) if metadata_json and metadata_json.exists() else ""

    for env in envs:
        task_key_file = task_key_dir / f"{env}_task_keys.json" if task_key_dir else None
        if env in candidate_snapshot:
            candidates[env] = candidate_snapshot[env]
            sources[env] = "candidate-snapshot"
            source_hashes[env] = ""
        elif task_key_file and task_key_file.exists():
            candidates[env] = load_task_key_file(task_key_file)
            sources[env] = source_label(task_key_file)
            source_hashes[env] = file_sha256(task_key_file)
        elif env in metadata_keys:
            candidates[env] = metadata_keys[env]
            sources[env] = source_label(metadata_json)
            source_hashes[env] = metadata_hash
        else:
            expected_task_key_file = str(task_key_file) if task_key_file else "<unset task-key dir>"
            raise FileNotFoundError(
                f"No task keys for env={env!r}. Expected {expected_task_key_file}, "
                "a --candidate-snapshot-json entry, or a --metadata-json containing records for that environment. "
                "Set FLEET_TASK_KEY_DIR/FLEET_TASK_METADATA_JSON or pass the flags explicitly."
            )

    return candidates, sources, source_hashes


def sample_tasks(
    envs: list[str],
    candidates: dict[str, list[str]],
    sources: dict[str, str],
    source_hashes: dict[str, str],
    tasks_per_env: int,
    seed: int,
    allow_fewer: bool,
) -> list[SelectedTask]:
    rng = random.Random(seed)
    selected: list[SelectedTask] = []
    for env in envs:
        keys = candidates[env]
        if len(keys) < tasks_per_env and not allow_fewer:
            raise ValueError(
                f"env={env!r} has only {len(keys)} candidate task keys; "
                f"requested {tasks_per_env}. Pass --allow-fewer-tasks to continue."
            )
        count = min(tasks_per_env, len(keys))
        sampled = rng.sample(keys, count)
        for index, task_key in enumerate(sampled, start=1):
            selected.append(
                SelectedTask(
                    env=env,
                    task_index=index,
                    task_key=task_key,
                    candidate_count=len(keys),
                    source=sources[env],
                    source_sha256=source_hashes.get(env, ""),
                )
            )
    return selected


def solve_grid_for(
    args: argparse.Namespace,
    env: str,
    generator_model: str,
    solver_model: str,
) -> list[tuple[int, int]]:
    base = [(args.base_eval_k_rollouts, args.base_max_eval_steps)]
    sweep = list(itertools.product(args.sweep_eval_k_rollouts, args.sweep_max_eval_steps))

    if args.no_sweep:
        combos = base
    elif args.sweep_all:
        combos = sweep
    elif (
        env == args.sweep_env
        and generator_model == args.sweep_generator
        and solver_model == args.sweep_solver
    ):
        combos = sweep
    else:
        combos = base

    return list(dict.fromkeys(combos))


def build_generation_command(args: argparse.Namespace, task: SelectedTask, model: str, output: Path) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "integrations.fleet.task_gen_baseline",
        "generate",
        "--fleet-task-key",
        task.task_key,
        "--model",
        model,
        "--max-turns",
        str(args.generator_max_turns),
        "--max-tokens",
        str(args.generator_max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--tool-mode",
        args.tool_mode,
        "--retry",
        str(args.retry),
        "-o",
        str(output),
    ]
    if args.include_transcript or args.capture_all_artifacts:
        command.append("--include-transcript")
    return command


def build_solve_command(
    args: argparse.Namespace,
    generated_file: Path,
    solver_model: str,
    eval_k_rollouts: int,
    max_eval_steps: int,
    rollout_dir: Path,
    output: Path,
) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "integrations.fleet.task_gen_baseline",
        "solve",
        "--file",
        str(generated_file),
        "--evaluator-model",
        solver_model,
        "--training-phase",
        "eval",
        "--k-rollouts",
        str(eval_k_rollouts),
        "--eval-k-rollouts",
        str(eval_k_rollouts),
        "--max-eval-steps",
        str(max_eval_steps),
        "--max-turns",
        str(args.solver_max_turns),
        "--rollout-dir",
        str(rollout_dir),
        "--judge-model",
        args.judge_model,
        "--base-quality-reward",
        str(args.base_quality_reward),
        "-o",
        str(output),
    ]
    if args.enable_hints:
        command.append("--enable-hints")
    return command


def build_jobs(
    args: argparse.Namespace,
    output_dir: Path,
    selected_tasks: list[SelectedTask],
) -> tuple[list[GenerationJob], list[SolveJob]]:
    generation_jobs: list[GenerationJob] = []
    solve_jobs: list[SolveJob] = []

    for task in selected_tasks:
        task_slug = f"{task.task_index:02d}_{slug(task.task_key)[:80]}"
        for generator_model in args.generators:
            generator_slug = slug(generator_model)
            generation_id = f"gen__{task.env}__{task.task_index:02d}__{generator_slug}"
            generated_file = output_dir / "generated" / task.env / generator_slug / f"{task_slug}.json"
            generation_jobs.append(
                GenerationJob(
                    job_id=generation_id,
                    env=task.env,
                    task_index=task.task_index,
                    task_key=task.task_key,
                    generator_model=generator_model,
                    output=str(generated_file),
                    command=build_generation_command(args, task, generator_model, generated_file),
                )
            )

            for solver_model in args.solvers:
                for eval_k_rollouts, max_eval_steps in solve_grid_for(args, task.env, generator_model, solver_model):
                    solver_slug = slug(solver_model)
                    solve_id = (
                        f"solve__{task.env}__{task.task_index:02d}__{generator_slug}"
                        f"__{solver_slug}__k{eval_k_rollouts}__steps{max_eval_steps}"
                    )
                    solve_file = (
                        output_dir
                        / "solves"
                        / task.env
                        / generator_slug
                        / solver_slug
                        / f"{task_slug}__k{eval_k_rollouts}__steps{max_eval_steps}.json"
                    )
                    rollout_dir = (
                        output_dir
                        / "rollouts"
                        / task.env
                        / generator_slug
                        / solver_slug
                        / f"{task_slug}__k{eval_k_rollouts}__steps{max_eval_steps}"
                    )
                    solve_jobs.append(
                        SolveJob(
                            job_id=solve_id,
                            generation_job_id=generation_id,
                            env=task.env,
                            task_index=task.task_index,
                            task_key=task.task_key,
                            generator_model=generator_model,
                            solver_model=solver_model,
                            eval_k_rollouts=eval_k_rollouts,
                            max_eval_steps=max_eval_steps,
                            generated_file=str(generated_file),
                            rollout_dir=str(rollout_dir),
                            output=str(solve_file),
                            command=build_solve_command(
                                args,
                                generated_file,
                                solver_model,
                                eval_k_rollouts,
                                max_eval_steps,
                                rollout_dir,
                                solve_file,
                            ),
                        )
                    )

    if args.limit_generation_jobs is not None:
        generation_jobs = generation_jobs[: args.limit_generation_jobs]
        kept = {job.job_id for job in generation_jobs}
        solve_jobs = [job for job in solve_jobs if job.generation_job_id in kept]
    if args.limit_solve_jobs is not None:
        solve_jobs = solve_jobs[: args.limit_solve_jobs]

    return generation_jobs, solve_jobs


def shell_prefix_for_env_file(path: Path) -> str:
    return f"set -a; source {shlex.quote(str(path))}; set +a"


def shell_arg(value: str, output_dir: Path, repo_root: Path, python_executable: str) -> str:
    if value == python_executable:
        return '"${PYTHON}"'

    path = Path(value)
    if not path.is_absolute():
        return shlex.quote(value)

    try:
        rel = path.resolve().relative_to(output_dir.resolve())
        return '"${RUN_DIR}/' + str(rel).replace('"', '\\"') + '"'
    except (ValueError, OSError):
        pass
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return '"${REPO_ROOT}/' + str(rel).replace('"', '\\"') + '"'
    except (ValueError, OSError):
        pass
    return shlex.quote(value)


def render_shell_command(command: list[str], output_dir: Path, repo_root: Path, python_executable: str) -> str:
    return " ".join(shell_arg(value, output_dir, repo_root, python_executable) for value in command)


def jobs_jsonl_text(generation_jobs: list[GenerationJob], solve_jobs: list[SolveJob]) -> str:
    lines = []
    for job in generation_jobs:
        lines.append(json.dumps({"kind": "generate", **asdict(job)}, sort_keys=True))
    for job in solve_jobs:
        lines.append(json.dumps({"kind": "solve", **asdict(job)}, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def commands_sh_text(
    output_dir: Path,
    repo_root: Path,
    secrets_file: Path,
    python_executable: str,
    generation_jobs: list[GenerationJob],
    solve_jobs: list[SolveJob],
) -> str:
    default_secrets = "$HOME/.secrets/api_keys.env"
    if secrets_file != Path.home() / ".secrets" / "api_keys.env":
        default_secrets = shlex.quote(str(secrets_file))
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        f'REPO_ROOT="${{REPO_ROOT:-{shlex.quote(str(repo_root))}}}"',
        f'PYTHON="${{PYTHON:-{shlex.quote(str(python_executable))}}}"',
        f'SECRETS_FILE="${{SECRETS_FILE:-{default_secrets}}}"',
        'if [ -f "$SECRETS_FILE" ]; then set -a; source "$SECRETS_FILE"; set +a; fi',
        'export PYTHONPATH="$REPO_ROOT/skyrl-gym:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        'export FLEET_CLIENT_TIMEOUT="${FLEET_CLIENT_TIMEOUT:-60}"',
        'export FLEET_JOB_POLL_INTERVAL="${FLEET_JOB_POLL_INTERVAL:-10}"',
        'export FLEET_JOB_POLL_TIMEOUT="${FLEET_JOB_POLL_TIMEOUT:-900}"',
        'cd "$REPO_ROOT"',
        'FAILURE_LOG="$RUN_DIR/job_failures.tsv"',
        ': > "$FAILURE_LOG"',
        "run_job() {",
        '  local job_id="$1"',
        "  shift",
        '  echo "+ [$job_id] $*"',
        '  "$@"',
        "  local status=$?",
        '  if [ "$status" -ne 0 ]; then',
        '    printf "%s\\t%s\\n" "$job_id" "$status" >> "$FAILURE_LOG"',
        '    if [ "${FAIL_FAST:-0}" = "1" ]; then exit "$status"; fi',
        "  fi",
        "}",
        "",
    ]
    for job in generation_jobs:
        lines.append(f"mkdir -p {shell_arg(str(Path(job.output).parent), output_dir, repo_root, python_executable)}")
        lines.append(
            f"run_job {shlex.quote(job.job_id)} "
            f"{render_shell_command(job.command, output_dir, repo_root, python_executable)}"
        )
    for job in solve_jobs:
        lines.append(f"mkdir -p {shell_arg(str(Path(job.output).parent), output_dir, repo_root, python_executable)}")
        lines.append(
            f"run_job {shlex.quote(job.job_id)} "
            f"{render_shell_command(job.command, output_dir, repo_root, python_executable)}"
        )
    lines.extend(
        [
            "",
            'if [ -s "$FAILURE_LOG" ] && [ "${ALLOW_FAILURES:-0}" != "1" ]; then',
            '  echo "Some jobs failed; see $FAILURE_LOG" >&2',
            "  exit 1",
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def write_jobs_files(
    output_dir: Path,
    repo_root: Path,
    secrets_file: Path,
    python_executable: str,
    generation_jobs: list[GenerationJob],
    solve_jobs: list[SolveJob],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_jsonl = output_dir / "jobs.jsonl"
    jobs_jsonl.write_text(jobs_jsonl_text(generation_jobs, solve_jobs))

    commands = output_dir / "commands.sh"
    commands.write_text(commands_sh_text(output_dir, repo_root, secrets_file, python_executable, generation_jobs, solve_jobs))
    commands.chmod(0o755)


def command_needs_run(output: str, force: bool) -> bool:
    return force or not Path(output).exists()


def json_file(path: Path) -> dict[str, Any]:
    try:
        data = load_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except Exception:
        return {}
    if not lines:
        return {}
    try:
        data = json.loads(lines[-1])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def file_has_text(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def generation_output_complete(path: Path) -> bool:
    data = json_file(path)
    return (
        data.get("done_reason") == "task_generated"
        and isinstance(data.get("prompt"), str)
        and bool(data.get("prompt"))
        and isinstance(data.get("verifier"), str)
        and bool(data.get("verifier"))
        and data.get("mode") not in {"generate_failure", "solve_failure"}
        and not data.get("error")
    )


def rollout_status_has_terminal_success(status_path: Path, raw_job_id: str) -> bool:
    if not file_has_text(status_path):
        return False
    try:
        lines = status_path.read_text().splitlines()
    except Exception:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("job_id") != raw_job_id:
            continue
        if event.get("event") == "terminal" and event.get("status") == "completed":
            return True
    return False


def has_usable_session_trace(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    if not isinstance(value, dict):
        return bool(value)

    trace_keys = (
        "trajectory",
        "trajectories",
        "rollout",
        "rollouts",
        "transcript",
        "messages",
        "steps",
        "actions",
        "events",
    )
    container_keys = ("session_transcript", "transcript_payload", "payload", "data", "result")
    return any(has_usable_session_trace(value.get(key)) for key in (*trace_keys, *container_keys))


def session_has_usable_trace(session: dict[str, Any]) -> bool:
    return any(
        has_usable_session_trace(session.get(key))
        for key in (
            "trajectory",
            "messages",
            "steps",
            "actions",
            "events",
            "transcript",
            "transcript_payload",
            "session_transcript",
        )
    )


def solve_rollout_artifacts_complete(data: dict[str, Any], job: SolveJob) -> bool:
    if job.eval_k_rollouts <= 0:
        return False

    raw_job_id = data.get("raw_job_id")
    if not isinstance(raw_job_id, str) or not raw_job_id:
        return False

    rollout_dir = Path(job.rollout_dir)
    rollout_file_value = data.get("rollout_file")
    if not isinstance(rollout_file_value, str) or not rollout_file_value:
        return False
    rollout_file = Path(rollout_file_value)
    if not rollout_file.is_absolute():
        rollout_file = rollout_dir / rollout_file
    if not file_has_text(rollout_file):
        return False

    rollout_record = read_last_jsonl(rollout_file)
    if rollout_record.get("raw_job_id") != raw_job_id:
        return False
    raw_scores = rollout_record.get("raw_scores", [])
    if not isinstance(raw_scores, list) or len(raw_scores) != job.eval_k_rollouts:
        return False

    raw_sessions = rollout_record.get("raw_sessions", [])
    if not isinstance(raw_sessions, list) or len(raw_sessions) != job.eval_k_rollouts:
        return False
    for session in raw_sessions:
        if not isinstance(session, dict) or not session.get("session_id"):
            return False
        if not session_has_usable_trace(session):
            return False

    status_path = rollout_dir / "fleet_job_status.jsonl"
    snapshot_path = rollout_dir / "fleet_jobs" / f"{raw_job_id}.json"
    return file_has_text(snapshot_path) and rollout_status_has_terminal_success(status_path, raw_job_id)


def solve_reached_solver(data: dict[str, Any]) -> bool:
    metadata = data.get("metadata", {})
    breakdown = metadata.get("reward_breakdown", {}) if isinstance(metadata, dict) else {}
    return (
        isinstance(breakdown, dict)
        and breakdown.get("sandbox") == 1.0
        and breakdown.get("dryrun") == 1.0
        and breakdown.get("judge") != 0.0
    )


def solve_output_complete(path: Path, job: SolveJob) -> bool:
    data = json_file(path)
    expected_rollouts = job.eval_k_rollouts
    if data.get("mode") != "gates_and_solve" or data.get("error"):
        return False
    if not solve_reached_solver(data):
        metadata = data.get("metadata", {})
        return isinstance(data.get("task_gen_reward"), (int, float)) and isinstance(metadata, dict)
    return (
        data.get("raw_job_id") is not None
        and data.get("solver_rollouts") == expected_rollouts
        and isinstance(data.get("solver_pass_rate"), (int, float))
        and solve_rollout_artifacts_complete(data, job)
    )


def job_output_complete(kind: str, job: GenerationJob | SolveJob) -> bool:
    path = Path(job.output)
    if not path.exists():
        return False
    if kind == "generate":
        return generation_output_complete(path)
    return solve_output_complete(path, job)  # type: ignore[arg-type]


def archive_incomplete_output(output_dir: Path, kind: str, job_id: str, output: str) -> str:
    path = Path(output)
    if not path.exists():
        return ""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir / "failed_outputs" / kind / f"{slug(job_id)}__{stamp}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, archive_path)
    log_job_event(
        output_dir,
        {
            "event": "archive_incomplete_output",
            "kind": kind,
            "job_id": job_id,
            "output": output,
            "archive": str(archive_path),
        },
    )
    return str(archive_path)


def job_needs_run(
    output_dir: Path,
    kind: str,
    job: GenerationJob | SolveJob,
    force: bool,
) -> bool:
    path = Path(job.output)
    if not path.exists():
        return True
    if job_output_complete(kind, job):
        if force:
            archive_incomplete_output(output_dir, kind, job.job_id, job.output)
            return True
        return False
    archive_incomplete_output(output_dir, kind, job.job_id, job.output)
    return True


def progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def log_job_event(output_dir: Path, event: dict[str, Any]) -> None:
    append_jsonl(output_dir / "job_events.jsonl", {"created_at": utc_now(), **event})


def print_and_log(output_dir: Path, message: str) -> None:
    with WRITE_LOCK:
        print(message, flush=True)
        append_text(output_dir / "run.log", message + "\n")


def job_log_path(output_dir: Path, kind: str, job_id: str) -> Path:
    return output_dir / "job_logs" / kind / f"{slug(job_id)}.log"


def write_job_status(output_dir: Path, kind: str, job_id: str, status: dict[str, Any]) -> None:
    write_json(output_dir / "job_status" / kind / f"{slug(job_id)}.json", status)


def read_job_status(output_dir: Path, kind: str, job_id: str) -> dict[str, Any]:
    return json_file(output_dir / "job_status" / kind / f"{slug(job_id)}.json")


ARCHIVED_GENERATION_LOCK = threading.RLock()
ARCHIVED_GENERATIONS: dict[str, str] = {}


def solve_failure_payload(job: SolveJob, status: str, message: str) -> dict[str, Any]:
    return {
        "mode": "solve_failure",
        "status": status,
        "error": message,
        "job_id": job.job_id,
        "generation_job_id": job.generation_job_id,
        "env_key": job.env,
        "fleet_task_key": job.task_key,
        "generator_model": job.generator_model,
        "evaluator_model": job.solver_model,
        "eval_k_rollouts": job.eval_k_rollouts,
        "max_eval_steps": job.max_eval_steps,
        "generated_file": job.generated_file,
        "rollout_dir": job.rollout_dir,
        "output": job.output,
    }


def generation_archive_for_solve(output_dir: Path, job: SolveJob) -> str:
    with ARCHIVED_GENERATION_LOCK:
        existing = ARCHIVED_GENERATIONS.get(job.generation_job_id)
        if existing:
            return existing
        generated_path = Path(job.generated_file)
        if generated_path.exists():
            archive = archive_incomplete_output(output_dir, "generate", job.generation_job_id, job.generated_file)
        else:
            archived = sorted(
                (output_dir / "failed_outputs" / "generate").glob(f"{slug(job.generation_job_id)}__*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            archive = str(archived[0]) if archived else ""
        ARCHIVED_GENERATIONS[job.generation_job_id] = archive
        return archive


class PhaseTracker:
    def __init__(self, output_dir: Path, kind: str, total: int):
        self.output_dir = output_dir
        self.kind = kind
        self.total = total
        self.started = 0
        self.completed = 0
        self.lock = threading.Lock()

    def start(self, job_id: str) -> int:
        with self.lock:
            self.started += 1
            started = self.started
            completed = self.completed
        print_and_log(
            self.output_dir,
            f"{progress_bar(completed, self.total)} {self.kind} START {started}/{self.total} {job_id}",
        )
        return started

    def finish(self, job_id: str, status: str, duration: float) -> int:
        with self.lock:
            self.completed += 1
            completed = self.completed
        print_and_log(
            self.output_dir,
            f"{progress_bar(completed, self.total)} {self.kind} {status} {completed}/{self.total} "
            f"{job_id} ({duration:.1f}s)",
        )
        return completed

    def note(self, job_id: str, status: str, detail: str = "") -> int:
        with self.lock:
            self.completed += 1
            completed = self.completed
        suffix = f": {detail}" if detail else ""
        print_and_log(
            self.output_dir,
            f"{progress_bar(completed, self.total)} {self.kind} {status} {completed}/{self.total} "
            f"{job_id}{suffix}",
        )
        return completed


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    kind: str,
    index: int,
    total: int,
    job_id: str,
    tracker: PhaseTracker | None = None,
) -> None:
    started = time.time()
    command_text = shlex.join(command)
    per_job_log = job_log_path(output_dir, kind, job_id)
    if tracker:
        tracker.start(job_id)
    else:
        print_and_log(output_dir, f"{progress_bar(index - 1, total)} {kind} START {index}/{total} {job_id}")
    append_text(output_dir / "run.log", "+ " + command_text + "\n")
    append_text(per_job_log, f"=== {kind} {job_id} started {utc_now()} ===\n+ {command_text}\n")
    log_job_event(
        output_dir,
        {
            "event": "start",
            "kind": kind,
            "index": index,
            "total": total,
            "job_id": job_id,
            "command": command,
            "job_log": str(per_job_log),
        },
    )

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    with (output_dir / "run.log").open("a") as run_log, per_job_log.open("a") as job_log:
        for line in process.stdout:
            prefixed = f"[{kind}:{job_id}] {line}"
            with WRITE_LOCK:
                sys.stdout.write(prefixed)
                sys.stdout.flush()
                run_log.write(prefixed)
                run_log.flush()
                job_log.write(line)
                job_log.flush()
            fleet_match = re.search(r"Harness job created: ([0-9a-fA-F-]+) for task (\S+)", line)
            if fleet_match:
                append_jsonl(
                    output_dir / "fleet_jobs_seen.jsonl",
                    {
                        "created_at": utc_now(),
                        "kind": kind,
                        "job_id": job_id,
                        "fleet_job_id": fleet_match.group(1),
                        "fleet_task_key": fleet_match.group(2),
                        "job_log": str(per_job_log),
                    },
                )

    returncode = process.wait()
    duration = time.time() - started
    event = {
        "event": "end" if returncode == 0 else "failed",
        "kind": kind,
        "index": index,
        "total": total,
        "job_id": job_id,
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
    }
    log_job_event(output_dir, event)
    status = "DONE" if returncode == 0 else f"FAILED rc={returncode}"
    write_job_status(
        output_dir,
        kind,
        job_id,
        {
            "job_id": job_id,
            "kind": kind,
            "status": "done" if returncode == 0 else "failed",
            "returncode": returncode,
            "duration_seconds": round(duration, 3),
            "command": command,
            "job_log": str(per_job_log),
        },
    )
    append_text(per_job_log, f"=== {kind} {job_id} {status} {utc_now()} duration={duration:.1f}s ===\n")
    if tracker:
        tracker.finish(job_id, status, duration)
    else:
        print_and_log(
            output_dir,
            f"{progress_bar(index, total)} {kind} {status} {index}/{total} {job_id} ({duration:.1f}s)",
        )
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def record_failure(output_dir: Path, kind: str, job_id: str, error: str) -> None:
    append_jsonl(
        output_dir / "job_failures.jsonl",
        {
            "created_at": utc_now(),
            "kind": kind,
            "job_id": job_id,
            "error": error,
        },
    )


def record_skip(
    output_dir: Path,
    tracker: PhaseTracker,
    kind: str,
    index: int,
    total: int,
    job_id: str,
    output: str,
    reason: str,
) -> None:
    tracker.note(job_id, "SKIP", reason)
    log_job_event(
        output_dir,
        {
            "event": "skip",
            "kind": kind,
            "index": index,
            "total": total,
            "job_id": job_id,
            "output": output,
            "reason": reason,
        },
    )
    write_job_status(
        output_dir,
        kind,
        job_id,
        {
            "job_id": job_id,
            "kind": kind,
            "status": "skipped",
            "output": output,
            "reason": reason,
        },
    )


def record_missing_input(
    output_dir: Path,
    tracker: PhaseTracker,
    job: SolveJob,
    index: int,
    total: int,
    message: str,
) -> None:
    archive = generation_archive_for_solve(output_dir, job)
    tracker.note(job.job_id, "MISSING_INPUT", message)
    payload = solve_failure_payload(job, "missing_input", message)
    payload["generation_archive"] = archive
    write_json(Path(job.output), payload)
    log_job_event(
        output_dir,
        {
            "event": "missing_input",
            "kind": "solve",
            "index": index,
            "total": total,
            "job_id": job.job_id,
            "generated_file": job.generated_file,
            "generation_archive": archive,
        },
    )
    record_failure(output_dir, "solve", job.job_id, message)
    write_job_status(
        output_dir,
        "solve",
        job.job_id,
        {
            "job_id": job.job_id,
            "kind": "solve",
            "status": "missing_input",
            "generated_file": job.generated_file,
            "output": job.output,
            "generation_archive": archive,
        },
    )


def record_invalid_generation_input(
    output_dir: Path,
    tracker: PhaseTracker,
    job: SolveJob,
    index: int,
    total: int,
    message: str,
) -> None:
    archive = generation_archive_for_solve(output_dir, job)
    tracker.note(job.job_id, "INVALID_GENERATION", message)
    payload = solve_failure_payload(job, "invalid_generation_input", message)
    payload["generation_archive"] = archive
    write_json(Path(job.output), payload)
    log_job_event(
        output_dir,
        {
            "event": "invalid_generation_input",
            "kind": "solve",
            "index": index,
            "total": total,
            "job_id": job.job_id,
            "generation_job_id": job.generation_job_id,
            "generated_file": job.generated_file,
            "generation_archive": archive,
        },
    )
    record_failure(output_dir, "solve", job.job_id, message)
    write_job_status(
        output_dir,
        "solve",
        job.job_id,
        {
            "job_id": job.job_id,
            "kind": "solve",
            "status": "invalid_generation_input",
            "generated_file": job.generated_file,
            "output": job.output,
            "generation_archive": archive,
        },
    )


def collect_phase_futures(
    futures: list[concurrent.futures.Future[None]],
    output_dir: Path,
    kind: str,
    fail_fast: bool,
) -> int:
    failures = 0
    for future in concurrent.futures.as_completed(futures):
        job_id = getattr(future, "job_id", "<unknown>")
        try:
            future.result()
        except subprocess.CalledProcessError as exc:
            failures += 1
            record_failure(output_dir, kind, str(job_id), str(exc))
            if fail_fast:
                raise
        except Exception as exc:
            failures += 1
            record_failure(output_dir, kind, str(job_id), repr(exc))
            if fail_fast:
                raise
    return failures


def run_jobs(
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    generation_jobs: list[GenerationJob],
    solve_jobs: list[SolveJob],
) -> None:
    env = os.environ.copy()
    env.update(load_env_file(Path(args.secrets_file).expanduser()))
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(repo_root / "skyrl-gym"), str(repo_root)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("FLEET_CLIENT_TIMEOUT", "60")
    env.setdefault("FLEET_JOB_POLL_INTERVAL", "10")
    env.setdefault("FLEET_JOB_POLL_TIMEOUT", "900")
    failures = 0
    run_started = time.time()
    append_text(output_dir / "run.log", f"\n=== run started {utc_now()} phase={args.phase} ===\n")
    log_job_event(
        output_dir,
        {
            "event": "run_start",
            "phase": args.phase,
            "generation_jobs": len(generation_jobs),
            "solve_jobs": len(solve_jobs),
            "generate_concurrency": args.generate_concurrency,
            "solve_concurrency": args.solve_concurrency,
        },
    )

    if args.phase in {"all", "generate"}:
        total = len(generation_jobs)
        tracker = PhaseTracker(output_dir, "generate", total)
        futures: list[concurrent.futures.Future[None]] = []
        max_workers = 1 if args.fail_fast else max(1, args.generate_concurrency)
        print_and_log(output_dir, f"=== generate phase concurrency={max_workers} jobs={total} ===")
        if args.fail_fast:
            for index, job in enumerate(generation_jobs, start=1):
                Path(job.output).parent.mkdir(parents=True, exist_ok=True)
                if job_needs_run(output_dir, "generate", job, args.force):
                    try:
                        run_command(
                            job.command,
                            repo_root,
                            env,
                            output_dir,
                            "generate",
                            index,
                            total,
                            job.job_id,
                            tracker,
                        )
                    except subprocess.CalledProcessError as exc:
                        failures += 1
                        record_failure(output_dir, "generate", job.job_id, str(exc))
                        break
                    except Exception as exc:
                        failures += 1
                        record_failure(output_dir, "generate", job.job_id, repr(exc))
                        break
                else:
                    record_skip(
                        output_dir,
                        tracker,
                        "generate",
                        index,
                        total,
                        job.job_id,
                        job.output,
                        "existing successful output",
                    )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for index, job in enumerate(generation_jobs, start=1):
                    Path(job.output).parent.mkdir(parents=True, exist_ok=True)
                    if job_needs_run(output_dir, "generate", job, args.force):
                        future = executor.submit(
                            run_command,
                            job.command,
                            repo_root,
                            env,
                            output_dir,
                            "generate",
                            index,
                            total,
                            job.job_id,
                            tracker,
                        )
                        setattr(future, "job_id", job.job_id)
                        futures.append(future)
                    else:
                        record_skip(
                            output_dir,
                            tracker,
                            "generate",
                            index,
                            total,
                            job.job_id,
                            job.output,
                            "existing successful output",
                        )
                failures += collect_phase_futures(futures, output_dir, "generate", args.fail_fast)

    if args.phase in {"all", "solve"} and not (args.fail_fast and failures):
        total = len(solve_jobs)
        tracker = PhaseTracker(output_dir, "solve", total)
        futures = []
        max_workers = 1 if args.fail_fast else max(1, args.solve_concurrency)
        print_and_log(output_dir, f"=== solve phase concurrency={max_workers} jobs={total} ===")
        if args.fail_fast:
            for index, job in enumerate(solve_jobs, start=1):
                if not Path(job.generated_file).exists():
                    failures += 1
                    message = f"Missing generated task file for solve: {job.generated_file}"
                    record_missing_input(output_dir, tracker, job, index, total, message)
                    if args.fail_fast:
                        break
                    continue
                if not generation_output_complete(Path(job.generated_file)):
                    failures += 1
                    message = f"Generated task file is not a successful task: {job.generated_file}"
                    record_invalid_generation_input(output_dir, tracker, job, index, total, message)
                    if args.fail_fast:
                        break
                    continue
                Path(job.output).parent.mkdir(parents=True, exist_ok=True)
                if job_needs_run(output_dir, "solve", job, args.force):
                    try:
                        run_command(
                            job.command,
                            repo_root,
                            env,
                            output_dir,
                            "solve",
                            index,
                            total,
                            job.job_id,
                            tracker,
                        )
                    except subprocess.CalledProcessError as exc:
                        failures += 1
                        record_failure(output_dir, "solve", job.job_id, str(exc))
                        break
                    except Exception as exc:
                        failures += 1
                        record_failure(output_dir, "solve", job.job_id, repr(exc))
                        break
                else:
                    record_skip(
                        output_dir,
                        tracker,
                        "solve",
                        index,
                        total,
                        job.job_id,
                        job.output,
                        "existing successful output",
                    )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for index, job in enumerate(solve_jobs, start=1):
                    if not Path(job.generated_file).exists():
                        failures += 1
                        message = f"Missing generated task file for solve: {job.generated_file}"
                        record_missing_input(output_dir, tracker, job, index, total, message)
                        continue
                    if not generation_output_complete(Path(job.generated_file)):
                        failures += 1
                        message = f"Generated task file is not a successful task: {job.generated_file}"
                        record_invalid_generation_input(output_dir, tracker, job, index, total, message)
                        continue
                    Path(job.output).parent.mkdir(parents=True, exist_ok=True)
                    if job_needs_run(output_dir, "solve", job, args.force):
                        future = executor.submit(
                            run_command,
                            job.command,
                            repo_root,
                            env,
                            output_dir,
                            "solve",
                            index,
                            total,
                            job.job_id,
                            tracker,
                        )
                        setattr(future, "job_id", job.job_id)
                        futures.append(future)
                    else:
                        record_skip(
                            output_dir,
                            tracker,
                            "solve",
                            index,
                            total,
                            job.job_id,
                            job.output,
                            "existing successful output",
                        )
                failures += collect_phase_futures(futures, output_dir, "solve", args.fail_fast)

    duration = time.time() - run_started
    log_job_event(
        output_dir,
        {
            "event": "run_end",
            "phase": args.phase,
            "failures": failures,
            "duration_seconds": round(duration, 3),
        },
    )
    print_and_log(output_dir, f"=== run finished {utc_now()} failures={failures} duration={duration:.1f}s ===")
    if failures:
        print(f"{failures} job(s) failed; see {output_dir / 'job_failures.jsonl'}", file=sys.stderr, flush=True)
        if not args.allow_failures:
            raise SystemExit(1)


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return manifest if isinstance(manifest, dict) else {}


def load_manifest_jobs(output_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = load_manifest(output_dir)
    jobs = manifest.get("solve_jobs", []) if isinstance(manifest, dict) else []
    return {
        str(job.get("output")): job
        for job in jobs
        if isinstance(job, dict) and isinstance(job.get("output"), str)
    }


def load_manifest_solve_jobs(output_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(output_dir)
    jobs = manifest.get("solve_jobs", []) if isinstance(manifest, dict) else []
    return [job for job in jobs if isinstance(job, dict) and isinstance(job.get("output"), str)]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "git_branch",
        "git_commit",
        "job_id",
        "job_status",
        "mode",
        "successful_solve",
        "env_key",
        "fleet_task_key",
        "generator_model",
        "solver_model",
        "eval_k_rollouts",
        "max_eval_steps",
        "solver_pass_rate",
        "solver_pass_at_k",
        "solver_pass_count",
        "solver_rollouts",
        "task_gen_reward",
        "raw_job_id",
        "rollout_file",
        "generation_archive",
        "error",
        "path",
    ]
    flattened = [flatten_for_csv(row) for row in rows]
    keys = []
    seen = set()
    for key in preferred:
        if any(key in row for row in flattened):
            keys.append(key)
            seen.add(key)
    for row in flattened:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flattened)


def solve_row_from_manifest_job(output_dir: Path, git: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job.get("output", "")))
    status = read_job_status(output_dir, "solve", str(job.get("job_id", "")))
    row: dict[str, Any] = {
        "git_branch": git.get("branch", ""),
        "git_commit": git.get("commit", ""),
        "path": str(path),
        "job_id": job.get("job_id", ""),
        "job_status": status.get("status", "missing_output"),
        "env_key": job.get("env", ""),
        "fleet_task_key": job.get("task_key", ""),
        "generator_model": job.get("generator_model", ""),
        "solver_model": job.get("solver_model", ""),
        "eval_k_rollouts": job.get("eval_k_rollouts", ""),
        "max_eval_steps": job.get("max_eval_steps", ""),
        "generated_file": job.get("generated_file", ""),
        "rollout_dir": job.get("rollout_dir", ""),
        "error": status.get("error", ""),
    }
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = load_json(path)
            if not isinstance(data, dict):
                data = {"error": f"Expected object JSON, got {type(data).__name__}"}
        except Exception as exc:
            data = {"error": str(exc), "mode": "unreadable_output"}
    else:
        data = {"mode": "missing_output", "error": row["error"] or f"Missing solve output: {path}"}

    mode = data.get("mode", "")
    error = data.get("error", "") or row.get("error", "")
    audit_error = ""
    successful = mode == "gates_and_solve" and not error
    reached_solver = solve_reached_solver(data)
    if successful and reached_solver:
        try:
            audit_job = SolveJob(
                job_id=str(job.get("job_id", "")),
                generation_job_id=str(job.get("generation_job_id", "")),
                env=str(job.get("env", "")),
                task_index=int(job.get("task_index", 0) or 0),
                task_key=str(job.get("task_key", "")),
                generator_model=str(job.get("generator_model", "")),
                solver_model=str(job.get("solver_model", "")),
                eval_k_rollouts=int(job.get("eval_k_rollouts", 0) or 0),
                max_eval_steps=int(job.get("max_eval_steps", 0) or 0),
                generated_file=str(job.get("generated_file", "")),
                rollout_dir=str(job.get("rollout_dir", "")),
                output=str(job.get("output", "")),
                command=list(job.get("command", [])) if isinstance(job.get("command"), list) else [],
            )
            if not solve_rollout_artifacts_complete(data, audit_job):
                audit_error = "Missing or incomplete solve audit artifacts."
                successful = False
        except Exception as exc:
            audit_error = f"Could not verify solve audit artifacts: {exc}"
            successful = False
    if audit_error and not error:
        error = audit_error
    row.update(
        {
            "mode": mode,
            "job_status": "done" if successful and row["job_status"] == "missing_output" else row["job_status"],
            "successful_solve": successful,
            "audit_complete": successful and (not reached_solver or not audit_error),
            "error": error,
            "exception_type": data.get("exception_type", ""),
            "env_key": data.get("env_key", "") or row["env_key"],
            "fleet_task_key": data.get("fleet_task_key", "") or row["fleet_task_key"],
            "solver_model": data.get("evaluator_model", "") or row["solver_model"],
            "eval_k_rollouts": data.get("eval_k_rollouts", "") or row["eval_k_rollouts"],
            "max_eval_steps": data.get("max_eval_steps", "") or row["max_eval_steps"],
            "task_gen_reward": data.get("task_gen_reward", 0.0 if not successful else ""),
            "solver_pass_rate": data.get("solver_pass_rate", 0.0 if not successful else ""),
            "solver_pass_at_k": data.get("solver_pass_at_k", False if not successful else ""),
            "solver_pass_count": data.get("solver_pass_count", 0 if not successful else ""),
            "solver_rollouts": data.get("solver_rollouts", 0 if not successful else ""),
            "done": data.get("done", ""),
            "raw_job_id": data.get("raw_job_id", ""),
            "hinted_job_id": data.get("hinted_job_id", ""),
            "rollout_file": data.get("rollout_file", ""),
            "generation_archive": data.get("generation_archive", data.get("archive", "")),
        }
    )
    return row


def solve_rows_without_manifest(output_dir: Path, git: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_jobs = load_manifest_jobs(output_dir)
    for path in sorted((output_dir / "solves").glob("**/*.json")):
        manifest_job = manifest_jobs.get(str(path), {})
        rows.append(
            solve_row_from_manifest_job(
                output_dir,
                git,
                {
                    "output": str(path),
                    "job_id": manifest_job.get("job_id", path.stem),
                    "env": manifest_job.get("env", ""),
                    "task_key": manifest_job.get("task_key", ""),
                    "generator_model": manifest_job.get("generator_model", ""),
                    "solver_model": manifest_job.get("solver_model", ""),
                    "eval_k_rollouts": manifest_job.get("eval_k_rollouts", ""),
                    "max_eval_steps": manifest_job.get("max_eval_steps", ""),
                    "generated_file": manifest_job.get("generated_file", ""),
                    "rollout_dir": manifest_job.get("rollout_dir", ""),
                },
            )
        )
    return rows


def summarize_results(output_dir: Path) -> None:
    manifest = load_manifest(output_dir)
    git = manifest.get("git", {}) if isinstance(manifest.get("git", {}), dict) else {}
    manifest_solve_jobs = load_manifest_solve_jobs(output_dir)
    if manifest_solve_jobs:
        solve_rows = [solve_row_from_manifest_job(output_dir, git, job) for job in manifest_solve_jobs]
    else:
        solve_rows = solve_rows_without_manifest(output_dir, git)

    if solve_rows:
        detail_path = output_dir / "solve_results.csv"
        write_csv_rows(detail_path, solve_rows)
    else:
        detail_path = output_dir / "solve_results.csv"

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in solve_rows:
        key = (
            row.get("env_key", ""),
            row.get("generator_model", ""),
            row.get("solver_model", ""),
            row.get("eval_k_rollouts", ""),
            row.get("max_eval_steps", ""),
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (env, generator, solver, k, steps), rows in sorted(groups.items()):
        completed_rows = [row for row in rows if row.get("successful_solve") is True]
        failed_rows = [row for row in rows if row.get("successful_solve") is not True]
        pass_rates = [float(row.get("solver_pass_rate", 0.0) or 0.0) for row in rows]
        completed_pass_rates = [
            float(row["solver_pass_rate"]) for row in completed_rows if isinstance(row.get("solver_pass_rate"), (int, float))
        ]
        task_rewards = [float(row.get("task_gen_reward", 0.0) or 0.0) for row in rows]
        completed_task_rewards = [
            float(row["task_gen_reward"]) for row in completed_rows if isinstance(row.get("task_gen_reward"), (int, float))
        ]
        pass_at_k = [1.0 if row.get("solver_pass_at_k") is True else 0.0 for row in rows]
        summary_rows.append(
            {
                "env_key": env,
                "git_branch": git.get("branch", ""),
                "git_commit": git.get("commit", ""),
                "generator_model": generator,
                "solver_model": solver,
                "eval_k_rollouts": k,
                "max_eval_steps": steps,
                "n": len(rows),
                "n_completed": len(completed_rows),
                "n_failed": len(failed_rows),
                "completion_rate": len(completed_rows) / len(rows) if rows else "",
                "mean_solver_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else "",
                "mean_solver_pass_rate_completed": (
                    sum(completed_pass_rates) / len(completed_pass_rates) if completed_pass_rates else ""
                ),
                "pass_at_k_rate": sum(pass_at_k) / len(pass_at_k) if pass_at_k else "",
                "mean_task_gen_reward": sum(task_rewards) / len(task_rewards) if task_rewards else "",
                "mean_task_gen_reward_completed": (
                    sum(completed_task_rewards) / len(completed_task_rewards) if completed_task_rewards else ""
                ),
            }
        )

    if summary_rows:
        summary_path = output_dir / "baseline_summary.csv"
        write_csv_rows(summary_path, summary_rows)
        print(f"Wrote summaries: {detail_path} and {summary_path}")
    else:
        print(f"No solve JSON files found under {output_dir / 'solves'}")


def write_input_snapshots(
    output_dir: Path,
    candidates: dict[str, list[str]],
    selected_tasks: list[SelectedTask],
    metadata_contexts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    candidate_path = output_dir / "candidate_task_keys.json"
    sampled_path = output_dir / "sampled_task_keys.json"
    selected_context_path = output_dir / "selected_task_contexts.json"

    write_json(candidate_path, {"candidates": candidates})
    write_json(sampled_path, [asdict(task) for task in selected_tasks])
    write_json(
        selected_context_path,
        {
            task.task_key: metadata_contexts[task.task_key]
            for task in selected_tasks
            if task.task_key in metadata_contexts
        },
    )
    return {
        "candidate_task_keys": str(candidate_path),
        "sampled_task_keys": str(sampled_path),
        "selected_task_contexts": str(selected_context_path),
    }


def plan_payload(
    args: argparse.Namespace,
    selected_tasks: list[SelectedTask],
    generation_jobs: list[GenerationJob],
    solve_jobs: list[SolveJob],
) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "tasks_per_env": args.tasks_per_env,
        "generators": args.generators,
        "solvers": args.solvers,
        "base_eval_k_rollouts": args.base_eval_k_rollouts,
        "base_max_eval_steps": args.base_max_eval_steps,
        "sweep": {
            "enabled": not args.no_sweep,
            "all": args.sweep_all,
            "env": args.sweep_env,
            "generator": args.sweep_generator,
            "solver": args.sweep_solver,
            "eval_k_rollouts": args.sweep_eval_k_rollouts,
            "max_eval_steps": args.sweep_max_eval_steps,
        },
        "selected_tasks": [asdict(task) for task in selected_tasks],
        "generation_jobs": [asdict(job) for job in generation_jobs],
        "solve_jobs": [asdict(job) for job in solve_jobs],
    }


def fingerprint_payload(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def manifest_plan_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest.get("plan"), dict):
        return manifest["plan"]
    return {
        "seed": manifest.get("seed"),
        "tasks_per_env": manifest.get("tasks_per_env"),
        "generators": manifest.get("generators"),
        "solvers": manifest.get("solvers"),
        "base_eval_k_rollouts": manifest.get("base_eval_k_rollouts"),
        "base_max_eval_steps": manifest.get("base_max_eval_steps"),
        "sweep": manifest.get("sweep"),
        "selected_tasks": manifest.get("selected_tasks"),
        "generation_jobs": manifest.get("generation_jobs"),
        "solve_jobs": manifest.get("solve_jobs"),
    }


def manifest_plan_fingerprint(manifest: dict[str, Any]) -> str:
    existing = manifest.get("plan_fingerprint")
    if isinstance(existing, str) and existing:
        return existing
    return fingerprint_payload(manifest_plan_payload(manifest))


def ensure_existing_plan_compatible(output_dir: Path, planned_fingerprint: str, force: bool) -> tuple[bool, dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists() or force:
        return False, {}
    existing = load_manifest(output_dir)
    existing_fingerprint = manifest_plan_fingerprint(existing)
    if existing_fingerprint != planned_fingerprint:
        raise RuntimeError(
            f"{output_dir} already contains a different experiment manifest. "
            "Choose a fresh --output-dir or pass --force to replace the plan."
        )
    require_resume_sidecars(output_dir, existing)
    return True, existing


def require_existing_file(path_value: Any, label: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"Existing run is missing required {label} path in manifest.")
    if not Path(path_value).exists():
        raise RuntimeError(f"Existing run is missing required {label}: {path_value}")


def require_resume_sidecars(output_dir: Path, manifest: dict[str, Any]) -> None:
    input_snapshots = manifest.get("input_snapshots", {})
    if not isinstance(input_snapshots, dict):
        raise RuntimeError("Existing run manifest is missing input_snapshots.")
    for label in ("candidate_task_keys", "sampled_task_keys", "selected_task_contexts"):
        require_existing_file(input_snapshots.get(label), f"input snapshot {label}")

    reproducibility = manifest.get("reproducibility", {})
    if not isinstance(reproducibility, dict):
        raise RuntimeError("Existing run manifest is missing reproducibility snapshot metadata.")
    for label in ("git_status", "tracked_worktree_diff", "untracked_files_diff", "working_tree_diff"):
        require_existing_file(reproducibility.get(label), f"reproducibility artifact {label}")
    diff_path = Path(str(reproducibility.get("working_tree_diff", "")))
    expected_sha = reproducibility.get("working_tree_diff_sha256")
    if isinstance(expected_sha, str) and expected_sha and file_sha256(diff_path) != expected_sha:
        raise RuntimeError(f"Existing run reproducibility diff hash mismatch: {diff_path}")

    require_existing_file(str(output_dir / "manifest.json"), "manifest")
    require_existing_file(str(output_dir / "run_metadata.json"), "run metadata")
    require_existing_file(str(output_dir / "jobs.jsonl"), "job list")
    require_existing_file(str(output_dir / "commands.sh"), "commands script")


def build_manifest(
    args: argparse.Namespace,
    git: dict[str, Any],
    selected_tasks: list[SelectedTask],
    generation_jobs: list[GenerationJob],
    solve_jobs: list[SolveJob],
    input_snapshots: dict[str, str],
    plan: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    plan_fingerprint = fingerprint_payload(plan)
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git,
        "plan_fingerprint": plan_fingerprint,
        "plan": plan,
        "seed": args.seed,
        "tasks_per_env": args.tasks_per_env,
        "generators": args.generators,
        "solvers": args.solvers,
        "base_eval_k_rollouts": args.base_eval_k_rollouts,
        "base_max_eval_steps": args.base_max_eval_steps,
        "sweep": {
            "enabled": not args.no_sweep,
            "all": args.sweep_all,
            "env": args.sweep_env,
            "generator": args.sweep_generator,
            "solver": args.sweep_solver,
            "eval_k_rollouts": args.sweep_eval_k_rollouts,
            "max_eval_steps": args.sweep_max_eval_steps,
        },
        "counts": {
            "selected_tasks": len(selected_tasks),
            "generation_jobs": len(generation_jobs),
            "solve_jobs": len(solve_jobs),
            "solver_rollouts": sum(job.eval_k_rollouts for job in solve_jobs),
        },
        "artifact_policy": {
            "capture_all_artifacts": args.capture_all_artifacts,
            "generator_transcripts": args.capture_all_artifacts or args.include_transcript,
            "per_job_logs": True,
            "persistent_solver_rollouts": True,
        },
        "execution": {
            "generate_concurrency": args.generate_concurrency,
            "solve_concurrency": args.solve_concurrency,
        },
        "reproducibility": reproducibility,
        "input_snapshots": input_snapshots,
        "selected_tasks": [asdict(task) for task in selected_tasks],
        "generation_jobs": [asdict(job) for job in generation_jobs],
        "solve_jobs": [asdict(job) for job in solve_jobs],
    }


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    default_output = repo_root / "outputs" / "task_gen_baseline_grid" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_env_file = repo_root / "integrations" / "fleet" / "task_gen_baseline" / "environments.json"
    default_task_key_dir = env_path("FLEET_TASK_KEY_DIR")
    default_metadata = env_path("FLEET_TASK_METADATA_JSON")
    default_candidate_snapshot = repo_root / "integrations" / "fleet" / "task_gen_baseline" / "task_key_candidates.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute jobs. Default is to only write a dry-run manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and commands without executing jobs.")
    parser.add_argument("--phase", choices=["all", "generate", "solve"], default="all")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--summarize-only", help="Only summarize an existing output directory.")
    parser.add_argument("--force", action="store_true", help="Re-run jobs even if output files already exist.")

    parser.add_argument("--env-file", default=default_path(default_env_file))
    parser.add_argument("--envs", help="Comma-separated environment list. Overrides --env-file.")
    parser.add_argument("--task-key-dir", default=str(default_task_key_dir) if default_task_key_dir else "")
    parser.add_argument("--metadata-json", default=str(default_metadata) if default_metadata else "")
    parser.add_argument(
        "--candidate-snapshot-json",
        default=str(default_candidate_snapshot) if default_candidate_snapshot.exists() else "",
        help="Portable env -> task-key-list snapshot. Overrides task-key dir and metadata for candidate loading.",
    )
    parser.add_argument("--tasks-per-env", type=int, default=4)
    parser.add_argument("--allow-fewer-tasks", action="store_true")
    parser.add_argument("--seed", type=int, default=20260520)

    parser.add_argument("--generators", type=parse_csv, default=DEFAULT_MODELS)
    parser.add_argument("--solvers", type=parse_csv, default=DEFAULT_MODELS)
    parser.add_argument("--base-eval-k-rollouts", type=int, default=3)
    parser.add_argument("--base-max-eval-steps", type=int, default=20)
    parser.add_argument("--sweep-eval-k-rollouts", type=parse_int_csv, default=[1, 3, 5])
    parser.add_argument("--sweep-max-eval-steps", type=parse_int_csv, default=[10, 20, 30])
    parser.add_argument("--sweep-env", default="zillow")
    parser.add_argument("--sweep-generator", default=DEFAULT_SWEEP_MODEL)
    parser.add_argument("--sweep-solver", default=DEFAULT_SWEEP_MODEL)
    parser.add_argument("--sweep-all", action="store_true", help="Apply k/max-step Cartesian sweep to every solve job.")
    parser.add_argument("--no-sweep", action="store_true", help="Use only the base k/max-step settings.")

    parser.add_argument("--generator-max-turns", type=int, default=10)
    parser.add_argument("--generator-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tool-mode", choices=["xml", "native"], default="native")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--include-transcript", action="store_true")
    parser.add_argument(
        "--no-capture-all-artifacts",
        dest="capture_all_artifacts",
        action="store_false",
        help="Do not force generator transcripts and persistent per-job artifact capture.",
    )
    parser.set_defaults(capture_all_artifacts=True)

    parser.add_argument("--solver-max-turns", type=int, default=10)
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.7")
    parser.add_argument("--base-quality-reward", type=float, default=0.0)
    parser.add_argument("--enable-hints", action="store_true")

    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--secrets-file", default=str(Path("~/.secrets/api_keys.env").expanduser()))
    parser.add_argument("--generate-concurrency", type=int, default=8)
    parser.add_argument("--solve-concurrency", type=int, default=32)
    parser.add_argument("--limit-generation-jobs", type=int)
    parser.add_argument("--limit-solve-jobs", type=int)
    parser.add_argument("--fail-fast", action="store_true", help="Stop execution on the first failed job.")
    parser.add_argument("--allow-failures", action="store_true", help="Exit 0 after recording failed jobs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize_results(Path(args.summarize_only))
        return
    if args.run and args.dry_run:
        raise ValueError("Pass either --run or --dry-run, not both.")

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_json = Path(args.metadata_json).resolve() if args.metadata_json else None
    task_key_dir = Path(args.task_key_dir).resolve() if args.task_key_dir else None
    candidate_snapshot_path = Path(args.candidate_snapshot_json).resolve() if args.candidate_snapshot_json else None
    git = git_metadata(repo_root)

    candidate_snapshot = load_candidate_snapshot(candidate_snapshot_path)
    envs = load_environments(args, candidate_snapshot)
    candidates, sources, source_hashes = load_candidate_task_keys(envs, candidate_snapshot, task_key_dir, metadata_json)
    metadata_contexts = load_metadata_contexts(metadata_json)
    selected_tasks = sample_tasks(
        envs=envs,
        candidates=candidates,
        sources=sources,
        source_hashes=source_hashes,
        tasks_per_env=args.tasks_per_env,
        seed=args.seed,
        allow_fewer=args.allow_fewer_tasks,
    )
    generation_jobs, solve_jobs = build_jobs(args, output_dir, selected_tasks)
    plan = plan_payload(args, selected_tasks, generation_jobs, solve_jobs)
    plan_fingerprint = fingerprint_payload(plan)

    preserving_existing_plan, existing_manifest = ensure_existing_plan_compatible(output_dir, plan_fingerprint, args.force)
    if preserving_existing_plan:
        manifest = existing_manifest
        input_snapshots = manifest.get("input_snapshots", {}) if isinstance(manifest.get("input_snapshots"), dict) else {}
        print(f"Resuming existing compatible plan; preserving manifest and input snapshots in {output_dir}")
    else:
        input_snapshots = write_input_snapshots(output_dir, candidates, selected_tasks, metadata_contexts)
        reproducibility = write_reproducibility_snapshots(output_dir, repo_root)
        manifest = build_manifest(
            args,
            git,
            selected_tasks,
            generation_jobs,
            solve_jobs,
            input_snapshots,
            plan,
            reproducibility,
        )
        write_json(output_dir / "manifest.json", manifest)
        write_json(
            output_dir / "run_metadata.json",
            {
                "created_at": manifest["created_at"],
                "git": git,
                "counts": manifest["counts"],
                "input_snapshots": input_snapshots,
                "plan_fingerprint": plan_fingerprint,
                "reproducibility": reproducibility,
            },
        )
        write_jobs_files(
            output_dir,
            repo_root,
            Path(args.secrets_file).expanduser(),
            args.python_executable,
            generation_jobs,
            solve_jobs,
        )

    counts = manifest["counts"]
    print(f"Output dir: {output_dir}")
    print(
        "Grid: "
        f"{counts['selected_tasks']} sampled tasks, "
        f"{counts['generation_jobs']} generation jobs, "
        f"{counts['solve_jobs']} solve jobs, "
        f"{counts['solver_rollouts']} solver rollouts"
    )
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Commands: {output_dir / 'commands.sh'}")

    if args.run:
        try:
            run_jobs(args, repo_root, output_dir, generation_jobs, solve_jobs)
        finally:
            summarize_results(output_dir)
    else:
        print("Dry run only. Pass --run to execute.")


if __name__ == "__main__":
    main()
