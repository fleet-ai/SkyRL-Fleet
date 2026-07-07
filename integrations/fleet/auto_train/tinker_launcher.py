"""Tinker variant of the auto-train launcher.

Drives integrations.fleet.entrypoints.main_fleet_tinker (via the existing
scripts/fleet-tinker-tool-use-run.sh wrapper) per (dataset, modality) pair.

Mirrors the public surface of launcher.py:
  - build_launch_command()
  - launch_training()

Key differences vs the SkyPilot launcher:
  - No sky launch (Tinker runs in Tinker cloud; the GH Actions runner only
    drives API calls). The runner needs no GPU.
  - 90/10 train/holdout split is done in-process via splitter.split_90_10
    before invoking the entrypoint; the entrypoint reports pre/post pass
    rates back through --results-out.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .splitter import split_90_10
from .tinker_config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_NUM_STEPS,
    EVAL_RESULTS_DIR,
    REQUIRED_TINKER_ENV_VARS,
    TINKER_MODALITY_SUPPORT,
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts" / "fleet-tinker-tool-use-run.sh").exists():
            return parent
    raise FileNotFoundError("scripts/fleet-tinker-tool-use-run.sh not found above auto_train module")


def build_launch_env(
    dataset_key: str,
    modality: str,
    tasks_file: str,
    dataset_file: str,
    eval_dataset_file: str,
    results_out: str,
    base_model: str = DEFAULT_BASE_MODEL,
    num_steps: int = DEFAULT_NUM_STEPS,
    lora_rank: int = DEFAULT_LORA_RANK,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> dict[str, str]:
    """Env dict for fleet-tinker-tool-use-run.sh. Inherits caller env."""
    if modality not in TINKER_MODALITY_SUPPORT:
        raise NotImplementedError(f"modality {modality!r} not supported by Tinker launcher " f"(dataset={dataset_key})")
    env = os.environ.copy()
    env.update(
        {
            "TASKS_FILE": tasks_file,
            "DATASET_FILE": dataset_file,
            "EVAL_DATASET_FILE": eval_dataset_file,
            "RESULTS_OUT": results_out,
            "MODEL_NAME": base_model,
            "MAX_STEPS": str(num_steps),
            "LORA_RANK": str(lora_rank),
            "MAX_CONCURRENT": str(max_concurrent),
            "EVAL_BEFORE_TRAIN": "1",
            "WANDB_NAME": f"tinker_{dataset_key}_{modality}",
            "AUTO_TRAIN_BACKEND": "tinker",  # routes notify to the Tinker Slack channel
        }
    )
    return env


def _write_tasks_json(tasks: list[dict], dataset_key: str, modality: str) -> str:
    """Write the in-memory task list to a temp tasks.json the entrypoint can
    open as --tasks-file. The shape matches what exporter.export_to_s3 writes
    to S3, which is also what fleet-common-setup.sh / prepare_dataset.py expect."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{dataset_key}_{modality}.json",
        prefix="tinker_tasks_",
        delete=False,
    )
    json.dump({"tasks": list(tasks)}, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def launch_training(
    dataset_key: str,
    modality: str,
    tasks: list[dict],
    dry_run: bool = False,
    base_model: str = DEFAULT_BASE_MODEL,
    num_steps: int = DEFAULT_NUM_STEPS,
    lora_rank: int = DEFAULT_LORA_RANK,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    eval_results_dir: str = EVAL_RESULTS_DIR,
) -> tuple[bool, dict | None]:
    """Returns (success, eval_results_dict).

    eval_results_dict is the parsed JSON written by main_fleet_tinker.py via
    --results-out, or None if the run failed before producing it. Success is
    True iff the shell script exits 0 AND the eval results JSON was written.
    """
    missing = [v for v in REQUIRED_TINKER_ENV_VARS if not os.environ.get(v)]
    if missing and not dry_run:
        logger.error("Missing required Tinker env vars: %s", missing)
        return False, None

    if modality not in TINKER_MODALITY_SUPPORT:
        logger.warning(
            "modality %r not supported by Tinker launcher (dataset=%s)",
            modality,
            dataset_key,
        )
        return False, None

    root = _repo_root()
    eval_results_dir_abs = (root / eval_results_dir).resolve()
    eval_results_dir_abs.mkdir(parents=True, exist_ok=True)
    results_out = str(eval_results_dir_abs / f"{dataset_key}_{modality}.json")

    if dry_run:
        logger.info(
            "[DRY RUN] tinker launch %s/%s base=%s steps=%d (would write %s)",
            dataset_key,
            modality,
            base_model,
            num_steps,
            results_out,
        )
        return True, None

    tasks_file = _write_tasks_json(tasks, dataset_key, modality)
    try:
        train_parquet, holdout_parquet, n_train, n_holdout = split_90_10(
            tasks=tasks,
            output_dir=str(eval_results_dir_abs),
            dataset_key=dataset_key,
            modality=modality,
        )
    except ValueError as e:
        logger.exception("Splitter failed for %s/%s: %s", dataset_key, modality, e)
        return False, None

    env = build_launch_env(
        dataset_key=dataset_key,
        modality=modality,
        tasks_file=tasks_file,
        dataset_file=train_parquet,
        eval_dataset_file=holdout_parquet,
        results_out=results_out,
        base_model=base_model,
        num_steps=num_steps,
        lora_rank=lora_rank,
        max_concurrent=max_concurrent,
    )

    script = root / "scripts" / "fleet-tinker-tool-use-run.sh"
    cmd = ["bash", str(script)]
    logger.info(
        "Launching Tinker: dataset=%s modality=%s model=%s steps=%d n_train=%d n_holdout=%d",
        dataset_key,
        modality,
        base_model,
        num_steps,
        n_train,
        n_holdout,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):  # type: ignore[arg-type]
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()

    if proc.returncode != 0:
        logger.error(
            "fleet-tinker-tool-use-run.sh failed (exit=%d) for %s/%s",
            proc.returncode,
            dataset_key,
            modality,
        )
        return False, None

    if not os.path.exists(results_out):
        logger.error("Run exited 0 but no results file at %s", results_out)
        return False, None
    try:
        with open(results_out) as fh:
            results = json.load(fh)
    except Exception as e:
        logger.exception("Failed to parse results JSON at %s: %s", results_out, e)
        return False, None

    results.setdefault("dataset_key", dataset_key)
    results.setdefault("modality", modality)
    return True, results
