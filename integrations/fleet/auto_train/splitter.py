"""90/10 train/holdout split for the Tinker auto-train pipeline.

Reads an in-memory task list (the same shape exporter.build_openenv_tasks
produces), converts each task to the SkyRL dataset record format via
prepare_dataset._task_to_record, then writes train.parquet + holdout.parquet
that main_fleet_tinker.py can consume as --dataset-file / --eval-dataset-file.

Deterministic per (dataset_key, modality, seed): same tasks always land in
the same split, so a re-run reports the same pre/post pass rates.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Sequence

from datasets import Dataset

from integrations.fleet.prepare_dataset import _task_to_record

from .tinker_config import HOLDOUT_FRACTION, HOLDOUT_SEED

logger = logging.getLogger(__name__)


def _tasks_to_records(tasks: Sequence[dict]) -> list[dict]:
    """Convert exported task dicts to SkyRL dataset records.

    Skips tasks without a usable prompt / task_key (matching prepare_dataset
    behavior). Preserves env_key as the SkyRL `data_source` for per-env metrics.
    """
    records: list[dict] = []
    for task in tasks:
        env_key = task.get("env_key") or "unknown"
        record = _task_to_record(task, env_key=env_key, env_class="fleet_task")
        if record is not None:
            records.append(record)
    return records


def split_90_10(
    tasks: Sequence[dict],
    output_dir: str,
    dataset_key: str,
    modality: str,
    seed: int = HOLDOUT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> tuple[str, str, int, int]:
    """Split tasks into train (90%) + holdout (10%) parquet files.

    Returns (train_parquet_path, holdout_parquet_path, n_train, n_holdout).

    The split is deterministic: sorts records by task_key first so the upstream
    Supabase query order doesn't matter, then shuffles with the supplied seed
    salted by dataset_key + modality so different (dataset, modality) pairs get
    different but reproducible splits.
    """
    records = _tasks_to_records(tasks)
    if not records:
        raise ValueError(f"No usable records for {dataset_key}/{modality}")
    records.sort(key=lambda r: r["task_key"])

    salt = f"{dataset_key}:{modality}:{seed}"
    rng = random.Random(salt)
    rng.shuffle(records)

    n_holdout = max(1, int(round(len(records) * holdout_fraction)))
    n_holdout = min(n_holdout, len(records) - 1)
    holdout_records = records[:n_holdout]
    train_records = records[n_holdout:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train_{dataset_key}_{modality}.parquet")
    holdout_path = os.path.join(output_dir, f"holdout_{dataset_key}_{modality}.parquet")

    Dataset.from_list(train_records).to_parquet(train_path)
    Dataset.from_list(holdout_records).to_parquet(holdout_path)

    logger.info(
        "Split %s/%s: %d total → %d train + %d holdout (seed=%s)",
        dataset_key, modality, len(records), len(train_records), len(holdout_records), salt,
    )
    return train_path, holdout_path, len(train_records), len(holdout_records)
