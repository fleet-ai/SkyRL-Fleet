"""CLI entry point: python -m integrations.fleet.auto_train ...

Subcommands:
  trigger    Discover new (dataset, modality) pairs, smoke, export, launch
  status     Print processed pairs from S3 state
  seed       Force-seed the S3 state with all current datasets (first-run helper)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import httpx

from .config import MIN_TASKS_TO_LAUNCH, S3_STATE_KEY, SUPPORTED_MODALITIES
from .discovery import (
    get_dataset_modalities,
    list_active_datasets,
)
from .exporter import apply_export_cap, build_openenv_tasks, export_to_s3
from .launcher import launch_training
from .notify import (
    notify_below_threshold,
    notify_launch,
    notify_launch_failure,
    notify_not_implemented,
    notify_smoke_failure,
    notify_tinker_eval,
)
from .smoke import run_smoke_test
from .state import ProcessedState
from .tinker_config import (
    DEFAULT_BASE_MODEL as TINKER_DEFAULT_BASE_MODEL,
)
from .tinker_config import (
    DEFAULT_LORA_RANK as TINKER_DEFAULT_LORA_RANK,
)
from .tinker_config import (
    DEFAULT_MAX_CONCURRENT as TINKER_DEFAULT_MAX_CONCURRENT,
)
from .tinker_config import (
    DEFAULT_NUM_STEPS as TINKER_DEFAULT_NUM_STEPS,
)
from .tinker_config import (
    TINKER_MODALITY_SUPPORT,
    TINKER_STATE_KEY,
)
from .tinker_launcher import launch_training as launch_tinker_training

logger = logging.getLogger("auto_train")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _state_key_for_backend(backend: str) -> str:
    return TINKER_STATE_KEY if backend == "tinker" else S3_STATE_KEY


def cmd_status(args: argparse.Namespace) -> int:
    state = ProcessedState(key=_state_key_for_backend(args.backend))
    print(f"Backend: {args.backend}")
    print(f"State key: {state.key}")
    print(f"Seeded at: {state._seeded_at or '(not seeded yet)'}")
    print(f"Seen datasets: {len(state.all_seen())}")
    pairs = sorted(state.all_processed())
    print(f"Processed pairs: {len(pairs)}")
    for dataset_key, modality in pairs:
        print(f"  {dataset_key:<40} {modality}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    state = ProcessedState(key=_state_key_for_backend(args.backend))
    if state.is_seeded and not args.force:
        logger.error("State already seeded; pass --force to overwrite")
        return 1
    with httpx.Client() as client:
        datasets = list_active_datasets(client)
    keys = [d.dataset_key for d in datasets]
    if args.force:
        state._seen_datasets = set()
        state._processed_pairs = set()
        state._seeded_at = None
    state.seed(keys)
    print(f"Seeded {len(keys)} datasets")
    return 0


def _process_one(
    client: httpx.Client,
    dataset,
    modality: str,
    state: ProcessedState,
    *,
    dry_run: bool,
    skip_smoke: bool,
    api_key: str,
    backend: str = "sky",
    tinker_base_model: str = TINKER_DEFAULT_BASE_MODEL,
    tinker_num_steps: int = TINKER_DEFAULT_NUM_STEPS,
    tinker_lora_rank: int = TINKER_DEFAULT_LORA_RANK,
    tinker_max_concurrent: int = TINKER_DEFAULT_MAX_CONCURRENT,
) -> int:
    """Returns 0 on success, non-zero on failure (still marks processed if appropriate)."""
    dataset_key = dataset.dataset_key

    # Backend-specific modality gate. The Tinker pipeline currently supports
    # the same modalities as the SkyPilot pipeline, but keep the gate explicit
    # so adding/removing one doesn't silently change behavior on the other.
    if backend == "tinker":
        backend_supported = TINKER_MODALITY_SUPPORT
    else:
        backend_supported = set(SUPPORTED_MODALITIES)

    if modality not in SUPPORTED_MODALITIES:
        logger.warning("Unknown modality %s for %s; skipping", modality, dataset_key)
        return 0
    if modality not in backend_supported:
        logger.warning(
            "modality %s not supported by backend=%s for %s; skipping",
            modality,
            backend,
            dataset_key,
        )
        notify_not_implemented(dataset_key, modality, 0)
        state.mark_processed(dataset_key, modality)
        return 0

    # Build tasks once, used by both smoke test and exporter
    logger.info("Building task list for %s/%s", dataset_key, modality)
    try:
        tasks = build_openenv_tasks(client, dataset.id, modality)
    except Exception as e:
        logger.exception("Failed to build tasks for %s/%s: %s", dataset_key, modality, e)
        notify_launch_failure(dataset_key, modality, f"task build error: {e}")
        return 1
    if not tasks:
        logger.warning("No exportable tasks for %s/%s; skipping", dataset_key, modality)
        state.mark_processed(dataset_key, modality)
        return 0

    # Minimum-tasks gate. Datasets below threshold will fail training with
    # "dataset should be at least as large as train_batch_size" regardless of
    # smoke result. Slack-alert and skip rather than burn a cluster.
    if len(tasks) < MIN_TASKS_TO_LAUNCH:
        logger.warning(
            "Skipping %s/%s: %d tasks < %d threshold",
            dataset_key,
            modality,
            len(tasks),
            MIN_TASKS_TO_LAUNCH,
        )
        notify_below_threshold(dataset_key, modality, len(tasks), MIN_TASKS_TO_LAUNCH)
        state.mark_processed(dataset_key, modality)
        return 0

    # Smoke test
    if not skip_smoke:
        if not api_key:
            logger.error("FLEET_API_KEY not set; cannot run smoke test")
            return 2
        try:
            report = run_smoke_test(tasks, modality, dataset_key, api_key)
        except NotImplementedError as e:
            notify_not_implemented(dataset_key, modality, len(tasks))
            state.mark_processed(dataset_key, modality)
            logger.warning("%s", e)
            return 0
        if not report.passed:
            notify_smoke_failure(dataset_key, modality, report)
            logger.error("Smoke failed for %s/%s, NOT launching", dataset_key, modality)
            # Do NOT mark processed; retry on next tick
            return 2

    # Cap to bound training time. Deterministic per dataset_key so re-runs
    # of the same dataset always export the same subset.
    tasks = apply_export_cap(tasks, dataset_key)

    # Export
    if dry_run:
        logger.info(
            "[DRY RUN] Would upload %d tasks to s3 for %s/%s",
            len(tasks),
            dataset_key,
            modality,
        )
        s3_uri = f"s3://(dry-run)/{dataset_key}/openenv/all_{modality}.json"
    else:
        try:
            s3_uri = export_to_s3(tasks, dataset_key, modality)
        except Exception as e:
            logger.exception("Export failed: %s", e)
            notify_launch_failure(dataset_key, modality, f"S3 export error: {e}")
            return 1

    # Launch — backend-specific.
    if backend == "tinker":
        ok, eval_results = launch_tinker_training(
            dataset_key=dataset_key,
            modality=modality,
            tasks=tasks,
            dry_run=dry_run,
            base_model=tinker_base_model,
            num_steps=tinker_num_steps,
            lora_rank=tinker_lora_rank,
            max_concurrent=tinker_max_concurrent,
        )
        if not ok:
            notify_launch_failure(
                dataset_key,
                modality,
                "fleet-tinker-tool-use-run.sh failed or produced no results JSON",
            )
            return 1
        if not dry_run:
            state.mark_processed(dataset_key, modality)
            if eval_results is not None:
                notify_tinker_eval(
                    dataset_key=dataset_key,
                    modality=modality,
                    base_model=eval_results.get("model_name", tinker_base_model),
                    pre_pass_rate=float(eval_results.get("pre_pass_rate") or 0.0),
                    post_pass_rate=float(eval_results.get("post_pass_rate") or 0.0),
                    n_holdout=int(eval_results.get("n_holdout") or 0),
                    num_steps=int(eval_results.get("num_steps") or tinker_num_steps),
                    wandb_url=eval_results.get("wandb_url"),
                )
            else:
                # Dry run reached this branch above already; this is a real run
                # that exited 0 but with no results JSON — already notified
                # via notify_launch_failure above.
                pass
        return 0

    # Default: SkyPilot launcher.
    launched = launch_training(dataset_key, modality, dry_run=dry_run)
    if not launched:
        notify_launch_failure(dataset_key, modality, "fleet-launch.sh returned non-zero")
        return 1

    if not dry_run:
        state.mark_processed(dataset_key, modality)
        notify_launch(dataset_key, modality, len(tasks), s3_uri)
    return 0


def cmd_trigger(args: argparse.Namespace) -> int:
    state = ProcessedState(key=_state_key_for_backend(args.backend))
    api_key = os.environ.get("FLEET_API_KEY", "")
    # Tag the env so notify._channel() can route Tinker runs to a separate
    # Slack channel without per-call boilerplate.
    if args.backend == "tinker":
        os.environ["AUTO_TRAIN_BACKEND"] = "tinker"

    with httpx.Client() as client:
        datasets = list_active_datasets(client, team_id=args.team_id)

        # First-run seed: every current dataset marked 'seen' so we don't
        # retroactively trigger training for the existing datasets.
        # --ignore-seed bypasses this for targeted testing.
        if not state.is_seeded and not args.ignore_seed:
            keys = [d.dataset_key for d in datasets]
            if args.dry_run:
                logger.info("[DRY RUN] Would seed state with %d datasets", len(keys))
                return 0
            state.seed(keys)
            logger.info("First-run seed complete; no training launched this tick")
            return 0

        # Per-tick: only enumerate modalities for datasets we haven't seen
        # (unless --ignore-seed is set, in which case treat all as candidates).
        if args.ignore_seed:
            new_datasets = list(datasets)
            logger.info("--ignore-seed: treating all %d datasets as candidates", len(new_datasets))
        else:
            new_datasets = [d for d in datasets if not state.is_seen(d.dataset_key)]
            logger.info("Active datasets: %d, new since last tick: %d", len(datasets), len(new_datasets))

        if args.max_datasets and len(new_datasets) > args.max_datasets:
            logger.info("Capping to first %d datasets (newest-first)", args.max_datasets)
            new_datasets = new_datasets[: args.max_datasets]

        # Fail-fast: after the first launch failure we stop processing more
        # datasets in this tick. Otherwise a single bad cluster path (e.g.
        # GCP H200 spot Ray crashes) would burn ~$100/hr * N as we walk the
        # max_datasets cap. The next cron tick can retry.
        # Gate skips (below threshold), modality-skips, and NotImplementedError
        # paths return 0 and do NOT trigger fail-fast.
        exit_code = 0
        launch_failed = False
        for dataset in new_datasets:
            if launch_failed:
                logger.warning(
                    "Stopping early: a prior launch in this tick failed. "
                    "Remaining datasets will be picked up on the next cron tick."
                )
                break
            modalities = get_dataset_modalities(client, dataset.id)
            if not modalities:
                logger.info("No supported tasks in %s; marking seen", dataset.dataset_key)
                if not args.dry_run:
                    state.mark_seen(dataset.dataset_key)
                continue
            dataset_ok = True
            for modality in modalities:
                if state.is_processed(dataset.dataset_key, modality) and not args.ignore_seed:
                    continue
                if args.modality and modality != args.modality:
                    continue
                rc = _process_one(
                    client,
                    dataset,
                    modality,
                    state,
                    dry_run=args.dry_run,
                    skip_smoke=args.skip_smoke,
                    api_key=api_key,
                    backend=args.backend,
                    tinker_base_model=args.tinker_base_model,
                    tinker_num_steps=args.tinker_num_steps,
                    tinker_lora_rank=args.tinker_lora_rank,
                    tinker_max_concurrent=args.tinker_max_concurrent,
                )
                if rc != 0:
                    dataset_ok = False
                    exit_code = rc
                    # rc=2 is smoke-test failure (no cluster cost). Other
                    # non-zero values mean we attempted a sky launch and it
                    # failed, possibly leaving a cluster up. Fail-fast.
                    if rc != 2:
                        launch_failed = True
                        break
            # Only mark the dataset 'seen' if every modality landed in
            # processed_pairs (success or expected skip). Otherwise we leave
            # it unseen so the next tick retries the failed modalities.
            if dataset_ok and not args.dry_run:
                state.mark_seen(dataset.dataset_key)
        return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auto_train")
    parser.add_argument("-v", "--verbose", action="store_true")
    # Backend selects between the SkyPilot launcher (sky launch + vLLM training)
    # and the Tinker launcher (Tinker cloud LoRA + held-out eval reporting).
    # Each backend has a separate S3 state file so a single dataset can be
    # processed by both pipelines independently.
    parser.add_argument(
        "--backend",
        choices=("sky", "tinker"),
        default="sky",
        help="Training backend (default: sky, the SkyPilot/vLLM pipeline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_trig = sub.add_parser("trigger", help="Discover and process new (dataset, modality) pairs")
    p_trig.add_argument("--modality", choices=SUPPORTED_MODALITIES, default=None)
    p_trig.add_argument("--dry-run", action="store_true")
    p_trig.add_argument("--skip-smoke", action="store_true")
    p_trig.add_argument("--team-id", default=None, help="Filter datasets by team_id (e.g., fleet team a1025f0b...)")
    p_trig.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Cap the number of new datasets processed this tick (newest-first)",
    )
    p_trig.add_argument(
        "--ignore-seed",
        action="store_true",
        help="Treat all datasets as candidates (bypass seen_datasets and processed_pairs filters). For targeted testing.",
    )
    p_trig.add_argument(
        "--tinker-base-model",
        default=TINKER_DEFAULT_BASE_MODEL,
        help="Base model for --backend tinker (HF id, e.g., moonshotai/Kimi-K2.6).",
    )
    p_trig.add_argument(
        "--tinker-num-steps", type=int, default=TINKER_DEFAULT_NUM_STEPS, help="Training steps for --backend tinker."
    )
    p_trig.add_argument(
        "--tinker-lora-rank", type=int, default=TINKER_DEFAULT_LORA_RANK, help="LoRA rank for --backend tinker."
    )
    p_trig.add_argument(
        "--tinker-max-concurrent",
        type=int,
        default=TINKER_DEFAULT_MAX_CONCURRENT,
        help="Max concurrent rollouts for --backend tinker.",
    )
    p_trig.set_defaults(func=cmd_trigger)

    p_status = sub.add_parser("status", help="Show processed pairs")
    p_status.set_defaults(func=cmd_status)

    p_seed = sub.add_parser("seed", help="Force-seed state with all current datasets")
    p_seed.add_argument("--force", action="store_true", help="Overwrite existing seed")
    p_seed.set_defaults(func=cmd_seed)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
