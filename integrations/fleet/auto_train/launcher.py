"""Launch training via scripts/fleet-launch.sh."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import MODALITY_YAML_MAP, REQUIRED_LAUNCH_ENV_VARS, SUPPORTED_MODALITIES

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts" / "fleet-launch.sh").exists():
            return parent
    raise FileNotFoundError("scripts/fleet-launch.sh not found above auto_train module")


def build_launch_command(
    dataset_key: str,
    modality: str,
    env_vars: dict[str, str],
    extra_sky_args: list[str] | None = None,
) -> list[str]:
    if modality not in SUPPORTED_MODALITIES:
        raise NotImplementedError(
            f"modality {modality!r} not supported (dataset={dataset_key})"
        )

    root = _repo_root()
    yaml_path = root / MODALITY_YAML_MAP[modality]
    launch_script = root / "scripts" / "fleet-launch.sh"

    cmd: list[str] = [
        "bash", str(launch_script), str(yaml_path),
        "--env", f"DATA_VERSION={dataset_key}",
        "--env", f"MODALITY={modality}",
    ]
    for k, v in env_vars.items():
        cmd.extend(["--env", f"{k}={v}"])
    cmd.extend(["--retry-until-up", "-y"])
    # --down: terminate the cluster after the run script exits (success or
    # failure). Without this, sky launch leaves the cluster up for debug, and
    # when the run script fails (as it did 2026-05-26 with LocalRayletDiedError)
    # we orphan ~$50/hr H200:8 nodes. Note: this does NOT help with the GH
    # workflow timeout case (sky launch blocks; if the runner is killed mid-
    # launch, --down never fires). For that, the workflow has a best-effort
    # cleanup step that runs on always().
    cmd.append("--down")
    if extra_sky_args:
        cmd.extend(extra_sky_args)
    return cmd


def launch_training(
    dataset_key: str,
    modality: str,
    dry_run: bool = False,
    extra_sky_args: list[str] | None = None,
) -> bool:
    """Returns True if `sky launch` exited 0 (success)."""
    missing = [v for v in REQUIRED_LAUNCH_ENV_VARS if not os.environ.get(v)]
    if missing and not dry_run:
        logger.error("Missing required env vars: %s", missing)
        return False

    env_vars = {k: os.environ.get(k, "") for k in REQUIRED_LAUNCH_ENV_VARS}
    cmd = build_launch_command(dataset_key, modality, env_vars, extra_sky_args)

    if dry_run:
        # Redact credential values in dry-run output
        printable = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "--env" and i + 1 < len(cmd):
                kv = cmd[i + 1]
                k, _, _v = kv.partition("=")
                if k in REQUIRED_LAUNCH_ENV_VARS:
                    printable.extend([cmd[i], f"{k}=<redacted>"])
                else:
                    printable.extend([cmd[i], cmd[i + 1]])
                i += 2
            else:
                printable.append(cmd[i])
                i += 1
        logger.info("[DRY RUN] %s", " ".join(printable))
        return True

    logger.info("Launching: %s/%s", dataset_key, modality)

    # SkyPilot's sync_workdir does a git clone on the launch target, which on
    # the runpod slurm head intermittently fails ("Git clone failed on
    # 31.24.80.55-13122: /workspace/.sky_clusters/sky-XXXX-runner-NNN/sky_workdir").
    # Sky's --retry-until-up only retries cluster provisioning, not the
    # post-provision workdir sync. Retry the whole launch up to 3 times when
    # we see that exact failure mode. The workflow's "Best-effort cluster
    # cleanup" step downs any orphan sky-*-runner clusters between launches.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        proc = subprocess.Popen(
            cmd, cwd=str(_repo_root()),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        tail: list[str] = []
        for line in iter(proc.stdout.readline, ""):  # type: ignore[arg-type]
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
        proc.wait()
        if proc.returncode == 0:
            logger.info("fleet-launch.sh succeeded for %s/%s (attempt %d)",
                        dataset_key, modality, attempt)
            return True
        tail_text = "".join(tail)
        if "Git clone failed" in tail_text and attempt < max_attempts:
            logger.warning(
                "fleet-launch.sh hit Git clone failure (attempt %d/%d) for %s/%s; retrying",
                attempt, max_attempts, dataset_key, modality,
            )
            continue
        logger.error(
            "fleet-launch.sh failed (exit=%d, attempt %d) for %s/%s",
            proc.returncode, attempt, dataset_key, modality,
        )
        return False
    return False
