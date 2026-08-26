#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2"]
# ///
"""Submit a SkyRL-Fleet run payload to rl1 as a Kueue-queued batch Job.

The payload is the jobs contract: a run is a name, an image, a command, a
worker shape, env vars, and secret names. Everything else in the manifest
(queue label, suspend, topology annotation, node placement, SFS mount,
trace/checkpoint locations) is the platform side, rendered here.

Usage:
    ./submit_run.py examples/tool-use-qwen35-9b.json
    ./submit_run.py examples/browser-use-qwen35-9b.json --pool gpu-h200
    ./submit_run.py payload.json --dry-run        # print manifest, don't apply

Multi-worker runs (workers > 1) need a RayJob and are not supported by this
submitter; it errors on them.
"""

import argparse
import json
import subprocess
import sys

from pydantic import BaseModel, Field

KUBE_CONTEXT = "fleet-training-rl1-us-east-1"
NAMESPACE = "fleet-train-jobs"
SFS_ROOT = "/mnt/sfs/skyrl-fleet"


class RunPayload(BaseModel):
    """The POST /v1/runs shape: the job is five things plus secrets."""

    name: str = Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,50}[a-z0-9])?$")
    image: str
    command: str
    workers: int = 1
    gpus_per_worker: int = 8
    env: dict[str, str] = {}
    secrets: list[str] = []


def build_manifest(run: RunPayload, pool: str, submitted_by: str) -> dict:
    run_dir = f"{SFS_ROOT}/{run.name}"
    env = [
        {"name": "RUN_DIR", "value": run_dir},
        {"name": "SKYPILOT_NUM_GPUS_PER_NODE", "value": str(run.gpus_per_worker)},
        {"name": "WANDB_ENTITY", "value": "thefleet"},
    ] + [{"name": k, "value": v} for k, v in run.env.items()]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": run.name,
            "namespace": NAMESPACE,
            "labels": {
                "app": "skyrl-fleet",
                # every GPU job goes through Kueue; unqueued jobs get preempted
                "kueue.x-k8s.io/queue-name": "training-lq",
            },
            "annotations": {"fleet.ai/submitted-by": submitted_by},
        },
        "spec": {
            "suspend": True,  # Kueue unsuspends on admission
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {"app": "skyrl-fleet", "run": run.name},
                    # GPU ClusterQueue flavors admit only topology-aware pod sets
                    "annotations": {
                        "kueue.x-k8s.io/podset-required-topology": "topology.nebius.com/tier-1"
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"kubernetes.io/os": "linux", "workload": pool},
                    "tolerations": [
                        {"key": "workload", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                    "imagePullSecrets": [{"name": "ghcr-pull"}],
                    "volumes": [
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
                        },
                    ],
                    "initContainers": [
                        {
                            "name": "sfs-init",
                            "image": "busybox:1.36",
                            "securityContext": {"runAsUser": 0},
                            "command": [
                                "sh",
                                "-c",
                                f"mkdir -p {run_dir} && chmod 777 {run_dir}",
                            ],
                            "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                        }
                    ],
                    "containers": [
                        {
                            "name": "trainer",
                            "image": run.image,
                            "env": env,
                            "envFrom": [{"secretRef": {"name": s}} for s in run.secrets],
                            "volumeMounts": [
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "48",
                                    "memory": "700Gi",
                                    "nvidia.com/gpu": run.gpus_per_worker,
                                },
                                "limits": {
                                    "cpu": "90",
                                    "memory": "900Gi",
                                    "nvidia.com/gpu": run.gpus_per_worker,
                                },
                            },
                            "command": ["bash", "integrations/fleet/rl1/entrypoint.sh"],
                            "args": [run.command],
                        }
                    ],
                },
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("payload", help="path to a run payload JSON")
    ap.add_argument("--pool", default="gpu-b200", help="node pool (workload label)")
    ap.add_argument("--submitted-by", default="deniz@fleet.so")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.payload) as f:
        run = RunPayload.model_validate_json(f.read())

    if run.workers != 1:
        sys.exit("workers > 1 needs a RayJob; this submitter is single-node only")

    manifest = json.dumps(build_manifest(run, args.pool, args.submitted_by), indent=2)
    if args.dry_run:
        print(manifest)
        return

    subprocess.run(
        ["kubectl", "--context", KUBE_CONTEXT, "-n", NAMESPACE, "apply", "-f", "-"],
        input=manifest.encode(),
        check=True,
    )
    print(f"submitted: {run.name}  (queued in training-lq; suspend lifts on admission)")
    print(f"logs:   kubectl --context {KUBE_CONTEXT} -n {NAMESPACE} logs -f job/{run.name}")
    print(f"driver: {SFS_ROOT}/{run.name}/driver.log")


if __name__ == "__main__":
    main()
