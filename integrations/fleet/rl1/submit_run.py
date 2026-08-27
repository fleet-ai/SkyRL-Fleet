#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2"]
# ///
"""Submit a SkyRL-Fleet run payload to rl1 as a Kueue-queued RayJob.

This script stands in lieu of the future runs API: it is boxes 2 and 3 of
the design in README.md (render ONE RayJob manifest from the payload, then
kubectl apply). The payload is the jobs contract: a run is a name, an
image, a command, a worker shape, env vars, and secret names. Everything
else in the manifest (queue label, suspend, topology annotation, node
placement, SFS mount, log capture) is the platform side, rendered here.

The RayJob shape: the head group carries the first gpus_per_worker GPUs and
the driver runs there (same layout as the SkyPilot path, where the driver
lives on a GPU node — the infra nodes are too small for a VL driver).
workers > 1 adds a GPU worker group with workers-1 replicas; Kueue admits
head + workers as a gang on one tier-1 topology domain.

Usage:
    ./submit_run.py examples/tool-use-qwen35-9b.json
    ./submit_run.py payload.json --dry-run        # print manifest, don't apply
"""

import argparse
import json
import shlex
import subprocess

from pydantic import BaseModel, Field

KUBE_CONTEXT = "fleet-training-rl1-us-east-1"
NAMESPACE = "fleet-train-jobs"
SFS_ROOT = "/mnt/sfs/skyrl-fleet"
RAY_VERSION = "2.51.1"  # must match ray inside the trainer image
TOPOLOGY_ANNOTATION = {
    # GPU ClusterQueue flavors admit only topology-aware pod sets
    "kueue.x-k8s.io/podset-required-topology": "topology.nebius.com/tier-1"
}


class RunPayload(BaseModel):
    """The POST /v1/runs shape: the job is five things plus secrets."""

    name: str = Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,50}[a-z0-9])?$")
    image: str
    command: str
    workers: int = Field(default=1, ge=1)
    gpus_per_worker: int = Field(default=8, ge=1, le=8)
    env: dict[str, str] = {}
    secrets: list[str] = []


def gpu_pod_template(run: RunPayload, pool: str, is_head: bool) -> dict:
    """One GPU pod: head (driver + raylet + actors) or extra worker (raylet + actors)."""
    env = [
        {"name": "WANDB_ENTITY", "value": "thefleet"},
    ]
    if is_head:
        env = [
            {"name": "RUN_DIR", "value": f"{SFS_ROOT}/{run.name}"},
            {"name": "FLEET_EXTERNAL_RAY", "value": "1"},
            {"name": "WORKERS", "value": str(run.workers)},
            {"name": "SKYPILOT_NUM_GPUS_PER_NODE", "value": str(run.gpus_per_worker)},
            {"name": "WANDB_ENTITY", "value": "thefleet"},
        ] + [{"name": k, "value": v} for k, v in run.env.items()]

    return {
        "metadata": {"annotations": dict(TOPOLOGY_ANNOTATION)},
        "spec": {
            "nodeSelector": {"kubernetes.io/os": "linux", "workload": pool},
            "tolerations": [
                {"key": "workload", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "imagePullSecrets": [{"name": "ghcr-pull"}],
            "volumes": [
                {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
            ],
            "initContainers": [
                {
                    "name": "sfs-init",
                    "image": "busybox:1.36",
                    "securityContext": {"runAsUser": 0},
                    "command": [
                        "sh",
                        "-c",
                        f"mkdir -p {SFS_ROOT}/{run.name} && chown -R 1000:100 {SFS_ROOT}/{run.name}",
                    ],
                    "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                }
            ],
            "containers": [
                {
                    "name": "ray-head" if is_head else "ray-worker",
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
                }
            ],
        },
    }


def build_manifest(run: RunPayload, pool: str, submitted_by: str) -> dict:
    cluster_spec = {
        "rayVersion": RAY_VERSION,
        "headGroupSpec": {
            "rayStartParams": {"dashboard-host": "0.0.0.0"},
            "template": gpu_pod_template(run, pool, is_head=True),
        },
    }
    if run.workers > 1:
        cluster_spec["workerGroupSpecs"] = [
            {
                "groupName": "gpu-worker",
                "replicas": run.workers - 1,
                "minReplicas": run.workers - 1,
                "maxReplicas": run.workers - 1,
                "rayStartParams": {},
                "template": gpu_pod_template(run, pool, is_head=False),
            }
        ]

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
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
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 600,  # a finished RayJob keeps its GPU pods until TTL
            # runs on the head via the submitter; driver.log tee lives inside.
            # Absolute path: the ray-job driver's cwd is Ray's session dir,
            # not the image workdir.
            "entrypoint": f"bash /opt/skyrl/integrations/fleet/rl1/entrypoint.sh {shlex.quote(run.command)}",
            "submitterPodTemplate": {
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"kubernetes.io/os": "linux", "workload": "infra"},
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "infra",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "ray-job-submitter",
                            # slim on purpose: it only runs `ray job submit`;
                            # the trainer image would fill infra-node disks
                            "image": f"anyscale/ray:{RAY_VERSION}-slim-py312",
                            "env": [
                                # GPU-less node + CUDA image: without this the
                                # NVIDIA hook tries to init a missing driver
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "200m",
                                    "memory": "512Mi",
                                    "ephemeral-storage": "2Gi",
                                },
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                        }
                    ],
                }
            },
            "rayClusterSpec": cluster_spec,
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
    print(f"status: kubectl --context {KUBE_CONTEXT} -n {NAMESPACE} get rayjob {run.name}")
    print(f"logs:   kubectl --context {KUBE_CONTEXT} -n {NAMESPACE} logs -f job/{run.name}")
    print(f"driver: {SFS_ROOT}/{run.name}/driver.log")


if __name__ == "__main__":
    main()
