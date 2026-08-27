# SkyRL-Fleet jobs on rl1 — runbook

This directory lets you run SkyRL-Fleet training on the rl1 cluster (`fleet-training-rl1-us-east-1`) by submitting a job. A job is five things: a name, a Docker image, a command, a GPU count, and env vars (plus the names of the secrets it needs). The image contains everything the training needs. The submitter handles the cluster side: queueing, node placement, log capture, and where checkpoints go.

The sections below are in the order you would do them: design, payload, one-time setup, build, submit, monitor.

## 1. How it works

There is no jobs API yet. `submit_run.py` does by hand what the API will do later: it turns your payload into one Kubernetes manifest and applies it. Everything after that step is already the real production path.

```
 researcher
    │  POST /v1/runs  {name, image, command, workers, gpus_per_worker, env, secrets}
    ▼
┌─────────────────────────────────────────────────────────────┐
│  runs API  (today: submit_run.py, run by hand)              │
│  1. validate the payload, check the secrets exist           │
│  2. render ONE RayJob manifest:                             │
│       kueue.x-k8s.io/queue-name: training-lq   ← the tag    │
│       suspend: true                                         │
│       topology annotation on the GPU pods                   │
│  3. kubectl apply  (this IS the submission)                 │
│  4. GET /runs/{id} = read the RayJob status back            │
└──────────────┬──────────────────────────────────────────────┘
               │ apply
               ▼
┌────────────────────────── kubernetes cluster ────────────────────────┐
│   RayJob object (suspended: it exists but starts nothing)            │
│        │                                                             │
│   ┌────▼─────────────────────────┐                                   │
│   │ KUEUE (the cluster's queue)  │ sees the tag, puts the job in     │
│   │                              │ line, and waits until ALL the     │
│   │                              │ GPUs the job needs are free at    │
│   │                              │ the same time on nearby machines. │
│   │                              │ Then it lifts the suspend.        │
│   └────┬─────────────────────────┘                                   │
│   ┌────▼─────────────────────────┐                                   │
│   │ KUBERAY (pod builder)        │ now creates the actual pods,      │
│   │                              │ starts Ray in them, and a small   │
│   │                              │ helper pod sends your command     │
│   │                              │ to the head pod                   │
│   └────┬─────────────────────────┘                                   │
│   ┌────▼────────────────────┐  ┌──────────────────────┐              │
│   │ head pod (8 GPUs)       │◄─┤ more GPU pods ...    │ workers > 1  │
│   │ your image, your        │  └──────────────────────┘              │
│   │ command                 │                                        │
│   └───────┬─────────────────┘                                        │
│           │ /mnt/sfs is a shared disk: logs and checkpoints          │
│           │ written there survive after the pods are gone            │
└───────────┼───────────────────────────────────────────────────────────┘
            ▼
   checkpoints → shared disk + S3     traces → ATOF / Fleet     metrics → W&B
```

Why the queue matters: the cluster has more demand than GPUs. A job that skips the queue can be killed mid-run when a queued job is granted the same GPUs (this happened on rl1 and was measured). The queue tag is how your job's GPUs get reserved for you.

To cancel a run: `kubectl delete rayjob <name>`. To see its state: `kubectl get rayjob <name>`.

## 2. The run payload

A run is a JSON file:

```json
{
  "name": "skyrl-tu-qwen35-9b-01",
  "image": "ghcr.io/fleet-ai/skyrl-fleet/trainer:latest",
  "command": "bash scripts/fleet-9b-run.sh",
  "workers": 1,
  "gpus_per_worker": 8,
  "env": {"MODALITY": "tool_use", "DATA_VERSION": "v7"},
  "secrets": ["fleet-api", "wandb-api", "aws-api"]
}
```

- `command` runs inside the image from the repo root (`/opt/skyrl`), after the startup script has downloaded the dataset from S3 and built the train files. Anything after the script name is passed through to the trainer as config overrides.
- `workers` is how many 8-GPU machines the run uses. Raise it only when the model does not fit or train fast enough on one machine.
- `secrets` are names of Kubernetes secrets; every key in them becomes an env var in the container. The three above provide `FLEET_API_KEY`, `WANDB_API_KEY`, and the AWS keys.

Two ready payloads are in `examples/`:

- `tool-use-qwen35-9b.json`: Qwen3.5-9B GRPO on the v7 tool_use dataset (`scripts/fleet-9b-run.sh`).
- `browser-use-qwen35-9b.json`: Qwen3.5-9B GRPO on the v7 browser_use dataset with screenshots (`scripts/fleet-vl-run.sh`).

Both are cut down (`MAX_TASKS`, one epoch) so a test run produces a training step and a checkpoint in about an hour. Remove those env vars for a full run. Qwen3.5-35B is not offered here: it needs 16 GPUs and the B200 node has 8, and 35B crashed on B200 before (Xid 31 GPU fault in the reference-model forward pass). 35B stays on its existing 2-machine H200 SkyPilot setup.

What your code can rely on:

- **Logs**: everything the run prints is saved to `/mnt/sfs/skyrl-fleet/<name>/driver.log` and survives after the pods are deleted.
- **Checkpoints**: write them under `/mnt/sfs/skyrl-fleet/<name>/` (the examples pass `trainer.ckpt_path=...` for this). The pod's own disk is wiped when the job ends; only `/mnt/sfs` survives. S3 checkpoint upload works the same as on SkyPilot (`S3_CHECKPOINT_BUCKET`).
- **Traces**: every rollout is uploaded as a Fleet trace; training batches are dumped as JSONL (`trainer.dump_data_batch=true`); ATOF events are emitted if the nemo-relay install at startup succeeds (if it fails, the run continues without it).
- **Metrics**: W&B, entity `thefleet`, project `fleet-tool-use-grpo` or `fleet-browser-use-grpo`.

## 3. One-time setup

You need kubeconfig access to `fleet-training-rl1-us-east-1`. For image builds you also need a GitHub token that can push packages (`write:packages`).

The secrets the examples use (`fleet-api`, `wandb-api`, `ghcr-pull`, `img-build-secrets`) already exist on rl1. `aws-api` was created 2026-08-26; on a fresh cluster, recreate it:

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs \
  create secret generic aws-api \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...
```

## 4. Build the image

```bash
export GH_TOKEN=<token with write:packages>
bash integrations/fleet/rl1/build_image.sh main
# -> ghcr.io/fleet-ai/skyrl-fleet/trainer:<8-char commit sha> and :latest
```

The build runs on the cluster's builder machine and clones the branch from GitHub, so uncommitted local changes never end up in an image. The image contains everything the setup scripts would install on a fresh VM: the locked Python dependencies, transformers 5.3.0, the flash-attn wheel, causal-conv1d compiled for both our GPU generations (sm_90 and sm_100, so one image runs on either pool), OpenEnv pinned to a specific commit, the CUDA 12.8 compiler (Qwen3.5 compiles some GPU kernels at training time), and `ray[default]` (KubeRay's health checks need it; without it jobs cannot start). The build fails unless the training code imports cleanly.

## 5. Submit a run

```bash
cd integrations/fleet/rl1
./submit_run.py examples/tool-use-qwen35-9b.json
./submit_run.py examples/browser-use-qwen35-9b.json
./submit_run.py my-run.json --dry-run     # print the manifest without applying it
```

The submitter always produces a RayJob, whatever the `workers` count. The manifest carries the three things every GPU job on this cluster must have: the queue tag, `suspend: true`, and the topology annotation (it tells the queue the pods must land on machines wired close together, so GPU-to-GPU traffic is fast). The head pod gets the first 8 GPUs and runs the training driver; `workers > 1` adds more 8-GPU pods, and the queue only starts the job when all of them can start together. Inside the pods, `FLEET_EXTERNAL_RAY=1` tells `fleet-common-run.sh` that Ray is already running and it should connect to it instead of starting its own.

Jobs land on the B200 node by default (`--pool` switches the node group if that ever changes).

## 6. Monitor

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get workloads   # the queue; ADMITTED=True means running
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get rayjobs
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>
tail -f /mnt/sfs/skyrl-fleet/<name>/driver.log   # from any pod that mounts the shared disk
```

In the first 15 minutes, confirm four things: the pods are running on the intended node, the dataset downloaded and the task count printed, the vLLM engines started, and the first rollout is making progress. Checkpoints appear under `/mnt/sfs/skyrl-fleet/<name>/ckpts/` every `trainer.ckpt_interval` steps and at the end of the run.
