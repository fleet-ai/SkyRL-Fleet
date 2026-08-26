# SkyRL-Fleet jobs on rl1

Run SkyRL-Fleet training on the rl1 cluster (`fleet-training-rl1-us-east-1`) as submitted jobs: an image, a command, a GPU count, env vars, and secret names. The image does everything (all dependencies baked, dataset pulled at start); the cluster side (queueing, placement, log capture, checkpoint location) is handled by the submitter.

## The run payload

A run is a JSON payload in the jobs-API shape:

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

`secrets` are names of Kubernetes secrets in `fleet-train-jobs`; each is injected into the container env wholesale. The three above provide `FLEET_API_KEY`, `WANDB_API_KEY`, and `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.

Two ready payloads are in `examples/`:

- `tool-use-qwen35-9b.json`: Qwen3.5-9B GRPO on the v7 tool_use taskset (`scripts/fleet-9b-run.sh`).
- `browser-use-qwen35-9b.json`: Qwen3.5-9B GRPO on the v7 browser_use taskset with screenshots (`scripts/fleet-vl-run.sh`).

Both cap the dataset (`MAX_TASKS`) and run one epoch so a validation run finishes in hours. Drop those two env vars for a full run. Qwen3.5-35B is not offered here: the single B200 node has 8 GPUs and the 35B recipe needs 16, and 35B previously crashed on B200 (Xid 31 in the FSDP2 ref-model forward). 35B stays on the 2-node H200 SkyPilot path.

## Contract with the user's code

- **Run entry**: the `command` string runs from the repo root (`/opt/skyrl`) inside the image, after `entrypoint.sh` has downloaded `s3://fleet-internal-datasets/$DATA_VERSION/openenv/all_$MODALITY.json` and built the train/validation parquet files. Any trailing tokens are Hydra overrides passed through to the trainer.
- **Trace**: rollout events go to ATOF via nemo-relay (installed at container start, fail-open); training trajectories are dumped by `trainer.dump_data_batch=true`.
- **Checkpoints**: pass `trainer.ckpt_path=/mnt/sfs/skyrl-fleet/<name>/ckpts` in the command (the examples do). The pod filesystem is ephemeral; only `/mnt/sfs` survives the job. S3 checkpoint upload works as on SkyPilot (`S3_CHECKPOINT_BUCKET`).
- **Observability**: W&B (entity `thefleet`, project `fleet-tool-use-grpo` or `fleet-browser-use-grpo`), plus a full driver log at `/mnt/sfs/skyrl-fleet/<name>/driver.log` that survives pod deletion.

## Build the image

```bash
export GH_TOKEN=<PAT with write:packages>
bash integrations/fleet/rl1/build_image.sh main
# -> ghcr.io/fleet-ai/skyrl-fleet/trainer:<8-char commit sha> and :latest
```

The build runs on the cluster's builder node and clones the ref from GitHub, so uncommitted local changes never reach an image. The image bakes everything `scripts/fleet-common-setup.sh` and `scripts/fleet-qwen35-extra-setup.sh` install on a fresh VM: the locked dependency set (`uv sync --frozen --extra fsdp`), transformers 5.3.0, the flash-attn 2.8.3 wheel, causal-conv1d compiled for sm_90 and sm_100 (H200 and B200), OpenEnv, and the CUDA 12.8 toolkit for the GDN JIT kernels. The build fails unless the training entrypoint imports.

## Submit a run

```bash
cd integrations/fleet/rl1
./submit_run.py examples/tool-use-qwen35-9b.json                  # B200 pool (default)
./submit_run.py examples/browser-use-qwen35-9b.json --pool gpu-h200
./submit_run.py my-run.json --dry-run                             # print manifest only
```

The submitter renders one RayJob for any `workers` count, with the three things every GPU job on rl1 must carry: the `training-lq` queue label, `suspend: true`, and the tier-1 topology annotation on the GPU pod sets. Kueue admits the whole pod set as a gang when the pool has capacity; until then it waits suspended. The head group carries the first `gpus_per_worker` GPUs and runs the driver (the same layout as the SkyPilot path, where the driver lives on a GPU node); `workers > 1` adds a GPU worker group with `workers - 1` replicas. Inside the pods, `FLEET_EXTERNAL_RAY=1` tells `fleet-common-run.sh` to attach to the Ray cluster KubeRay booted instead of starting its own.

## System design (this directory stands in lieu of the runs API)

`submit_run.py` implements boxes 2 and 3 of the future `POST /v1/runs` API: render one RayJob manifest from the payload, then `kubectl apply`. Everything below the API box already runs in production form.

```
 researcher
    │  POST /v1/runs  {name, image, command, workers, gpus_per_worker, env, secrets}
    ▼
┌─────────────────────────────────────────────────────────────┐
│  runs API  (today: submit_run.py, by hand)                  │
│  1. validate payload, resolve secrets                       │
│  2. render ONE RayJob manifest:                             │
│       kueue.x-k8s.io/queue-name: training-lq   ← the tag    │
│       suspend: true                                         │
│       topology annotation on GPU pod sets                   │
│  3. kubectl apply  (this IS the submission)                 │
│  4. GET /runs/{id} = read the RayJob status back            │
└──────────────┬──────────────────────────────────────────────┘
               │ apply
               ▼
┌────────────────────────── kubernetes cluster ────────────────────────┐
│   RayJob object (suspended, does nothing yet)                        │
│        │                                                             │
│   ┌────▼─────────────────────────┐                                   │
│   │ KUEUE (admission)            │ sees the queue tag, records a     │
│   │                              │ Workload, waits until             │
│   │                              │ workers × gpus_per_worker GPUs    │
│   │                              │ are free at once in one tier-1    │
│   │                              │ domain, then flips suspend off    │
│   └────┬─────────────────────────┘                                   │
│   ┌────▼─────────────────────────┐                                   │
│   │ KUBERAY (RayJob controller)  │ creates the pods: GPU head        │
│   │                              │ (+ GPU workers if workers > 1),   │
│   │                              │ boots Ray, submitter pod posts    │
│   │                              │ `command` to the head             │
│   └────┬─────────────────────────┘                                   │
│   ┌────▼────────────────────┐  ┌──────────────────────┐              │
│   │ GPU head pod (8 GPU)    │◄─┤ GPU worker pods ...  │ workers > 1  │
│   │ your image, entrypoint, │  └──────────────────────┘              │
│   │ your command            │                                        │
│   └───────┬─────────────────┘                                        │
│           │ mounts /mnt/sfs (driver.log, ckpts survive the pods)     │
└───────────┼───────────────────────────────────────────────────────────┘
            ▼
   checkpoints → SFS + S3     traces → ATOF / Fleet     metrics → W&B
```

Division of labor, one line each: the API owns the payload contract and renders it; the RayJob object is a passive record of what to run; Kueue is the bouncer that reserves all the GPUs at once; KubeRay is the builder that creates pods and runs the command; the pods are your image on the reserved GPUs. Cancel = `kubectl delete rayjob <name>`; status = the RayJob's own status fields.

One-time secret setup (already present on rl1 except `aws-api`):

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs \
  create secret generic aws-api \
  --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...
```

## Monitor

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get jobs,pods -l app=skyrl-fleet
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>
tail -f /mnt/sfs/skyrl-fleet/<name>/driver.log   # from any pod with the SFS mount
```

In the first 15 minutes confirm: the pod is Running on the intended pool, the dataset downloaded and prepared (task counts in the log), the vLLM engines came up, and the first rollout is progressing. Checkpoints appear under `/mnt/sfs/skyrl-fleet/<name>/ckpts/` at `trainer.ckpt_interval` steps.
