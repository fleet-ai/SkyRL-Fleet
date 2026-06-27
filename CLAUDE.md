# SkyRL-v2 (fleet-ai/SkyRL-v2)

Fork of SkyRL with Fleet-specific optimizations for multi-node FSDP2 training at scale.

SkyRL is a full-stack reinforcement learning library for training LLMs, designed for modularity and extensibility.

## Critical Rules

- **Always use `uv run --isolated`** to run commands. Never use bare `python`, `pip`, or `pip install`.
- **Log output to files**: `<cmd> > /tmp/results_1.log 2>&1` for persistence.
- Backend extras (`fsdp`, `megatron`, `jax`) conflict with each other -- never combine them.
- Always read the relevant documentation files in `.claude/docs` before troubleshooting or working on any changes. Follow the routing rules below.

## Test Commands

```bash
# CPU tests
uv run --extra dev --extra jax pytest tests/tx/ tests/tinker/ tests/utils/
uv run --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu/

# GPU tests (requires Ray cluster with GPUs)
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py

# Lint / format
bash format.sh
```

## Training Quick Start

```bash
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron trainer.policy.model.path=<model> environment.env_class=gsm8k ...
```

## Routing Rules

When working on these areas, read the corresponding doc first:

| Area | Read first |
|------|-----------|
| Package management, uv, formatting | `.claude/docs/development.md` |
| Overall guide for modifying or working with SkyRL | `.claude/docs/contributing.md` |
| Tests, fixtures, CI quirks | `.claude/docs/testing.md` |
| Project layout, Ray actors, config | `.claude/docs/architecture.md` |
| Training entrypoints, configs | `.claude/docs/training.md` |
| Inference engines, vLLM, PD disagg | `.claude/docs/inference.md` |
| GitHub Actions, Anyscale CI | `.claude/docs/ci.md` |
| Tinker API server | `.claude/docs/tinker.md` |
| Megatron backend | `.claude/docs/backends/megatron.md` |
| FSDP backend | `.claude/docs/backends/fsdp.md` |
| JAX/TPU backend | `.claude/docs/backends/jax.md` |
| Weight sync | `.claude/docs/weight_sync.md` |


## Troubleshooting

For troubleshooting training runs with SkyRL:

1. Go through the troubleshooting section in the docs for known errors: `docs/content/docs/troubleshooting/troubleshooting.mdx`
2. Go through the contributing guide for overall guidelines: `.claude/docs/contributing.md`

## Fleet Integration

Fleet-specific changes, fixes, and context are documented in:
- **[integrations/fleet/CHANGELOG.md](integrations/fleet/CHANGELOG.md)** — detailed changelog with root causes and fixes

Always consult the changelog before modifying Fleet training paths (`fsdp_worker.py`, `worker.py`, `model_wrapper.py`, `dispatch.py`, `fleet-*.sh`).

## Key Differences from Upstream SkyRL

1. **Multi-node FSDP2 stability**: Synchronous ref model offload/backload with `torch.distributed.barrier()` in `fsdp_worker.py`. Required because cross-node colocated training has no shared CUDA context.

2. **Chunked lm_head forward**: `model_wrapper.py` has `loss_chunk_size` support ported from the old fork. Avoids materializing full `(B, S, vocab_size)` logits — critical for 35B with 131K vocab at 97K sequence length. Without it, OOM/Xid 31 during training forward.

3. **CUDA memory management for 35B**: `torch.cuda.empty_cache()` before backward pass in `worker.py` (policy + critic). Prevents OOM from fragmentation.

4. **Reduced sequence length (72K) for 35B**: `fleet-35b-run.sh` uses `MAX_INPUT_LENGTH=72000` (down from 96000) with `--no-pytorch-alloc-conf` (disables `expandable_segments` which conflicts with vLLM 0.18.0's `CuMemAllocator`). At 97K, SDPA OOM'd and flash_attn hit Xid 31 in GatedDeltaNet. At 72K, flash_attn=true + chunked lm_head + empty_cache fits without expandable_segments.

5. **`stage_chunks` pre-staging**: `dispatch.py` has a `stage_chunks` optimization (not in upstream) that pre-stages mini-batch chunks in Ray object store. Includes dynamic `mini_batch_size` adjustment for hint augmentation's variable batch sizes.

## Training Scripts

- `scripts/fleet-common-run.sh` — shared infra (Ray, NCCL, gIB detection, deps). Used by all runs.
- `scripts/fleet-35b-run.sh` — Qwen3.5-35B config. Calls `fleet-common-run.sh`.
- `scripts/fleet-9b-run.sh` — Qwen3.5-9B config. Calls `fleet-common-run.sh`.

All training flags live in these scripts. Never duplicate flags in SkyPilot YAMLs or fleet-research scripts.

## Launching Jobs

Always launch via `scripts/fleet-launch.sh`, not `sky launch` directly. It runs
`scripts/fleet-preflight.sh` first to validate env vars (FLEET / WANDB / AWS),
gcloud auth state, and `sky check` *before* provisioning a VM — so misconfigured
launches fail in seconds locally instead of after setup on a remote node.

```
bash scripts/fleet-launch.sh tasks/openenv-fleet-grpo-vl.yaml \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
```

Pass extra preflight requirements (e.g. for task-gen) before a literal `--`:

```
bash scripts/fleet-launch.sh --require OPENROUTER_API_KEY -- \
  tasks/task-gen-grpo-qwen3_5-9b.yaml --env ...
```

Tokens before `--` go to `fleet-preflight.sh`; tokens after go to `sky launch`.
Without `--`, every token is forwarded to `sky launch`.

## Task-Gen Metrics

When reporting task-gen training metrics, distinguish between:
- **pass@8 / avg_raw_reward**: includes `base_quality=0.1` for passing sandbox+judge. Misleading — inflated by gate-passing alone.
- **binary variance reward**: the actual learning signal. `1.0` when solver rollouts are mixed (at least 1 pass + 1 fail), `0.0` otherwise. This is what matters.

Report binary variance reward count (how many tasks got `reward >= 1.0`) separately from gate-pass count. Check `EVAL` log lines for `total=1.0000` vs `total=0.0000`.

## Colocated Training Memory Model

SkyRL uses **colocated training**: vLLM inference engines and FSDP training share the same GPUs but **alternate, never run simultaneously**. The cycle is:

1. vLLM wakes: loads model weights + allocates KV cache
2. Generate trajectories
3. vLLM sleeps: frees KV cache + offloads weights
4. FSDP wakes: loads sharded weights + gradients + runs forward/backward
5. FSDP sleeps
6. Repeat

`generator.gpu_memory_utilization` controls how much GPU memory vLLM claims during **its phase only**. It does NOT reduce memory available for FSDP. Setting it low "to leave room for training" wastes KV cache capacity without helping training.

When sizing `gpu_memory_utilization`: consider model weights under TP sharding + desired KV cache for your context length. FSDP memory budget is a separate, independent calculation. Typical values: 0.65-0.80.

## Branch

Primary development branch: `main`
