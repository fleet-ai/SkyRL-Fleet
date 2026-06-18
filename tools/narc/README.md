# narc

`narc` runs deterministic accelerator correctness and performance checks for
compute clusters.

The first implementation is deliberately small:

- `narc run-local` runs a deterministic PyTorch probe on one assigned device.
- `narc aggregate` summarizes per-device JSON results.

Run from this directory with uv:

```bash
uv run narc run-local --device cpu --out-dir /tmp/narc
uv run narc aggregate /tmp/narc
```

On a Slurm node with one process per GPU:

```bash
srun --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=single:1 \
  uv run --project tools/narc narc run-local --out-dir /workspace/narc/$SLURM_JOB_ID
```
