# narc

`narc` runs deterministic accelerator correctness and performance checks for
compute clusters.

The first implementation is deliberately small:

- `narc run` runs a deterministic PyTorch probe on one assigned device.
- `narc aggregate` summarizes per-device JSON results.
- `narc compare` partitions result JSON files into equivalence classes.

Run from this directory with uv:

```bash
uv run narc run --device cpu --out-dir /tmp/narc
uv run narc run --device cuda --out-dir s3://fleet-research/path/to/narc-results/
uv run narc aggregate /tmp/narc
uv run narc compare /tmp/narc/*.json
uv run narc aggregate s3://fleet-research/path/to/narc-results/
uv run narc compare s3://fleet-research/path/to/narc-results/
uv run narc compare s3://fleet-research/path/to/narc-results/ \
  -o s3://fleet-research/path/to/compare.json
```

`narc run` uses deterministic randomized token IDs and labels. `--seed`
controls model initialization and torch setup; `--input-seed` controls the input
token IDs and labels. Both are serialized in `probe_config`. Each result also
stores `narc_data_version` and `checks.input_hash`, so result files record both
the input-generation version and the generated input tensor hash.

`narc compare` exits non-zero when valid inputs split into more than one
equivalence class, contain non-pass probe results, or have pass results without
an output hash. Use `--nofail` to still emit that valid non-passing comparison as
JSON while exiting zero. Load errors, schema errors, and empty comparisons remain
non-zero.

On a Slurm node with one process per GPU:

```bash
srun --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=single:1 \
  uv run --project tools/narc narc run \
    --device cuda \
    --out-dir s3://fleet-research/narc/$SLURM_JOB_ID
```

Launch the included SkyPilot Slurm task only after installing the research-jobs
AWS credentials on the cluster as an AWS shared-credentials file:

```bash
sky launch tasks/narc-slurm.yaml
```
