# SkyRL Trace Viewer

A zero-dependency, browser-based viewer for **training trajectories**, built to answer
one question: **how do a run's traces evolve over training steps?**

It reads the per-step JSONL files that
[`dump_training_trajectories`](../../skyrl/train/utils/trainer_utils.py) writes to
`{export_path}/dumped_trajectories/global_step_{N}.jsonl` — one file per step, one
trajectory per line. The format is generic, so this works for **any** environment
(negotiation, task-gen, vision agents, …), not just one.

![evolution + step + track views] <!-- screenshots optional -->

## What it shows

- **Evolution over steps** — mean reward, mean turns/tokens, and the stop-reason mix
  charted across every step in the run. Click a point to jump to that step.
- **This step** — every trajectory at the selected step, with reward/turns/tokens/
  stop-reason badges, filters (env, stop reason, min reward, full-text search), and an
  expandable, syntax-highlighted conversation view (`<think>` blocks and action tags
  like `<propose>`/`<accept>` are highlighted) plus a raw prompt+response view.
- **Track one prompt** — pick a prompt that recurs across steps (as with a fixed
  train/val set) and see the model's responses to that *same* prompt side-by-side at
  each step. This is the most direct way to watch behavior change over training.

## Quick start

```bash
cd tools/trace-viewer
./serve.sh                 # generates sample data on first run, then serves
# open the printed URL, e.g. http://localhost:8792
```

The first run synthesizes a demo negotiation run so you immediately have something to
click around in.

## Viewing your own run

Three options, easiest first:

**1. Drag & drop (no server).** Open `public/index.html` in a browser and drag your
`global_step_*.jsonl` files (or use *Load JSONL files* / *Load folder*) onto the page.
Everything runs locally; nothing is uploaded.

**2. Import + serve.** Copy a run's trajectories into the viewer and serve it:

```bash
python3 build_manifest.py --import ~/exports/dumped_trajectories --name my-run
./serve.sh
```

Use `--link` to symlink instead of copy. Re-run `build_manifest.py` any time to
re-index everything under `public/data/`.

**3. Point at an existing `public/data/<run>/` dir.** Drop step files under
`public/data/<run>/` yourself, then `python3 build_manifest.py`.

### Producing trajectories from a run

Trajectory dumping is gated by a trainer flag (off by default):

```
trainer.dump_training_trajectories=true
trainer.export_path=$HOME/exports          # files land in $export_path/dumped_trajectories/
```

## Trajectory schema

Each line is one trajectory (extra keys are ignored, missing keys degrade gracefully):

| field | meaning |
|-------|---------|
| `step` | training step (also parsed from the filename) |
| `env_key` | environment id (`negotiation`, `task_gen`, …) |
| `data_source` | dataset tag |
| `reward` | scalar reward for the trajectory |
| `turns` | number of environment turns |
| `tokens` | response length in tokens |
| `stop_reason` | why generation stopped (`stop`, `length`, …) |
| `prompt` | decoded prompt (chat-template text) |
| `text` | decoded response (chat-template text) |
| `timestamp` | unix time the step was dumped |

## Files

- `public/` — the static app (`index.html`, `app.js`, `styles.css`). Chart.js is loaded
  from a CDN; everything else is vanilla JS.
- `build_manifest.py` — index/import runs into `public/data/manifest.json`.
- `gen_sample_data.py` — synthesize a demo run for testing.
- `serve.sh` — build sample data if needed and serve `public/`.
