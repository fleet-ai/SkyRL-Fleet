#!/usr/bin/env python3
"""Visualize how the per-episode *number of turns* distribution evolves over a
negotiation training run.

The trace dumps written by ``dump_training_trajectories`` (``global_step_*.jsonl``)
contain one row per rollout episode with a ``turns`` field (count of policy
messages in that episode). This script aggregates those per-step into a turn-count
distribution and renders, for each run:

  * a heatmap  (x = training step, y = #turns, color = fraction of episodes), with
    the per-step mean turns overlaid, and
  * a shared line panel comparing mean turns across runs.

Usage
-----
    # point at directories of global_step_*.jsonl files (one per run)
    python turns_distribution.py \
        --run rawbase0616=/tmp/turns_data/rawbase0616 \
        --run nothink0616=/tmp/turns_data/nothink0616 \
        --out turns_distribution.png

Each ``--run`` is ``LABEL=DIR``. ``DIR`` holds ``global_step_*.jsonl`` files.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

STEP_RE = re.compile(r"global_step_(\d+)\.jsonl$")


def load_run(run_dir: Path, data_source: str | None):
    """Return (steps, counts) where counts[i] is a dict {turn: n_episodes} for steps[i]."""
    files = []
    for p in run_dir.glob("global_step_*.jsonl"):
        m = STEP_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    steps, per_step = [], []
    for step, path in files:
        counts: dict[int, int] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data_source and row.get("data_source") != data_source:
                    continue
                t = row.get("turns", row.get("num_turns"))
                if t is None:
                    continue
                t = int(round(float(t)))
                counts[t] = counts.get(t, 0) + 1
        if counts:
            steps.append(step)
            per_step.append(counts)
    return steps, per_step


def build_matrix(steps, per_step, max_turn):
    """Return (frac, mean) — frac is (max_turn, n_steps) fraction grid; mean per step."""
    n = len(steps)
    frac = np.zeros((max_turn, n))
    mean = np.zeros(n)
    for j, counts in enumerate(per_step):
        total = sum(counts.values())
        s = 0.0
        for t, c in counts.items():
            if 1 <= t <= max_turn:
                frac[t - 1, j] = c / total
            s += t * c
        mean[j] = s / total if total else np.nan
    return frac, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="run to plot, e.g. rawbase0616=/tmp/turns_data/rawbase0616 (repeatable)")
    ap.add_argument("--data-source", default="negotiation_dnd",
                    help="filter episodes by data_source (default: negotiation_dnd; '' = all)")
    ap.add_argument("--max-turn", type=int, default=None,
                    help="max turn count on the y axis (default: inferred from data)")
    ap.add_argument("--out", default="turns_distribution.png", help="output image path")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=DIR, got: {spec}")
        label, d = spec.split("=", 1)
        steps, per_step = load_run(Path(d).expanduser(), args.data_source or None)
        if not steps:
            raise SystemExit(f"no usable trajectory data found in {d}")
        runs.append((label, steps, per_step))

    if args.max_turn is None:
        max_turn = max(t for _, _, ps in runs for c in ps for t in c)
    else:
        max_turn = args.max_turn

    n_runs = len(runs)
    fig = plt.figure(figsize=(11, 3.4 * n_runs + 3.2), constrained_layout=True)
    gs = fig.add_gridspec(n_runs + 1, 1, height_ratios=[1] * n_runs + [0.85])

    cmap = LinearSegmentedColormap.from_list(
        "turns", ["#f7fbff", "#9ecae1", "#3182bd", "#08306b"]
    )
    line_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    means = []
    for i, (label, steps, per_step) in enumerate(runs):
        frac, mean = build_matrix(steps, per_step, max_turn)
        means.append((label, steps, mean))
        ax = fig.add_subplot(gs[i, 0])
        x = np.asarray(steps)
        # heatmap: use pcolormesh so non-contiguous steps render at their true x.
        # Build edges around integer turn values and actual steps.
        xedges = np.concatenate([[x[0] - 0.5], (x[:-1] + x[1:]) / 2, [x[-1] + 0.5]])
        yedges = np.arange(0.5, max_turn + 1.5)
        mesh = ax.pcolormesh(xedges, yedges, frac, cmap=cmap, vmin=0, vmax=min(1.0, frac.max() * 1.05))
        ax.plot(x, mean, color=line_colors[i % len(line_colors)], lw=2.2,
                marker="o", ms=3, label="mean turns")
        ax.set_yticks(range(1, max_turn + 1))
        ax.set_ylabel("# turns / episode")
        ax.set_title(f"{label} — turn-count distribution over training", fontsize=11, loc="left")
        ax.set_xlim(xedges[0], xedges[-1])
        cb = fig.colorbar(mesh, ax=ax, pad=0.01)
        cb.set_label("fraction of episodes")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        if i == n_runs - 1:
            ax.set_xlabel("training step")

    # comparison panel
    axc = fig.add_subplot(gs[n_runs, 0])
    for j, (label, steps, mean) in enumerate(means):
        axc.plot(steps, mean, color=line_colors[j % len(line_colors)], lw=2,
                 marker="o", ms=3, label=label)
    axc.set_xlabel("training step")
    axc.set_ylabel("mean # turns")
    axc.set_title("Mean turns per episode over training", fontsize=11, loc="left")
    axc.grid(alpha=0.3)
    axc.legend(loc="best", fontsize=9)

    src = args.data_source or "all sources"
    fig.suptitle(f"Negotiation: number-of-turns distribution over training  (data_source = {src})",
                 fontsize=13, fontweight="bold")
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # brief text summary
    for label, steps, mean in means:
        print(f"  {label}: steps {steps[0]}–{steps[-1]} "
              f"({len(steps)} dumps), mean turns {mean[0]:.2f} → {mean[-1]:.2f}")


if __name__ == "__main__":
    main()
