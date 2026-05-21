#!/usr/bin/env python3
"""Create publication-style taste/performance analysis figures.

Example:
  MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  .judge-venv/bin/python scripts/plot_analysis_figures.py \
    --input /Users/fleet-wt-6/Desktop/cua-taste/data/fleet-cu-claude-trajectories/taste_scores_all_200_rowkey.jsonl \
    --out-dir /Users/fleet-wt-6/Desktop/cua-taste/data/fleet-cu-claude-trajectories/taste_plots_all_200_rowkey
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "This script requires matplotlib and seaborn. Install into the active env with: "
        "python -m pip install matplotlib seaborn"
    ) from exc


CORR_METRICS = [
    ("D5_recovery", "Recovery"),
    ("D4_consistency", "Consistency"),
    ("D3_visual_grounding", "Visual grounding"),
    ("D2_redundancy", "Redundancy"),
    ("taste_rl_4", "Total taste score"),
]

DIST_METRICS = [
    ("D2_redundancy", "Redundancy"),
    ("D4_consistency", "Consistency"),
    ("D5_recovery", "Recovery"),
    ("D3_visual_grounding", "Visual grounding"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("task_key") or f"row_{i}")
            latest[key] = row
    return [row for row in latest.values() if "error" not in row]


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "patch.linewidth": 0.0,
        },
    )


def make_correlation_figure(rows: list[dict[str, Any]], out_path: Path) -> None:
    verifier = [as_float(row, "verifier_score") for row in rows]
    data = []
    for key, label in CORR_METRICS:
        corr = pearson([as_float(row, key) for row in rows], verifier)
        data.append({"Metric": label, "Pearson r": 0.0 if corr is None else corr})

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    palette = sns.color_palette("crest", n_colors=len(data))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Pearson r", hue="Metric", palette=palette, legend=False, ax=ax)
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_ylim(0, max(0.5, max(d["Pearson r"] for d in data) + 0.08))
    ax.set_xlabel("")
    ax.set_ylabel("Correlation with verifier")
    ax.set_title("Taste Dimensions vs Verifier Success", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for patch, row in zip(ax.patches, data):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_score_gap_figure(rows: list[dict[str, Any]], out_path: Path) -> None:
    data = []
    for key, label in CORR_METRICS:
        pass_vals = [as_float(row, key) for row in rows if as_float(row, "verifier_score") >= 1.0]
        fail_vals = [as_float(row, key) for row in rows if as_float(row, "verifier_score") < 1.0]
        gap = mean(pass_vals) - mean(fail_vals)
        data.append({"Metric": label, "Score gap": gap})

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    palette = sns.color_palette("crest", n_colors=len(data))
    sns.barplot(data=pd.DataFrame(data), x="Metric", y="Score gap", hue="Metric", palette=palette, legend=False, ax=ax)
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_ylim(0, max(1.45, max(d["Score gap"] for d in data) + 0.15))
    ax.set_xlabel("")
    ax.set_ylabel("Pass mean - fail mean")
    ax.set_title("Taste Score Gap by Verifier Outcome")
    ax.tick_params(axis="x", rotation=25)
    for patch, row in zip(ax.patches, data):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.04,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_distribution_figure(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.1), constrained_layout=True)
    axes_flat = axes.ravel()
    palette = {"Fail": "#E76F51", "Pass": "#2A9D8F"}
    for ax, (metric_key, label) in zip(axes_flat, DIST_METRICS):
        fail = [as_float(row, metric_key) for row in rows if as_float(row, "verifier_score") < 1.0]
        passed = [as_float(row, metric_key) for row in rows if as_float(row, "verifier_score") >= 1.0]
        pass_counts = {score: sum(1 for value in passed if round(value) == score) for score in range(1, 6)}
        fail_counts = {score: sum(1 for value in fail if round(value) == score) for score in range(1, 6)}
        for score in range(1, 6):
            pair = [
                ("Pass", pass_counts[score], palette["Pass"]),
                ("Fail", fail_counts[score], palette["Fail"]),
            ]
            pair.sort(key=lambda item: item[1], reverse=True)
            for outcome, count, color in pair:
                ax.bar(
                    score,
                    count,
                    width=0.82,
                    color=color,
                    alpha=0.77,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=1 if outcome == pair[0][0] else 2,
                )
        ax.set_title(label)
        ax.set_xlabel("Taste score")
        ax.set_ylabel("Trajectory count")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)
        ax.legend(
            handles=[
                Patch(facecolor=palette["Pass"], edgecolor="none", alpha=0.77, label=f"Pass (n={len(passed)})"),
                Patch(facecolor=palette["Fail"], edgecolor="none", alpha=0.77, label=f"Fail (n={len(fail)})"),
            ],
            frameon=False,
            loc="upper left",
        )
        sns.despine(ax=ax)

    # fig.suptitle("Taste Scores by Verifier Outcome", y=1.02, fontsize=13, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL from scripts/run_taste_judge_local.py")
    parser.add_argument("--out-dir", required=True, help="Directory for analysis_figure_1.png and analysis_figure_2.png")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit("No scored rows found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    make_correlation_figure(rows, out_dir / "analysis_figure_1.png")
    make_distribution_figure(rows, out_dir / "analysis_figure_2.png")
    make_score_gap_figure(rows, out_dir / "analysis_figure_3.png")
    print(f"Wrote {out_dir / 'analysis_figure_1.png'}")
    print(f"Wrote {out_dir / 'analysis_figure_2.png'}")
    print(f"Wrote {out_dir / 'analysis_figure_3.png'}")
    print(f"Scored rows used: {len(rows)}")


if __name__ == "__main__":
    main()
