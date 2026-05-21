#!/usr/bin/env python3
"""Plot how taste-judge scores relate to verifier performance.

Input is the JSONL produced by scripts/run_taste_judge_local.py or
scripts/judge_eval_outputs.py. Outputs are PNG figures plus CSV/Markdown
summaries in --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required. Install with: python -m pip install matplotlib") from exc


METRICS = [
    ("D1_efficiency", "Efficiency"),
    ("D2_redundancy", "Redundancy"),
    ("D3_visual_grounding", "Visual grounding"),
    ("D4_consistency", "Consistency"),
    ("D5_recovery", "Recovery"),
    ("taste_all_5", "Taste all 5"),
    ("taste_rl_4", "Taste RL 4"),
    ("rl_reward_formula", "RL reward formula"),
    ("estimated_actions", "Estimated actions"),
]

SCORE_METRICS = [
    ("D1_efficiency", "Efficiency"),
    ("D2_redundancy", "Redundancy"),
    ("D3_visual_grounding", "Visual grounding"),
    ("D4_consistency", "Consistency"),
    ("D5_recovery", "Recovery"),
]


def load_latest(path: Path) -> tuple[list[dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    n_errors = 0
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("task_key") or f"row_{i}")
            latest[key] = row
    rows = []
    for row in latest.values():
        if "error" in row:
            n_errors += 1
        else:
            rows.append(row)
    return rows, n_errors


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


def stderr_diff(pass_vals: list[float], fail_vals: list[float]) -> float | None:
    if len(pass_vals) < 2 or len(fail_vals) < 2:
        return None
    mp = mean(pass_vals)
    mf = mean(fail_vals)
    vp = sum((x - mp) ** 2 for x in pass_vals) / (len(pass_vals) - 1)
    vf = sum((x - mf) ** 2 for x in fail_vals) / (len(fail_vals) - 1)
    return math.sqrt(vp / len(pass_vals) + vf / len(fail_vals))


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y = [as_float(r, "verifier_score") for r in rows]
    out = []
    for key, label in METRICS:
        xs = [as_float(r, key) for r in rows]
        pass_vals = [x for x, yy in zip(xs, y) if yy >= 1.0]
        fail_vals = [x for x, yy in zip(xs, y) if yy < 1.0]
        pass_mean = mean(pass_vals) if pass_vals else None
        fail_mean = mean(fail_vals) if fail_vals else None
        diff = None if pass_mean is None or fail_mean is None else pass_mean - fail_mean
        se = stderr_diff(pass_vals, fail_vals)
        out.append(
            {
                "metric": key,
                "label": label,
                "n": len(xs),
                "n_pass": len(pass_vals),
                "n_fail": len(fail_vals),
                "overall_mean": mean(xs) if xs else None,
                "pass_mean": pass_mean,
                "fail_mean": fail_mean,
                "pass_minus_fail": diff,
                "diff_normal_se": se,
                "pearson_vs_verifier": pearson(xs, y),
            }
        )
    return out


def score_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, label in SCORE_METRICS:
        by_score: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            score = int(round(as_float(row, key)))
            by_score[score].append(as_float(row, "verifier_score"))
        for score in range(1, 6):
            vals = by_score.get(score, [])
            out.append(
                {
                    "metric": key,
                    "label": label,
                    "score": score,
                    "n": len(vals),
                    "pass_rate": mean(vals) if vals else None,
                }
            )
    return out


def taste_bins(rows: list[dict[str, Any]], metric: str, n_bins: int = 4) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: as_float(r, metric))
    if not ordered:
        return []
    bins = []
    for b in range(n_bins):
        lo = round(b * len(ordered) / n_bins)
        hi = round((b + 1) * len(ordered) / n_bins)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        vals = [as_float(r, "verifier_score") for r in chunk]
        xs = [as_float(r, metric) for r in chunk]
        bins.append(
            {
                "bin": b + 1,
                "n": len(chunk),
                "score_min": min(xs),
                "score_max": max(xs),
                "score_mean": mean(xs),
                "pass_rate": mean(vals),
            }
        )
    return bins


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, ndigits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


def plot_metric_bars(summary: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [r["label"] for r in summary]
    corrs = [r["pearson_vs_verifier"] or 0.0 for r in summary]
    diffs = [r["pass_minus_fail"] or 0.0 for r in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors_corr = ["#2f6f9f" if v >= 0 else "#b84a4a" for v in corrs]
    colors_diff = ["#2f6f9f" if v >= 0 else "#b84a4a" for v in diffs]

    axes[0].barh(labels, corrs, color=colors_corr)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Pearson correlation with verifier")
    axes[0].set_xlabel("correlation")

    axes[1].barh(labels, diffs, color=colors_diff)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Pass mean minus fail mean")
    axes[1].set_xlabel("score difference")

    fig.savefig(out_dir / "metric_associations.png", dpi=180)
    plt.close(fig)


def plot_score_success(score_rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, (key, label) in zip(axes_flat, SCORE_METRICS):
        rows = [r for r in score_rows if r["metric"] == key]
        scores = [r["score"] for r in rows]
        rates = [0.0 if r["pass_rate"] is None else r["pass_rate"] for r in rows]
        counts = [r["n"] for r in rows]
        ax.bar(scores, rates, color="#3b7f78")
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(label)
        ax.set_xlabel("taste score")
        ax.set_ylabel("pass rate")
        for x, y, n in zip(scores, rates, counts):
            if n:
                ax.text(x, min(y + 0.03, 0.95), f"n={n}", ha="center", va="bottom", fontsize=8)
    axes_flat[-1].axis("off")
    fig.savefig(out_dir / "success_rate_by_score.png", dpi=180)
    plt.close(fig)


def plot_pass_fail_distributions(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, (key, label) in zip(axes_flat, SCORE_METRICS):
        fail = [as_float(r, key) for r in rows if as_float(r, "verifier_score") < 1.0]
        passed = [as_float(r, key) for r in rows if as_float(r, "verifier_score") >= 1.0]
        ax.hist(fail, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.75, label=f"fail n={len(fail)}", color="#b84a4a")
        ax.hist(passed, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.75, label=f"pass n={len(passed)}", color="#2f6f9f")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(label)
        ax.set_xlabel("taste score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.savefig(out_dir / "pass_fail_score_distributions.png", dpi=180)
    plt.close(fig)


def plot_taste_quartiles(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    all_bins = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "taste_all_5", "All 5 taste dimensions"),
        (axes[1], "taste_rl_4", "RL taste dimensions"),
    ]:
        bins = taste_bins(rows, metric)
        all_bins.extend([{**b, "metric": metric} for b in bins])
        ax.bar([b["bin"] for b in bins], [b["pass_rate"] for b in bins], color="#6f6aa8")
        ax.set_ylim(0, 1)
        ax.set_xticks([b["bin"] for b in bins])
        ax.set_title(title)
        ax.set_xlabel("taste quartile")
        ax.set_ylabel("pass rate")
        for b in bins:
            ax.text(b["bin"], min(b["pass_rate"] + 0.03, 0.95), f"n={b['n']}", ha="center", fontsize=8)
    fig.savefig(out_dir / "pass_rate_by_taste_quartile.png", dpi=180)
    plt.close(fig)
    return all_bins


def plot_actions_vs_taste(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fail_x = [as_float(r, "estimated_actions") for r in rows if as_float(r, "verifier_score") < 1.0]
    fail_y = [as_float(r, "taste_rl_4") for r in rows if as_float(r, "verifier_score") < 1.0]
    pass_x = [as_float(r, "estimated_actions") for r in rows if as_float(r, "verifier_score") >= 1.0]
    pass_y = [as_float(r, "taste_rl_4") for r in rows if as_float(r, "verifier_score") >= 1.0]
    ax.scatter(fail_x, fail_y, alpha=0.65, color="#b84a4a", label=f"fail n={len(fail_x)}")
    ax.scatter(pass_x, pass_y, alpha=0.9, color="#2f6f9f", label=f"pass n={len(pass_x)}")
    ax.set_title("Taste vs action count")
    ax.set_xlabel("estimated actions")
    ax.set_ylabel("taste_rl_4")
    ax.legend()
    fig.savefig(out_dir / "taste_vs_actions.png", dpi=180)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]], n_errors: int, summary: list[dict[str, Any]], quartiles: list[dict[str, Any]]) -> None:
    n = len(rows)
    n_pass = sum(1 for r in rows if as_float(r, "verifier_score") >= 1.0)
    pass_rate = n_pass / n if n else 0.0
    by_abs_corr = sorted(summary, key=lambda r: abs(r["pearson_vs_verifier"] or 0.0), reverse=True)
    by_diff = sorted(summary, key=lambda r: abs(r["pass_minus_fail"] or 0.0), reverse=True)

    lines = [
        "# Taste vs Performance",
        "",
        f"- Scored rows: {n}",
        f"- Error rows excluded: {n_errors}",
        f"- Verifier passes: {n_pass}/{n} ({pass_rate:.3f})",
        "- Note: `RL reward formula` includes the verifier term, so its verifier correlation is expected and should not be interpreted as an independent taste signal.",
        "",
        "## Strongest Associations",
        "",
        "| metric | corr vs verifier | pass mean | fail mean | pass - fail |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in by_abs_corr:
        lines.append(
            f"| {r['label']} | {fmt(r['pearson_vs_verifier'])} | {fmt(r['pass_mean'])} | {fmt(r['fail_mean'])} | {fmt(r['pass_minus_fail'])} |"
        )

    lines.extend(["", "## Largest Mean Gaps", "", "| metric | pass - fail | corr vs verifier |", "|---|---:|---:|"])
    for r in by_diff:
        lines.append(f"| {r['label']} | {fmt(r['pass_minus_fail'])} | {fmt(r['pearson_vs_verifier'])} |")

    lines.extend(["", "## Taste Quartiles", "", "| metric | quartile | n | score range | pass rate |", "|---|---:|---:|---|---:|"])
    for b in quartiles:
        lines.append(
            f"| {b['metric']} | {b['bin']} | {b['n']} | {fmt(b['score_min'])}-{fmt(b['score_max'])} | {fmt(b['pass_rate'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="taste_scores.jsonl")
    parser.add_argument("--out-dir", help="Defaults to <input parent>/taste_plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / "taste_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, n_errors = load_latest(input_path)
    if not rows:
        raise SystemExit(f"No scored rows found in {input_path}")

    summary = metric_summary(rows)
    bins = score_bins(rows)
    quartiles = plot_taste_quartiles(rows, out_dir)
    write_csv(out_dir / "metric_summary.csv", summary)
    write_csv(out_dir / "success_rate_by_score.csv", bins)
    write_csv(out_dir / "pass_rate_by_taste_quartile.csv", quartiles)
    write_markdown(out_dir / "summary.md", rows, n_errors, summary, quartiles)

    plot_metric_bars(summary, out_dir)
    plot_score_success(bins, out_dir)
    plot_pass_fail_distributions(rows, out_dir)
    plot_actions_vs_taste(rows, out_dir)

    print(f"Wrote plots and summaries to {out_dir}")
    print(f"Open {out_dir / 'summary.md'} for the compact readout.")


if __name__ == "__main__":
    main()
