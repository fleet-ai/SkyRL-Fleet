#!/usr/bin/env python3
"""Plot Ticketmaster pass-rate results for the taste-reward work trial."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 360,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.32,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_1.png")
    args = parser.parse_args()

    data = pd.DataFrame(
        [
            {"Method": "Qwen 9B\nbaseline", "Pass rate": 0.08},
            {"Method": "Verifier-only\nGRPO", "Pass rate": 0.12},
            {"Method": "Verifier + taste\nGRPO", "Pass rate": 0.15},
        ]
    )

    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    colors = ["#8D99AE", "#457B9D", "#2A9D8F"]
    sns.barplot(data=data, x="Method", y="Pass rate", hue="Method", palette=colors, legend=False, ax=ax)

    ax.set_ylim(0, 0.18)
    ax.set_xlabel("")
    ax.set_ylabel("Verifier success rate")
    ax.set_title("Ticketmaster Pass Rate", fontweight="bold")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", visible=False)

    for patch, value in zip(ax.patches, data["Pass rate"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.006,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # ax.text(
    #     2,
    #     0.025,
    #     "taste success uses\nreward >= 0.5",
    #     ha="center",
    #     va="center",
    #     fontsize=8.5,
    #     color="0.32",
    # )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
