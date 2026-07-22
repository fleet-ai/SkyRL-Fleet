#!/usr/bin/env python3
"""Compare cross-play outcomes of the two candidate base checkpoints —
qwen3.5-35b-a3b (previous base) vs qwen3.6-27b — across the three elicitation
conditions (disclosure=can_ask, neutral=can_ask_modified, deception).

Metrics plotted:
  - self-score      : the policy's OWN normalized outcome, seat-agnostic, no-deal=0
  - joint efficiency : achieved/best joint score, conditioned on a deal (integrative pie capture)
  - nash product     : you_norm * them_norm, no-deal=0 (integrative gain incl. agreement)
Agreement rate is printed for context (nash/self already fold no-deal in as 0).

Reads the per-game sidecars so the policy's own score is correct in BOTH seats.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"

# (condition key, pretty label)
CONDS = [("canask", "Disclosure\n(can_ask)"),
         ("canaskmod", "Neutral\n(can_ask_modified)"),
         ("deception", "Deception")]

# (file-tag, policy label as it appears in the JSON, series label, color)  -- fixed categorical order
POLICIES = [
    ("base", "Base-qwen35-35b", "qwen3.5-35b-a3b (prev base)", "#2a78d6"),  # slot 1 blue
    ("qwen36", "Qwen3.6-27b",   "qwen3.6-27b",                 "#eda100"),  # slot 4 yellow
]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"


def agg(file_tag, cond, policy_label):
    """Return per-condition aggregates for the policy across every game it played."""
    fp = RESULTS / f"crossplay_matrix_{file_tag}_{cond}_games.json"
    if not fp.exists():
        return None
    games = json.load(open(fp))["games"]
    own, jeff, nash, agreed = [], [], [], []
    for g in games:
        if policy_label not in (g["opener"], g["partner"]):
            continue
        is_opener = g["opener"] == policy_label
        a = bool(g.get("agreed"))
        agreed.append(1.0 if a else 0.0)
        own.append((g["you_norm"] if is_opener else g["them_norm"]) if a else 0.0)
        nash.append(g.get("nash_product", 0.0) if a else 0.0)
        if a:
            jeff.append(g.get("joint_efficiency", 0.0))
    if not own:
        return None
    return {
        "self": float(np.mean(own)),
        "joint": float(np.mean(jeff)) if jeff else 0.0,
        "nash": float(np.mean(nash)),
        "agree": float(np.mean(agreed)),
        "n": len(own),
    }


def main():
    # collect: data[metric][policy_idx] = [val per condition]
    metrics = [("self", "Self-score  (own normalized outcome, no-deal=0)"),
               ("joint", "Joint efficiency  (achieved / best joint | deal)"),
               ("nash", "Nash product  (you×them normalized, no-deal=0)")]
    table = {}
    for ft, plabel, _, _ in POLICIES:
        for ck, _ in CONDS:
            table[(ft, ck)] = agg(ft, ck, plabel)

    # ---- console table ----
    print(f"\n{'policy':<28}{'cond':<12}{'self':>7}{'joint':>8}{'nash':>7}{'agree':>7}{'n':>5}")
    for ft, plabel, slabel, _ in POLICIES:
        for ck, clabel in CONDS:
            r = table[(ft, ck)]
            if r is None:
                print(f"{slabel:<28}{ck:<12}{'(missing)':>7}")
                continue
            print(f"{slabel:<28}{ck:<12}{r['self']:>7.3f}{r['joint']:>8.3f}"
                  f"{r['nash']:>7.3f}{r['agree']:>7.2f}{r['n']:>5d}")

    # ---- figure: 3 metric panels, grouped bars (2 policies x 3 conditions) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    x = np.arange(len(CONDS))
    w = 0.38
    for ax, (mk, mtitle) in zip(axes, metrics):
        for pi, (ft, plabel, slabel, color) in enumerate(POLICIES):
            vals = [(table[(ft, ck)] or {}).get(mk, np.nan) for ck, _ in CONDS]
            bars = ax.bar(x + (pi - 0.5) * w, vals, w, label=slabel, color=color, zorder=3)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(mtitle, fontsize=10, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in CONDS], fontsize=8.5, color=MUTED)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("mean over policy's cross-play games", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Cross-play vs frontier pool (GPT-5.5 / Opus-4.8 / Gemini-3.1-Pro / Llama-3.3-70B / Qwen3.5-9B)\n"
                 "policy-only · single-proposer · dnd/val n=16/cell · seed 1 · think-gate off",
                 fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    out = RESULTS / "crossplay_qwen36_vs_base_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
