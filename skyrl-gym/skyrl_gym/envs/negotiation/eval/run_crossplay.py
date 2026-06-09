#!/usr/bin/env python3
"""Cross-play heatmap matrix for the negotiation task (dual-tag protocol).

Plays every ordered pair (row=seat A / opener, col=seat B / partner) of a small
set of models — one per vendor — over the same dnd scenarios, and records seat A's
mean normalized outcome. Emits:
  - results/crossplay_matrix.json   (raw + aggregated)
  - results/crossplay_heatmap.png   (heatmap of seat-A outcome)

Per-side `no_think` is honoured (Qwen3 hybrids run with thinking OFF, since the
task is turn-budgeted; other vendors are left untouched).

Usage:
  OPENROUTER_API_KEY=... python3 run_crossplay.py --n 6 --max-turns 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent))  # game/prompts/scenarios live in the package dir

import game  # noqa: E402
import prompts  # noqa: E402
import run_eval  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402

# (slug, short label, no_think)  — one model per vendor.
MODELS = [
    ("openai/gpt-5.5",                   "GPT-5.5",      False),
    ("anthropic/claude-opus-4.8",        "Opus-4.8",     False),
    ("google/gemini-3.1-pro-preview",    "Gemini-3.1-Pro", False),
    ("meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B", False),
    ("qwen/qwen3.5-9b",                  "Qwen3.5-9B",   True),   # hybrid -> no-think
]


async def play_dual_pair(client, sc, model_a, nt_a, model_b, nt_b, max_turns, temperature, max_tokens):
    """Dual-tag game with independent no_think per seat. Seat A holds you_values,
    opens; seat B holds them_values. Returns the game.Outcome dict."""
    items = list(sc.item_names)
    body_a = run_eval.NO_THINK_BODY if nt_a else None
    body_b = run_eval.NO_THINK_BODY if nt_b else None
    sys_a = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, list(sc.counts), list(sc.you_values), max_turns, protocol="dual"), nt_a)
    sys_b = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, list(sc.counts), list(sc.them_values), max_turns, protocol="dual"), nt_b)

    hist_a = [{"role": "system", "content": sys_a},
              {"role": "user", "content": prompts.OPENING_USER_MSG}]
    hist_b = [{"role": "system", "content": sys_b}]

    last = {"a": None, "b": None}
    count = {"a": 0, "b": 0}
    speaker = "a"
    nturns = 0
    while True:
        if speaker == "a":
            text = await run_eval.chat(client, model_a, hist_a, temperature, max_tokens, extra_body=body_a)
            hist_a.append({"role": "assistant", "content": text})
            hist_b.append({"role": "user", "content": text})
            last["a"] = game.parse_deal(text, items)
            count["a"] += 1
        else:
            text = await run_eval.chat(client, model_b, hist_b, temperature, max_tokens, extra_body=body_b)
            hist_b.append({"role": "assistant", "content": text})
            hist_a.append({"role": "user", "content": text})
            last["b"] = game.parse_deal(text, items)
            count["b"] += 1
        nturns += 1
        if last["a"] is not None and last["b"] is not None:
            break
        if count["a"] >= max_turns and count["b"] >= max_turns:
            break
        speaker = "b" if speaker == "a" else "a"

    outcome = game.evaluate(list(sc.counts), list(sc.you_values), list(sc.them_values), last["a"], last["b"])
    d = outcome.to_dict()
    d["num_turns"] = nturns
    return d


async def main_async(args):
    rng = random.Random(args.seed)
    scs = scenarios_mod.load_scenarios(args.dataset, args.split)
    rng.shuffle(scs)
    scs = scs[: args.n]

    client = run_eval.make_client("https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.concurrency)
    M = len(MODELS)

    async def one(ai, bi, sc):
        sa, la, nta = MODELS[ai]
        sb, lb, ntb = MODELS[bi]
        async with sem:
            try:
                return await play_dual_pair(client, sc, sa, nta, sb, ntb,
                                            args.max_turns, args.temperature, args.max_tokens)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e), "agreed": False, "you_norm": 0.0, "them_norm": 0.0,
                        "reason": "error"}

    tasks = []
    index = []
    for ai in range(M):
        for bi in range(M):
            for sc in scs:
                index.append((ai, bi))
                tasks.append(one(ai, bi, sc))
    print(f"running {len(tasks)} games ({M}x{M} cells x {len(scs)} scenarios)...", flush=True)
    flat = await asyncio.gather(*tasks)

    # aggregate per cell
    cells = {(ai, bi): [] for ai in range(M) for bi in range(M)}
    for (ai, bi), r in zip(index, flat):
        cells[(ai, bi)].append(r)

    you_outcome = [[0.0] * M for _ in range(M)]   # seat A normalized outcome
    agree = [[0.0] * M for _ in range(M)]
    joint = [[0.0] * M for _ in range(M)]
    for ai in range(M):
        for bi in range(M):
            rs = cells[(ai, bi)]
            n = len(rs)
            you_outcome[ai][bi] = round(sum((r["you_norm"] if r.get("agreed") else 0.0) for r in rs) / n, 4)
            agree[ai][bi] = round(sum(1 for r in rs if r.get("agreed")) / n, 4)
            ja = [r.get("joint_efficiency", 0.0) for r in rs if r.get("agreed")]
            joint[ai][bi] = round(sum(ja) / len(ja), 4) if ja else 0.0

    labels = [m[1] for m in MODELS]
    payload = {
        "config": {"dataset": args.dataset, "split": args.split, "n": len(scs),
                   "max_turns": args.max_turns, "seed": args.seed, "protocol": "dual",
                   "models": [{"slug": m[0], "label": m[1], "no_think": m[2]} for m in MODELS]},
        "labels": labels,
        "seatA_outcome": you_outcome,
        "agreement": agree,
        "joint_efficiency": joint,
    }
    out = RESULTS / "crossplay_matrix.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")

    render_heatmap(payload)


def render_heatmap(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = payload["labels"]
    A = np.array(payload["seatA_outcome"])
    M = len(labels)

    rowmeans = A.mean(axis=1)   # mean outcome as opener  (vs the field)
    colmeans = A.mean(axis=0)   # mean outcome conceded TO opponents as partner

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=max(0.6, A.max()))
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels([f"{l}\n(opp μ={cm:.2f})" for l, cm in zip(labels, colmeans)],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([f"{l}  (μ={rm:.2f})" for l, rm in zip(labels, rowmeans)], fontsize=8)
    ax.set_xlabel("Partner (seat B)")
    ax.set_ylabel("Opener (seat A) — score is for this side")
    title = (f"Cross-play seat-A outcome reward · dual · {payload['config']['dataset']}/"
             f"{payload['config']['split']} · n={payload['config']['n']}/cell\n"
             f"row μ = how well the opener does vs the field (higher = stronger)")
    ax.set_title(title, fontsize=10)
    for i in range(M):
        for j in range(M):
            v = A[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.33 else "black", fontsize=9,
                    fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("seat-A normalized outcome (no-deal=0)")
    fig.tight_layout()
    p = RESULTS / "crossplay_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dnd")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6, help="scenarios per cell")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--render-only", action="store_true", help="re-render heatmap from crossplay_matrix.json")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.render_only:
        render_heatmap(json.loads((RESULTS / "crossplay_matrix.json").read_text()))
    else:
        asyncio.run(main_async(args))
