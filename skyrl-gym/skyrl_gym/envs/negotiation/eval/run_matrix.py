#!/usr/bin/env python3
"""Run the negotiation self-play eval across a matrix of models x datasets and
emit a Markdown comparison report (REPORT.md) + combined JSON (matrix.json).

Usage:
  OPENROUTER_API_KEY=... python3 run_matrix.py            # full matrix
  python3 run_matrix.py --report-only                     # regenerate REPORT.md from matrix.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
from pathlib import Path

import run_eval

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
MATRIX_JSON = RESULTS_DIR / "matrix.json"
REPORT_MD = HERE / "REPORT.md"

# Protocol under test: "single" = one proposer + <accept>; "dual" = both emit <deal>.
PROTOCOL = "single"

# (slug, label, tier, max_tokens, temperature, concurrency)
MODELS = [
    ("openai/gpt-4o-mini",                "GPT-4o-mini",  "small ref",         800, 0.7, 15),
    ("qwen/qwen-2.5-7b-instruct",         "Qwen2.5-7B",   "7B (train target)", 800, 0.7, 15),
    ("qwen/qwen-2.5-coder-32b-instruct",  "Qwen2.5-32B*", "32B*",              800, 0.7, 15),
    ("qwen/qwen3.5-9b",                   "Qwen3.5-9B",   "9B",                900, 0.7, 15),
]

DATASETS = [("dnd", "val"), ("casino", "all")]
N = 20
MAX_TURNS = 6
SEED = 1

_DUAL_RE = re.compile(r"^(?P<model>.+)_(?P<ds>dnd|casino)_(?P<split>[a-z]+)_n(?P<n>\d+)\.json$")


def load_dual_baselines():
    """Aggregates from the legacy dual-tag result files (no protocol token in name),
    keyed by (model, dataset), keeping the largest-N file. Used for the
    dual -> single protocol comparison in the report."""
    best = {}
    for p in RESULTS_DIR.glob("*.json"):
        if p.name == "matrix.json":
            continue
        m = _DUAL_RE.match(p.name)
        if not m:
            continue  # single-/dual-tagged files have a protocol token -> skip
        n = int(m.group("n"))
        key = (m.group("model"), m.group("ds"))
        if key not in best or n > best[key][0]:
            best[key] = (n, p)
    out = {}
    for (model, ds), (n, p) in best.items():
        try:
            out[(model.replace("_", "/"), ds)] = json.loads(p.read_text())["aggregate"]
        except Exception:  # noqa: BLE001
            pass
    return out


async def run_matrix():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = run_eval.make_client("https://openrouter.ai/api/v1")
    payloads = []
    for slug, label, tier, max_tokens, temp, conc in MODELS:
        for dataset, split in DATASETS:
            print(f"\n=== {label} ({slug}) on {dataset}/{split} ===", flush=True)
            try:
                payload = await run_eval.evaluate_model(
                    model=slug, dataset=dataset, split=split, n=N, max_turns=MAX_TURNS,
                    temperature=temp, concurrency=conc, seed=SEED, max_tokens=max_tokens,
                    out_dir=str(RESULTS_DIR), client=client, label=label, protocol=PROTOCOL,
                )
            except Exception as e:  # noqa: BLE001
                print(f"!! {label} on {dataset} failed: {e}", flush=True)
                continue
            payload["config"]["tier"] = tier
            agg = payload["aggregate"]
            print(f"   -> agree {agg['agreement_rate']:.0%} | outcome {agg['avg_outcome_reward']:.3f} "
                  f"| pareto {agg['pareto_rate_of_agreements']:.0%} | errors {payload['config']['n_errors']}",
                  flush=True)
            payloads.append(payload)
            # checkpoint after every run
            MATRIX_JSON.write_text(json.dumps(payloads, indent=2))
    return payloads


def _fmt_pct(x):
    return f"{x*100:.0f}%"


def generate_report(payloads):
    by_ds = {}
    for p in payloads:
        by_ds.setdefault(p["config"]["dataset"], []).append(p)

    ds_titles = {
        "dnd": "Deal or No Deal (books / hats / balls, values sum to 10)",
        "casino": "CaSiNo (food / water / firewood, High/Med/Low = 5/4/3, max 36)",
    }
    dual = load_dual_baselines()

    lines = []
    lines.append("# Negotiation RLVR — Baseline Eval Report")
    lines.append("")
    lines.append(f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · protocol: **single-proposer**_")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "- **Task**: two agents divide a shared item pool; each has private per-item values."
    )
    lines.append(
        "- **Protocol (single-proposer)**: agents alternate short messages. To offer, an agent ends a "
        "message with `<propose>{...}</propose>` listing how many of each item *they* keep (the partner "
        "automatically gets the rest); the other agent finalizes it with `<accept>`. Because a single "
        "offer always partitions the pool, **`conflict` and `incomplete` outcomes are impossible** — the "
        "only failure is `no_deal` (nothing accepted within the budget). This replaces the earlier "
        "**dual-tag** protocol (both sides had to emit matching `<deal>` tags), whose coordination "
        "overhead produced large numbers of spurious `no_deal`/`conflict` losses for smaller models."
    )
    lines.append(
        "- **Self-play** (same model on both sides), up to "
        f"{MAX_TURNS} messages/agent, via OpenRouter. {N} scenarios/dataset (seed {SEED}, identical across models)."
    )
    lines.append("- **Verifiable metrics** (the candidate RLVR rewards):")
    lines.append("  - **Outcome reward** = normalized self-score (`score / max_possible`), no-deal = 0.")
    lines.append("  - **Pareto-optimal?** = is the agreed split on the exact Pareto frontier (enumerated).")
    lines.append("  - **Joint efficiency** = achieved joint score / best achievable joint score.")
    lines.append("")
    lines.append("> **Note on 32B**: OpenRouter only serves `qwen-2.5-coder-32b-instruct` for Qwen2.5-32B "
                 "(marked `32B*`). Swap for a plain Qwen2.5-32B-Instruct endpoint if you serve one locally.")
    lines.append("")

    for dataset, _split in DATASETS:
        ps = by_ds.get(dataset, [])
        if not ps:
            continue
        ps.sort(key=lambda p: -p["aggregate"]["avg_outcome_reward"])
        lines.append(f"## {ds_titles.get(dataset, dataset)}")
        lines.append("")
        lines.append("| Model | Tier | Agree | No-deal | Outcome reward | Pareto (of deals) | Joint eff | Pts/agent | Turns | Err |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for p in ps:
            c, a = p["config"], p["aggregate"]
            lines.append(
                f"| {c['model']} | {c.get('tier','')} | {_fmt_pct(a['agreement_rate'])} | "
                f"{_fmt_pct(a['no_deal_rate'])} | **{a['avg_outcome_reward']:.3f}** | "
                f"{_fmt_pct(a['pareto_rate_of_agreements'])} | {_fmt_pct(a['avg_joint_efficiency_of_agreements'])} | "
                f"{a['avg_points_per_agent']:.2f} | {a['avg_turns']:.1f} | {c['n_errors']} |"
            )
        lines.append("")

    # Dual -> single protocol comparison (where a dual-tag baseline exists).
    comp_rows = []
    for dataset, _split in DATASETS:
        for p in by_ds.get(dataset, []):
            model = p["config"]["model"]
            d = dual.get((model, dataset))
            if not d:
                continue
            s = p["aggregate"]
            comp_rows.append((
                model, dataset,
                d["agreement_rate"], s["agreement_rate"],
                d["avg_outcome_reward"], s["avg_outcome_reward"],
            ))
    if comp_rows:
        lines.append("## Protocol comparison: dual-tag → single-proposer")
        lines.append("")
        lines.append("Same models/seeds; shows how much of the earlier \"failure\" was protocol overhead "
                     "rather than negotiation ability. (Dual baselines are the most recent dual-tag runs on disk.)")
        lines.append("")
        lines.append("| Model | Dataset | Agree (dual→single) | Outcome reward (dual→single) | Δ outcome |")
        lines.append("|---|---|---|---|---|")
        for model, ds, ad, as_, od, os_ in comp_rows:
            lines.append(
                f"| {model} | {ds} | {_fmt_pct(ad)} → **{_fmt_pct(as_)}** | "
                f"{od:.3f} → **{os_:.3f}** | {os_-od:+.3f} |"
            )
        lines.append("")

    # Auto takeaways
    lines.append("## Takeaways")
    lines.append("")
    for dataset, _split in DATASETS:
        ps = by_ds.get(dataset, [])
        if not ps:
            continue
        best = max(ps, key=lambda p: p["aggregate"]["avg_outcome_reward"])
        target = next((p for p in ps if "train target" in p["config"].get("tier", "")), None)
        line = f"- **{dataset}**: best outcome reward = `{best['config']['model']}` " \
               f"({best['aggregate']['avg_outcome_reward']:.3f})."
        if target:
            a = target["aggregate"]
            gap = best["aggregate"]["avg_outcome_reward"] - a["avg_outcome_reward"]
            line += (f" Train target `{target['config']['model']}`: outcome {a['avg_outcome_reward']:.3f}, "
                     f"agree {_fmt_pct(a['agreement_rate'])}, no-deal {_fmt_pct(a['no_deal_rate'])}, "
                     f"pareto {_fmt_pct(a['pareto_rate_of_agreements'])} → **headroom {gap:+.3f}** to the best model here.")
        lines.append(line)
    lines.append("")
    lines.append("## Implications for RLVR")
    lines.append("")
    lines.append("- **Single-proposer isolates negotiation skill from formatting.** With coordination overhead "
                 "removed, agreement rates jump and the remaining `no_deal` cases reflect genuine failures to "
                 "reach/accept a deal in the budget — a cleaner, less noisy training signal.")
    lines.append("- **Outcome reward is the primary RLVR target**; the gap from the train target to the best "
                 "model here is the headroom RL should close (drive `no_deal` → 0 and push self-score up).")
    lines.append("- **Pareto rate stays informative even when agreement is high** — outcome-only reward optimizes "
                 "own score and need not reach the efficient frontier; the planned **outcome vs outcome+Pareto** "
                 "ablation tests whether a Pareto bonus raises joint efficiency without hurting self-score.")
    lines.append("- Per-run transcripts + metrics are saved in `results/<model>_<dataset>_<protocol>_n*.json`; "
                 "browse them (with failure-case highlighting) in the visualizer's **Model eval** tab.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **`no_deal` only**: under single-proposer there is no `conflict`/`incomplete` — a no-deal means "
                 "no offer was accepted within the message budget (stubborn back-and-forth, or a model that never "
                 "emits a clean `<propose>`/`<accept>`). Raising `max_turns` slightly can reduce it.")
    lines.append(f"- **Small samples**: with {N} scenarios/dataset, Pareto / joint-efficiency rates over "
                 "*agreements only* are noisy when agreement counts are small.")
    lines.append("- **Dual baselines aren't perfectly matched**: the dual-tag runs used for the comparison may "
                 "differ in N/max_turns from the single-proposer runs; treat the comparison as directional.")
    lines.append("- Closed frontier models (GPT-5, Claude) were skipped (too slow/costly in self-play). "
                 "`gpt-4o-mini` is the reference ceiling here; the others are realistic open RL training targets.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="regenerate REPORT.md from matrix.json")
    args = ap.parse_args()

    if args.report_only:
        payloads = json.loads(MATRIX_JSON.read_text())
    else:
        payloads = asyncio.run(run_matrix())

    REPORT_MD.write_text(generate_report(payloads))
    print(f"\nWrote report -> {REPORT_MD}")
    print(f"Combined results -> {MATRIX_JSON}")


if __name__ == "__main__":
    main()
