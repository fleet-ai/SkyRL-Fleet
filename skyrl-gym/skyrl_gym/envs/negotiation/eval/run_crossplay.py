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
import re
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


OPENROUTER_URL = "https://openrouter.ai/api/v1"

# A thinking <think> block must never re-enter a model's own multi-turn context
# (matches training's qwen3_without_thinking template + the value-inference probe).
# Parse the action tag from the FULL reply, but feed back only the stripped text.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# Protocol action tags used as generation stop sequences (matches the training rollout).
_STOP_TAGS = {"single": ["</propose>", "</deal>", "<accept>"], "dual": ["</deal>"]}


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _build_sampling(url: str, no_think: bool, *, temperature, max_tokens,
                    presence_penalty=None, stop_tags=None):
    """Per-model sampling config: {temperature, max_tokens, extra_body}. The no-think
    body, optional presence_penalty, and protocol stop tags are folded into extra_body."""
    extra: dict = {}
    if no_think:
        extra.update(_no_think_body(url))
    # Action tags as stop sequences are ONLY safe on the local vLLM, where
    # include_stop_str_in_output keeps the matched tag in the output. OpenRouter
    # providers (Anthropic/Google/Meta observed) strip the matched stop string AND
    # ignore include_stop_str_in_output, which deletes the </propose>/<accept> tag and
    # makes every deal fail to parse (cells collapse to 0.0). So for OpenRouter seats we
    # use NO stop tags: the model emits the full message and parsing scans all of it.
    if stop_tags and "openrouter" not in url:
        extra["stop"] = list(stop_tags)
        extra["include_stop_str_in_output"] = True
    if presence_penalty is not None:
        extra["presence_penalty"] = presence_penalty
    return {"temperature": temperature, "max_tokens": max_tokens, "extra_body": (extra or None)}


def _elicit_blocks(elicit: str):
    """Map an elicitation mode to (seat_A_block, seat_B_block), mirroring
    prepare_dataset.make_row so eval prompts match training exactly."""
    if elicit == "two_sided":
        return prompts.PROACTIVE_BLOCK, prompts.PROACTIVE_BLOCK
    if elicit == "one_sided":
        return prompts.ASK_ONLY_BLOCK, ""
    if elicit == "can_ask":
        return prompts.CAN_ASK_BLOCK, prompts.CAN_ASK_BLOCK
    return "", ""


def _no_think_body(base_url: str):
    """No-think payload differs by backend: OpenRouter honours {"reasoning": ...};
    a locally-served vLLM checkpoint uses chat_template_kwargs.enable_thinking."""
    if "openrouter" in base_url:
        return run_eval.NO_THINK_BODY
    return {"chat_template_kwargs": {"enable_thinking": False}}


async def play_pair(client_a, client_b, sc, model_a, nt_a, url_a, model_b, nt_b, url_b,
                    samp_a, samp_b, max_turns, protocol, you_block="", them_block=""):
    """One negotiation between seat A (you_values, opens) and seat B (them_values),
    under `protocol` ("dual" = both submit <deal>; "single" = one <propose>, other
    <accept>). `samp_a`/`samp_b` are per-model sampling dicts from _build_sampling so
    each side can be sampled to match its own training/rollout distribution.

    Returns the game.Outcome dict (+ num_turns). <think> blocks are parsed for tags
    but stripped before re-entering the multi-turn context (matches training)."""
    items = list(sc.item_names)
    counts = list(sc.counts)
    n = len(counts)
    sys_a = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, counts, list(sc.you_values), max_turns,
                                    protocol=protocol, elicit_block=you_block), nt_a)
    sys_b = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, counts, list(sc.them_values), max_turns,
                                    protocol=protocol, elicit_block=them_block), nt_b)

    hist_a = [{"role": "system", "content": sys_a},
              {"role": "user", "content": prompts.OPENING_USER_MSG}]
    hist_b = [{"role": "system", "content": sys_b}]

    last = {"a": None, "b": None}
    pending = None            # single-protocol: most recent valid offer {"by","keep"}
    you_take = them_take = None
    count = {"a": 0, "b": 0}
    speaker = "a"
    nturns = 0
    while True:
        if speaker == "a":
            text = await run_eval.chat(client_a, model_a, hist_a, samp_a["temperature"],
                                       samp_a["max_tokens"], extra_body=samp_a["extra_body"])
            stripped = _strip_think(text)
            hist_a.append({"role": "assistant", "content": stripped})
            hist_b.append({"role": "user", "content": stripped})
            count["a"] += 1
        else:
            text = await run_eval.chat(client_b, model_b, hist_b, samp_b["temperature"],
                                       samp_b["max_tokens"], extra_body=samp_b["extra_body"])
            stripped = _strip_think(text)
            hist_b.append({"role": "assistant", "content": stripped})
            hist_a.append({"role": "user", "content": stripped})
            count["b"] += 1
        nturns += 1

        if protocol == "dual":
            last[speaker] = game.parse_deal(text, items)
            if last["a"] is not None and last["b"] is not None:
                break
        else:  # single-proposer
            if game.has_accept(text) and pending and pending["by"] != speaker:
                keep = pending["keep"]
                other = [counts[i] - keep[i] for i in range(n)]
                you_take, them_take = (keep, other) if pending["by"] == "a" else (other, keep)
                break
            prop = game.parse_proposal(text, items)
            if prop is not None:
                pending = {"by": speaker, "keep": [min(counts[i], max(0, prop[i])) for i in range(n)]}

        if count["a"] >= max_turns and count["b"] >= max_turns:
            break
        speaker = "b" if speaker == "a" else "a"

    if protocol == "dual":
        outcome = game.evaluate(counts, list(sc.you_values), list(sc.them_values), last["a"], last["b"])
    else:
        outcome = game.evaluate(counts, list(sc.you_values), list(sc.them_values), you_take, them_take)
    d = outcome.to_dict()
    d["num_turns"] = nturns
    return d


async def main_async(args):
    rng = random.Random(args.seed)
    scs = scenarios_mod.load_scenarios(args.dataset, args.split)
    rng.shuffle(scs)
    scs = scs[: args.n]

    # Frontier pool plays via OpenRouter; an optional locally-served policy joins the
    # matrix with its own base_url so transfer (policy vs frontier) is measured directly.
    models = [(s, l, nt, OPENROUTER_URL) for (s, l, nt) in MODELS]
    if args.policy_model:
        models.append((args.policy_model, args.policy_label, args.policy_no_think, args.policy_base_url))
    clients = {url: run_eval.make_client(url) for url in {m[3] for m in models}}
    sem = asyncio.Semaphore(args.concurrency)
    M = len(models)
    you_block, them_block = _elicit_blocks(args.elicit)

    # Per-model sampling. With --match-train-sampling the POLICY model (the trained
    # checkpoint added via --policy-model, always last) is sampled to match the
    # selfplay-canask training rollout (temp 1.0 / max_tokens 8192 / presence_penalty
    # 1.5); every other model stays neutral (temp 1.0, max_tokens 8192). Both get the
    # protocol's action-tag stop sequences so each turn terminates cleanly, as in training.
    policy_idx = (len(models) - 1) if args.policy_model else None
    stop_tags = _STOP_TAGS.get(args.protocol) if args.match_train_sampling else None
    samplings = []
    for i, (s, l, nt, url) in enumerate(models):
        if args.match_train_sampling:
            pp = 1.5 if i == policy_idx else None
            samplings.append(_build_sampling(url, nt, temperature=1.0, max_tokens=8192,
                                             presence_penalty=pp, stop_tags=stop_tags))
        else:
            samplings.append(_build_sampling(url, nt, temperature=args.temperature,
                                             max_tokens=args.max_tokens))

    async def one(ai, bi, sc):
        sa, la, nta, ua = models[ai]
        sb, lb, ntb, ub = models[bi]
        async with sem:
            try:
                return await play_pair(clients[ua], clients[ub], sc, sa, nta, ua, sb, ntb, ub,
                                       samplings[ai], samplings[bi], args.max_turns, args.protocol,
                                       you_block=you_block, them_block=them_block)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e), "agreed": False, "you_norm": 0.0, "them_norm": 0.0,
                        "reason": "error"}

    # --policy-only: only compute the policy's row + column (policy vs each frontier,
    # both seats), skipping the frontier x frontier block to save API cost.
    def _wanted(ai, bi):
        if not args.policy_only or policy_idx is None:
            return True
        return ai == policy_idx or bi == policy_idx

    tasks = []
    index = []
    for ai in range(M):
        for bi in range(M):
            if not _wanted(ai, bi):
                continue
            for sc in scs:
                index.append((ai, bi))
                tasks.append(one(ai, bi, sc))
    ncells = sum(1 for ai in range(M) for bi in range(M) if _wanted(ai, bi))
    print(f"running {len(tasks)} games ({ncells} cells x {len(scs)} scenarios"
          f"{' [policy-only]' if args.policy_only else ''})...", flush=True)
    flat = await asyncio.gather(*tasks)

    # aggregate per cell
    cells = {(ai, bi): [] for ai in range(M) for bi in range(M)}
    for (ai, bi), r in zip(index, flat):
        cells[(ai, bi)].append(r)

    you_outcome = [[None] * M for _ in range(M)]   # seat A normalized outcome
    agree = [[None] * M for _ in range(M)]
    joint = [[None] * M for _ in range(M)]
    for ai in range(M):
        for bi in range(M):
            rs = cells[(ai, bi)]
            n = len(rs)
            if n == 0:
                continue
            you_outcome[ai][bi] = round(sum((r["you_norm"] if r.get("agreed") else 0.0) for r in rs) / n, 4)
            agree[ai][bi] = round(sum(1 for r in rs if r.get("agreed")) / n, 4)
            ja = [r.get("joint_efficiency", 0.0) for r in rs if r.get("agreed")]
            joint[ai][bi] = round(sum(ja) / len(ja), 4) if ja else 0.0

    labels = [m[1] for m in models]
    payload = {
        "config": {"dataset": args.dataset, "split": args.split, "n": len(scs),
                   "max_turns": args.max_turns, "seed": args.seed, "protocol": args.protocol,
                   "elicit": args.elicit, "policy_only": args.policy_only,
                   "match_train_sampling": args.match_train_sampling,
                   "sampling": {"shared_temperature": args.temperature, "shared_max_tokens": args.max_tokens,
                                "policy_idx": policy_idx, "stop_tags": stop_tags,
                                "per_model": [{"label": models[i][1], **samplings[i]} for i in range(M)]},
                   "models": [{"slug": m[0], "label": m[1], "no_think": m[2], "base_url": m[3]} for m in models]},
        "labels": labels,
        "seatA_outcome": you_outcome,
        "agreement": agree,
        "joint_efficiency": joint,
    }
    out = RESULTS / f"{args.out_prefix}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")

    render_heatmap(payload, args.out_prefix)

    if args.probe:
        await run_exploitation_probe(args)


async def run_exploitation_probe(args):
    """Run the exploitation probe alongside cross-play (periodic-eval companion).

    Measures how hard the policy (and base / frontier reference) squeeze a scripted
    pushover — see run_probe.py. Free apart from the measured-model API calls.
    """
    import run_probe  # local import: avoids a circular import at module load

    parts = []
    if args.policy_model:
        parts.append({"slug": args.policy_model, "label": args.policy_label,
                      "no_think": args.policy_no_think, "base_url": args.policy_base_url, "role": "policy"})
    if args.base_model:
        parts.append({"slug": args.base_model, "label": args.base_label,
                      "no_think": True, "base_url": run_probe.OPENROUTER_URL, "role": "base"})
    # Default reference pool = the cross-play frontier models (skipping dupes).
    if args.probe_reference or not parts:
        existing = {p["slug"] for p in parts}
        for slug, label, nt in MODELS:
            if slug not in existing:
                parts.append({"slug": slug, "label": label, "no_think": nt,
                              "base_url": run_probe.OPENROUTER_URL, "role": "reference"})

    await run_probe.run_probe(
        parts, dataset=args.dataset, split=args.split, n=args.n, max_turns=args.max_turns,
        temperature=args.temperature, max_tokens=args.max_tokens, concurrency=args.concurrency,
        seed=args.seed, protocol="dual")


def render_heatmap(payload, out_prefix="crossplay_matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = payload["labels"]
    A = np.array([[np.nan if v is None else v for v in row] for row in payload["seatA_outcome"]], dtype=float)
    M = len(labels)

    with np.errstate(all="ignore"):
        rowmeans = np.nanmean(A, axis=1)   # mean outcome as opener  (vs the field)
        colmeans = np.nanmean(A, axis=0)   # mean outcome conceded TO opponents as partner
    vmax = max(0.6, float(np.nanmax(A))) if np.isfinite(A).any() else 0.6

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=vmax)
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
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.33 else "black", fontsize=9,
                    fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("seat-A normalized outcome (no-deal=0)")
    fig.tight_layout()
    p = RESULTS / f"{out_prefix}_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dnd")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6, help="scenarios per cell")
    ap.add_argument("--protocol", default="dual", choices=["dual", "single"],
                    help="single = single-proposer (<propose>/<accept>); matches the selfplay training distribution")
    ap.add_argument("--match-train-sampling", action="store_true",
                    help="policy seat uses selfplay-canask training sampling (temp 1.0 / max_tokens 8192 / "
                         "presence_penalty 1.5); all seats get protocol stop tags. Others stay neutral (temp 1.0).")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--elicit", default="none", choices=["none", "two_sided", "one_sided", "can_ask"],
                    help="value-elicitation block injected into the system prompts (matches training "
                         "NEGOTIATION_ELICIT; can_ask -> CAN_ASK_BLOCK on BOTH seats)")
    ap.add_argument("--out-prefix", default="crossplay_matrix",
                    help="basename for results/<prefix>.json and results/<prefix>_heatmap.png")
    ap.add_argument("--render-only", action="store_true", help="re-render heatmap from results/<out-prefix>.json")
    # Exploitation probe (run_probe.py) — companion diagnostic for periodic eval.
    ap.add_argument("--probe", action="store_true",
                    help="also run the exploitation probe vs a scripted conceder (see run_probe.py)")
    ap.add_argument("--probe-reference", action="store_true",
                    help="include the frontier MODELS as probe reference even when --policy-model is set")
    ap.add_argument("--policy-model", default=None, help="trained checkpoint slug for the probe")
    ap.add_argument("--policy-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--policy-label", default="Policy")
    # These models are trained THINKING-ON; default to thinking enabled in the matrix.
    ap.add_argument("--policy-no-think", action="store_true", default=False,
                    help="disable thinking for the policy (default: thinking ON)")
    ap.add_argument("--policy-only", action="store_true", default=False,
                    help="only play the policy's row+column vs the frontier (skip frontier x frontier)")
    ap.add_argument("--base-model", default=None, help="pre-RL base model slug for the probe Δ")
    ap.add_argument("--base-label", default="Base")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.render_only:
        render_heatmap(json.loads((RESULTS / f"{args.out_prefix}.json").read_text()), args.out_prefix)
    else:
        asyncio.run(main_async(args))
