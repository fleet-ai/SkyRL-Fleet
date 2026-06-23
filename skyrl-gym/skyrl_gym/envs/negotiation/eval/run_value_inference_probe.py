#!/usr/bin/env python3
"""Value-inference probe — how accurately does the policy infer the opponent's
hidden item values?

Each measured model plays the *opener* (seat A) against a real negotiating
opponent (a frontier reference model, seat B, holding the hidden ``them_values``).
A real opponent — unlike the exploitation probe's scripted conceder — counters,
holds out for items it cares about, and so *leaks preference information* through
the dialogue. We then read the policy's belief about the opponent's values at two
points and score it against ground truth:

  - PRIOR     : before any exchange (only the pool + the policy's own values are
                known). Measures the model's inductive bias about an unseen partner.
  - POSTERIOR : after the negotiation. Measures the belief it actually formed.
  - DELTA     : posterior - prior. The in-context theory-of-mind signal — how much
                the conversation moved the belief toward the truth.

The belief is elicited on a *branched* copy of the policy's own history (a private
side-question the opponent never sees), so eliciting never perturbs the live game.

Scoring is deliberately scale-free: the policy cannot know the opponent's absolute
point scale, only the relative structure that matters for trading. Per game we score
the estimate against ``them_values`` with:
  - spearman  : rank correlation of estimated vs true values (got the ORDERING right).
  - cosine    : cosine similarity of the value vectors (scale-invariant shape match).
  - top1      : did argmax(estimate) land on one of the opponent's true top items?
  - norm_mae  : mean abs error after normalizing each vector to sum=1 (lower = better).

Emits:
  - results/value_inference_probe.json  (config, per-model prior/posterior/delta, raw runs)
  - results/value_inference_probe.png   (grouped bars: prior vs posterior per metric)

Usage (standalone — measure the frontier pool's own inference skill):
  OPENROUTER_API_KEY=... python3 run_value_inference_probe.py --n 16 --max-turns 6

Usage (periodic — local vLLM policy vs its OpenRouter base, opponent = GPT-5.5):
  OPENROUTER_API_KEY=... python3 run_value_inference_probe.py \
      --policy-model my-ckpt --policy-base-url http://localhost:8000/v1 \
      --base-model qwen/qwen3.5-9b --n 16 --max-turns 6
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
sys.path.insert(0, str(HERE))            # sibling modules (run_eval, run_crossplay)
sys.path.insert(0, str(HERE.parent))     # package dir (game / prompts / scenarios)

import game  # noqa: E402
import prompts  # noqa: E402
import run_eval  # noqa: E402

# Frontier reference pool (one per vendor), reused from the cross-play harness both
# as the default opponent and as reference participants. (slug, label, no_think)
try:
    from run_crossplay import MODELS as REFERENCE_MODELS  # noqa: E402
except Exception:  # noqa: BLE001 - keep the probe usable even if crossplay changes
    REFERENCE_MODELS = [("qwen/qwen3.5-9b", "Qwen3.5-9B", True)]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPPONENT = ("openai/gpt-5.5", "GPT-5.5", False)

# As in training, a thinking <think> block must never re-enter the policy's own
# multi-turn context. We parse from the FULL text (tags live after </think>) but
# feed back only the stripped reply. Mirrors run_probe / the env template.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# Private belief tag the policy emits when asked to estimate the opponent's values.
ESTIMATE_RE = re.compile(r"<estimate>\s*(\{.*?\})\s*</estimate>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _no_think_body(base_url: str):
    """No-think payload differs by backend: OpenRouter honours {"reasoning": ...};
    a locally-served vLLM checkpoint uses chat_template_kwargs.enable_thinking."""
    if "openrouter" in base_url:
        return run_eval.NO_THINK_BODY
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _estimate_example(item_names) -> str:
    return "{" + ", ".join(f'"{n}": 0' for n in item_names) + "}"


def _estimate_request(item_names) -> str:
    return (
        "Pause the negotiation for a private side-question — the OTHER player will NOT see this "
        "and it does NOT affect the game. Based on everything you know so far, estimate how many "
        "points EACH item is worth to the OTHER player (their hidden per-unit values, from THEIR "
        "point of view). Give your single best numeric guess for every item even if you are unsure. "
        "Reply with exactly one line of the form:\n"
        f"<estimate>{_estimate_example(item_names)}</estimate>\n"
        "Use whatever numeric scale you think their values are on. Output nothing but that line."
    )


def parse_estimate(text, item_names):
    """Extract a `{item: value}` belief from an <estimate> tag (preferred) or, as a
    lenient fallback, from any `name: number` pairs in the reply. Returns a list
    aligned to item_names, or None if nothing parseable / all zeros."""
    if not text:
        return None
    m = ESTIMATE_RE.search(text)
    vals = game._extract_counts(m.group(1), item_names) if m else game._extract_counts(text, item_names)
    if vals is None or sum(vals) == 0:
        return None
    return vals


# --------------------------------------------------------------------------- #
# Scale-free scoring of one estimate against the ground-truth them_values      #
# --------------------------------------------------------------------------- #
def _ranks(xs):
    """Average (tie-corrected) 1-based ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / ((va ** 0.5) * (vb ** 0.5))


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def _norm_mae(est, truth):
    se, st = sum(est), sum(truth)
    if se == 0 or st == 0:
        return None
    pe = [x / se for x in est]
    pt = [y / st for y in truth]
    return sum(abs(pe[i] - pt[i]) for i in range(len(est))) / len(est)


def _top1(est, truth):
    """1.0 if the predicted top item is one of the opponent's true top items."""
    pred = max(range(len(truth)), key=lambda i: est[i])
    return 1.0 if truth[pred] == max(truth) else 0.0


def score_estimate(est, truth):
    """Score one estimate (list or None) vs ground-truth them_values. Returns a dict
    of metrics; values are None where undefined (no estimate, or zero-variance)."""
    if est is None:
        return {"parsed": 0.0, "spearman": None, "cosine": None, "top1": None, "norm_mae": None}
    return {
        "parsed": 1.0,
        "spearman": _spearman(est, truth),
        "cosine": _cosine(est, truth),
        "top1": _top1(est, truth),
        "norm_mae": _norm_mae(est, truth),
    }


# --------------------------------------------------------------------------- #
# One game: measured policy (seat A / opener) vs a real opponent (seat B)      #
# + a branched prior/posterior value-belief elicitation                        #
# --------------------------------------------------------------------------- #
async def _elicit(client, model, base_hist, items, body, temperature, max_tokens):
    """Ask the policy for its private belief on a *copy* of base_hist (never mutates
    the live game history). Returns (estimate_or_None, raw_reply)."""
    branch = list(base_hist) + [{"role": "user", "content": _estimate_request(items)}]
    text = await run_eval.chat(client, model, branch, temperature, max_tokens, extra_body=body)
    return parse_estimate(text, items), text


async def play_value_game(client_pol, client_opp, *, policy_model, policy_no_think, pol_url,
                          opp_model, opp_no_think, opp_url, sc, max_turns, temperature,
                          max_tokens, est_max_tokens, protocol, opp_proactive):
    items = list(sc.item_names)
    counts = list(sc.counts)
    n = len(counts)
    body_pol = _no_think_body(pol_url) if policy_no_think else None
    body_opp = _no_think_body(opp_url) if opp_no_think else None

    sys_a = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, counts, list(sc.you_values), max_turns, protocol=protocol),
        policy_no_think)
    sys_b = run_eval._maybe_no_think(
        prompts.build_system_prompt(items, counts, list(sc.them_values), max_turns,
                                    protocol=protocol, proactive=opp_proactive),
        opp_no_think)

    # PRIOR belief: only the policy's own values + the pool are known yet.
    prior_hist = [{"role": "system", "content": sys_a}]
    prior_est, prior_raw = await _elicit(
        client_pol, policy_model, prior_hist, items, body_pol, temperature, est_max_tokens)

    # Play the negotiation. hist_a is the policy's own view (branched for the posterior).
    hist_a = [{"role": "system", "content": sys_a},
              {"role": "user", "content": prompts.OPENING_USER_MSG}]
    hist_b = [{"role": "system", "content": sys_b}]

    last = {"a": None, "b": None}
    pending = None  # single-protocol: most recent valid offer {"by","keep"}
    you_take = them_take = None
    cnt = {"a": 0, "b": 0}
    speaker = "a"
    nturns = 0

    while True:
        if speaker == "a":
            text = await run_eval.chat(client_pol, policy_model, hist_a, temperature, max_tokens, extra_body=body_pol)
            stripped = _strip_think(text)
            hist_a.append({"role": "assistant", "content": stripped})
            hist_b.append({"role": "user", "content": stripped})
            cnt["a"] += 1
        else:
            text = await run_eval.chat(client_opp, opp_model, hist_b, temperature, max_tokens, extra_body=body_opp)
            stripped = _strip_think(text)
            hist_b.append({"role": "assistant", "content": stripped})
            hist_a.append({"role": "user", "content": stripped})
            cnt["b"] += 1
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

        if cnt["a"] >= max_turns and cnt["b"] >= max_turns:
            break
        speaker = "b" if speaker == "a" else "a"

    if protocol == "dual":
        outcome = game.evaluate(counts, list(sc.you_values), list(sc.them_values), last["a"], last["b"])
    else:
        outcome = game.evaluate(counts, list(sc.you_values), list(sc.them_values), you_take, them_take)

    # POSTERIOR belief: branch off the policy's post-game history.
    post_est, post_raw = await _elicit(
        client_pol, policy_model, hist_a, items, body_pol, temperature, est_max_tokens)

    them_values = list(sc.them_values)
    return {
        "outcome": outcome.to_dict(),
        "item_names": items,
        "counts": counts,
        "you_values": list(sc.you_values),
        "them_values": them_values,
        "prior_estimate": prior_est,
        "post_estimate": post_est,
        "prior_scores": score_estimate(prior_est, them_values),
        "post_scores": score_estimate(post_est, them_values),
        "prior_raw": prior_raw,
        "post_raw": post_raw,
        "num_turns": nturns,
    }


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _r(x, nd=4):
    return round(x, nd) if x is not None else None


METRIC_KEYS = ["spearman", "cosine", "top1", "norm_mae"]


def aggregate(records):
    n = len(records)
    if n == 0:
        return {}
    recs = [r for r in records if "error" not in r]
    agg = {
        "n": n,
        "agreement_rate": _r(_mean(1.0 if r["outcome"]["agreed"] else 0.0 for r in recs)),
        "avg_turns": _r(_mean(r["num_turns"] for r in recs), 2),
        "prior_parse_rate": _r(_mean(r["prior_scores"]["parsed"] for r in recs)),
        "post_parse_rate": _r(_mean(r["post_scores"]["parsed"] for r in recs)),
    }
    for key in METRIC_KEYS:
        pr = _mean(r["prior_scores"][key] for r in recs)
        po = _mean(r["post_scores"][key] for r in recs)
        agg[f"prior_{key}"] = _r(pr)
        agg[f"post_{key}"] = _r(po)
        agg[f"delta_{key}"] = _r((po - pr) if (pr is not None and po is not None) else None)
    return agg


def _delta_models(policy_agg, base_agg):
    """Δ(policy - base) on the posterior beliefs — does RL change how well the model
    reads the opponent vs its pre-RL base?"""
    out = {}
    for key in METRIC_KEYS:
        pv, bv = policy_agg.get(f"post_{key}"), base_agg.get(f"post_{key}")
        out[f"post_{key}"] = _r(pv - bv) if (pv is not None and bv is not None) else None
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
async def run_value_inference(participants, *, opponent, dataset="dnd", split="val", n=16,
                              max_turns=6, temperature=0.7, max_tokens=2000, est_max_tokens=512,
                              concurrency=12, seed=1, protocol="dual", opp_proactive=False,
                              out_prefix="value_inference_probe", write=True):
    """Run the value-inference probe for each participant against a fixed opponent.

    ``participants`` is a list of dicts ``{slug, label, no_think, base_url, role}``
    (role ∈ policy | base | reference). ``opponent`` is a dict with the same keys.
    Returns the payload dict.
    """
    import scenarios as scenarios_mod  # local: only needed to build the scenario set

    rng = random.Random(seed)
    scs = scenarios_mod.load_scenarios(dataset, split)
    rng.shuffle(scs)
    scs = scs[:n]

    clients = {}
    for p in list(participants) + [opponent]:
        clients.setdefault(p["base_url"], run_eval.make_client(p["base_url"]))

    sem = asyncio.Semaphore(concurrency)

    async def one(p, sc):
        async with sem:
            try:
                return await play_value_game(
                    clients[p["base_url"]], clients[opponent["base_url"]],
                    policy_model=p["slug"], policy_no_think=p["no_think"], pol_url=p["base_url"],
                    opp_model=opponent["slug"], opp_no_think=opponent["no_think"],
                    opp_url=opponent["base_url"], sc=sc, max_turns=max_turns,
                    temperature=temperature, max_tokens=max_tokens, est_max_tokens=est_max_tokens,
                    protocol=protocol, opp_proactive=opp_proactive)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e),
                        "outcome": game.evaluate(list(sc.counts), list(sc.you_values),
                                                 list(sc.them_values), None, None).to_dict(),
                        "item_names": list(sc.item_names), "counts": list(sc.counts),
                        "you_values": list(sc.you_values), "them_values": list(sc.them_values),
                        "prior_estimate": None, "post_estimate": None,
                        "prior_scores": score_estimate(None, list(sc.them_values)),
                        "post_scores": score_estimate(None, list(sc.them_values)),
                        "num_turns": 0}

    print(f"running {len(participants)} models x {len(scs)} scenarios vs opponent "
          f"{opponent['label']} ({protocol} protocol)...", flush=True)

    per_model = {}
    for p in participants:
        recs = await asyncio.gather(*[one(p, sc) for sc in scs])
        per_model[p["label"]] = {"config": p, "aggregate": aggregate(recs), "runs": recs}
        a = per_model[p["label"]]["aggregate"]
        print(f"  [{p['label']:>16}] post_spearman={a['post_spearman']} "
              f"(prior={a['prior_spearman']}, Δ={a['delta_spearman']})  "
              f"post_top1={a['post_top1']}  parse={a['post_parse_rate']}", flush=True)

    policy_label = next((p["label"] for p in participants if p["role"] == "policy"), None)
    base_label = next((p["label"] for p in participants if p["role"] == "base"), None)
    deltas = None
    if policy_label and base_label:
        deltas = _delta_models(per_model[policy_label]["aggregate"], per_model[base_label]["aggregate"])

    payload = {
        "config": {"dataset": dataset, "split": split, "n": len(scs), "max_turns": max_turns,
                   "seed": seed, "protocol": protocol, "opponent": opponent,
                   "opp_proactive": opp_proactive, "participants": participants},
        "policy_label": policy_label,
        "base_label": base_label,
        "per_model": {lbl: {"config": d["config"], "aggregate": d["aggregate"]}
                      for lbl, d in per_model.items()},
        "deltas_policy_minus_base": deltas,
        "per_model_runs": {lbl: d["runs"] for lbl, d in per_model.items()},
    }
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"{out_prefix}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
        try:
            render(payload, out_prefix)
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"(skipped plot: {e})")
    return payload


def render(payload, out_prefix="value_inference_probe"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(payload["per_model"].keys())
    aggs = [payload["per_model"][lbl]["aggregate"] for lbl in labels]
    # spearman/cosine/top1 are "higher = better"; show prior vs posterior side by side.
    metrics = [("spearman", "rank corr"), ("cosine", "cosine"), ("top1", "top-1 item")]
    M = len(labels)
    x = np.arange(M)
    w = 0.13
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * M + 3), 5.4))
    for j, (key, name) in enumerate(metrics):
        prior = [(a.get(f"prior_{key}") or 0.0) for a in aggs]
        post = [(a.get(f"post_{key}") or 0.0) for a in aggs]
        ax.bar(x + (2 * j - 2.5) * w, prior, w, color=f"C{j}", alpha=0.45,
               label=f"{name} (prior)")
        ax.bar(x + (2 * j - 1.5) * w, post, w, color=f"C{j}",
               label=f"{name} (post)")
    ax.axhline(0.0, color="grey", lw=0.8)

    pol, base = payload.get("policy_label"), payload.get("base_label")
    tick_labels = []
    for lbl in labels:
        suffix = "  ◀ policy" if lbl == pol else ("  (base)" if lbl == base else "")
        tick_labels.append(lbl + suffix)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value (higher = better inference)")
    ax.set_ylim(min(-0.1, min((a.get("prior_spearman") or 0.0) for a in aggs) - 0.05), 1.0)
    cfg = payload["config"]
    title = (f"Opponent value-inference probe vs {cfg['opponent']['label']} · {cfg['protocol']} · "
             f"{cfg['dataset']}/{cfg['split']} · n={cfg['n']}\n"
             f"prior (faint) vs posterior (solid): how well each model reads the hidden them_values")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    p = RESULTS / f"{out_prefix}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_participants(args):
    parts = []
    if args.policy_model:
        parts.append({"slug": args.policy_model, "label": args.policy_label,
                      "no_think": args.policy_no_think, "base_url": args.policy_base_url,
                      "role": "policy"})
    if args.base_model:
        parts.append({"slug": args.base_model, "label": args.base_label,
                      "no_think": args.base_no_think, "base_url": args.base_base_url,
                      "role": "base"})
    if args.reference or not parts:
        existing = {p["slug"] for p in parts}
        for slug, label, nt in REFERENCE_MODELS:
            if slug not in existing and slug != args.opponent_model:
                parts.append({"slug": slug, "label": label, "no_think": nt,
                              "base_url": OPENROUTER_URL, "role": "reference"})
    return parts


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dnd")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=16, help="scenarios per model")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--est-max-tokens", type=int, default=512,
                    help="token budget for the private value-estimate reply")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--protocol", default="dual", choices=["dual", "single"])
    # The opponent (seat B) whose hidden values are being inferred.
    ap.add_argument("--opponent-model", default=DEFAULT_OPPONENT[0])
    ap.add_argument("--opponent-base-url", default=OPENROUTER_URL)
    ap.add_argument("--opponent-label", default=DEFAULT_OPPONENT[1])
    ap.add_argument("--opponent-no-think", dest="opponent_no_think", action="store_true",
                    default=DEFAULT_OPPONENT[2])
    ap.add_argument("--opponent-think", dest="opponent_no_think", action="store_false")
    ap.add_argument("--opponent-proactive", action="store_true",
                    help="opponent volunteers its priorities (upper bound on inferability)")
    # Trained policy (typically a local vLLM endpoint).
    ap.add_argument("--policy-model", default=None, help="trained checkpoint slug/name")
    ap.add_argument("--policy-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--policy-label", default="Policy")
    ap.add_argument("--policy-no-think", dest="policy_no_think", action="store_true", default=True)
    ap.add_argument("--policy-think", dest="policy_no_think", action="store_false")
    # Pre-RL base (turns the Δ into "did RL change opponent-modeling?").
    ap.add_argument("--base-model", default=None, help="pre-RL base model slug")
    ap.add_argument("--base-base-url", default=OPENROUTER_URL)
    ap.add_argument("--base-label", default="Base")
    ap.add_argument("--base-no-think", dest="base_no_think", action="store_true", default=True)
    ap.add_argument("--base-think", dest="base_no_think", action="store_false")
    # Frontier reference pool.
    ap.add_argument("--reference", action="store_true", help="also measure the frontier reference pool")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render the bar chart from value_inference_probe.json")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.render_only:
        payload = json.loads((RESULTS / "value_inference_probe.json").read_text())
        payload.pop("per_model_runs", None)
        render(payload)
        return
    participants = build_participants(args)
    opponent = {"slug": args.opponent_model, "label": args.opponent_label,
                "no_think": args.opponent_no_think, "base_url": args.opponent_base_url,
                "role": "opponent"}
    asyncio.run(run_value_inference(
        participants, opponent=opponent, dataset=args.dataset, split=args.split, n=args.n,
        max_turns=args.max_turns, temperature=args.temperature, max_tokens=args.max_tokens,
        est_max_tokens=args.est_max_tokens, concurrency=args.concurrency, seed=args.seed,
        protocol=args.protocol, opp_proactive=args.opponent_proactive))


if __name__ == "__main__":
    main()
