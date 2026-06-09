#!/usr/bin/env python3
"""Export self-play eval results into the visualizer's data dir so they can be
browsed in the web UI (with per-game outcome + failure-case classification).

Reads  eval/results/<model>_<dataset>_<split>_n<N>.json  (keeps the largest N
per (model, dataset)) and writes:
  visualizer/public/data/eval/<runid>.json     (one file per run, viz schema)
  visualizer/public/data/eval/manifest.json    (list of runs)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parent / "visualizer" / "public" / "data" / "eval"

ACCEPT_WORDS = ["agree", "works for me", "sounds good", "deal", "that works",
                "ok", "okay", "sure", "accept", "confirm", "perfect", "fine by me"]

FNAME_RE = re.compile(
    r"^(?P<model>.+)_(?P<dataset>dnd|casino)_(?P<split>[a-z]+)"
    r"(?:_(?P<protocol>single-nothink|dual-nothink|single|dual))?_n(?P<n>\d+)\.json$"
)

# Models to omit from the visualizer manifest. Matched against the filename's
# model prefix (e.g. "qwen_qwen-2.5-7b-instruct"), so this hides the noisy plain
# Qwen baselines while keeping DeepSeek's R1-distill-*qwen* candidate (prefix
# "deepseek_"). Exception: explicitly-requested non-thinking Qwen3.5 runs (the
# "-nothink" protocol variants) are always kept — see is_excluded().
EXCLUDE_MODEL_PREFIXES = ("qwen",)


def is_excluded(model: str, protocol: str) -> bool:
    if protocol.endswith("-nothink"):
        return False  # non-thinking runs are always shown
    return model.startswith(EXCLUDE_MODEL_PREFIXES)


def classify_flags(o):
    """Return human-readable failure flags for a no-deal/conflict outcome."""
    flags = []
    reason = o["reason"]
    a, b = o.get("you_take"), o.get("them_take")
    if reason == "no_deal":
        one_sided = (a is None) != (b is None)
        if one_sided:
            flags.append("one_sided_tag")  # only one side emitted a <deal>
        else:
            flags.append("no_tags")        # neither emitted a parseable <deal>
    elif reason == "conflict":
        flags.append("overclaim")          # both emitted deals, claims overlap
    elif reason == "incomplete":
        flags.append("leftover")           # claims leave items unassigned
    return flags


def transcript_accepts(transcript):
    """True if the last 1-2 messages contain an acceptance phrase (heuristic)."""
    for m in transcript[-2:]:
        t = m["text"].lower()
        if any(w in t for w in ACCEPT_WORDS):
            return True
    return False


def to_viz_game(g):
    o = g["outcome"]
    sc = g["scenario"]
    flags = classify_flags(o)
    # Detect the headline failure: a verbal agreement that never produced two tags.
    verbal_no_tag = ("one_sided_tag" in flags or "no_tags" in flags) and transcript_accepts(g["transcript"])
    if verbal_no_tag:
        flags.append("verbal_agreement_no_tag")
    return {
        "dataset": "eval",
        "item_names": sc["item_names"],
        "counts": sc["counts"],
        "you_values": sc["you_values"],
        "them_values": sc["them_values"],
        "you_max": o.get("you_max", 0),
        "them_max": o.get("them_max", 0),
        "turns": g["transcript"],
        "num_turns": g.get("num_turns", len(g["transcript"])),
        "first_speaker": g["transcript"][0]["speaker"] if g["transcript"] else "you",
        "agreed": o["agreed"],
        "valid_alloc": o.get("valid_alloc", o["agreed"]),
        "you_alloc": o.get("you_take"),
        "them_alloc": o.get("them_take"),
        "you_score": o.get("you_score") or 0,
        "them_score": o.get("them_score") or 0,
        "reason": o["reason"],
        "flags": flags,
        "you_norm": o.get("you_norm", 0.0),
        "them_norm": o.get("them_norm", 0.0),
        "pareto_optimal": o.get("pareto_optimal", False),
    }


def summarize(games, score_max):
    n = len(games)
    agreed = [g for g in games if g["agreed"] and g["valid_alloc"]]
    na = len(agreed)
    reason_hist = Counter(g["reason"] for g in games)
    turns_hist = Counter(g["num_turns"] for g in games)
    score_hist = Counter()
    for g in agreed:
        score_hist[g["you_score"]] += 1
        score_hist[g["them_score"]] += 1
    outcome_rewards = []
    for g in games:
        outcome_rewards.append(g["you_norm"] if g["agreed"] else 0.0)
        outcome_rewards.append(g["them_norm"] if g["agreed"] else 0.0)
    verbal = sum(1 for g in games if "verbal_agreement_no_tag" in g["flags"])
    return {
        "num_games": n,
        "num_agreed": na,
        "agreement_rate": round(na / n, 4) if n else 0,
        "avg_turns": round(sum(g["num_turns"] for g in games) / n, 2) if n else 0,
        "avg_you_score": round(sum(g["you_score"] for g in agreed) / na, 2) if na else 0,
        "avg_them_score": round(sum(g["them_score"] for g in agreed) / na, 2) if na else 0,
        "avg_joint_score": round(sum(g["you_score"] + g["them_score"] for g in agreed) / na, 2) if na else 0,
        "score_max": score_max,
        "joint_max": 2 * score_max,
        "avg_outcome_reward": round(sum(outcome_rewards) / len(outcome_rewards), 3) if outcome_rewards else 0,
        "pareto_rate": round(sum(1 for g in agreed if g["pareto_optimal"]) / na, 4) if na else 0,
        "reason_hist": dict(reason_hist),
        "turns_hist": {str(k): v for k, v in turns_hist.items()},
        "score_hist": {str(k): v for k, v in score_hist.items()},
        "extra_cards": [
            {"label": "Outcome reward", "value": f"{round(sum(outcome_rewards)/len(outcome_rewards),3) if outcome_rewards else 0}"},
            {"label": "Pareto (of deals)", "value": f"{round((sum(1 for g in agreed if g['pareto_optimal'])/na*100) if na else 0)}%"},
            {"label": "Verbal-agree, no tag", "value": str(verbal)},
        ],
    }


def pick_runs():
    """Latest (max N) result file per (model, dataset, protocol), excluding tiny smokes.

    Legacy files with no protocol token are treated as the dual-tag protocol.
    """
    best = {}
    for p in RESULTS.glob("*.json"):
        if p.name == "matrix.json":
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        n = int(m.group("n"))
        if n < 10:
            continue
        protocol = m.group("protocol") or "dual"
        if is_excluded(m.group("model"), protocol):
            continue
        key = (m.group("model"), m.group("dataset"), protocol)
        if key not in best or n > best[key][0]:
            best[key] = (n, p)
    return best


PROTO_LABEL = {
    "single": "single-proposer",
    "dual": "dual-tag",
    "single-nothink": "single-proposer · no-think",
    "dual-nothink": "dual-tag · no-think",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for (model, dataset, protocol), (n, path) in sorted(pick_runs().items()):
        payload = json.loads(path.read_text())
        score_max = 10 if dataset == "dnd" else 36
        games = [to_viz_game(g) for g in payload["results"] if "scenario" in g and g.get("transcript")]
        if not games:
            continue
        stats = summarize(games, score_max)
        runid = f"{model}__{dataset}__{protocol}"
        (OUT / f"{runid}.json").write_text(json.dumps({"split": "all", "dataset": "eval",
                                                       "model": model, "stats": stats, "games": games}))
        pretty_model = model.replace("_", "/").replace("qwen/qwen", "Qwen").replace("openai/", "")
        plabel = PROTO_LABEL.get(protocol, protocol)
        name_proto = protocol.replace("-nothink", " (no-think)")
        runs.append({
            "id": runid,
            "name": f"{pretty_model} · {dataset} · {name_proto}",
            "blurb": (f"Self-play eval ({plabel}) · {model.replace('_','/')} on {dataset} · "
                      f"agree {stats['agreement_rate']*100:.0f}% · outcome {stats['avg_outcome_reward']} · "
                      f"{n} scenarios"),
            "splits": ["all"],
            "items": games[0]["item_names"],
            "protocol": protocol,
            "agreement_rate": stats["agreement_rate"],
            "avg_outcome_reward": stats["avg_outcome_reward"],
        })
        print(f"exported {runid}: {len(games)} games, agree {stats['agreement_rate']:.0%}, "
              f"outcome {stats['avg_outcome_reward']}")
    # Group by dataset, then protocol family (single-* first), then outcome reward desc.
    runs.sort(key=lambda r: (r["id"].split("__")[1], 0 if r["protocol"].startswith("single") else 1,
                             -r["avg_outcome_reward"]))
    (OUT / "manifest.json").write_text(json.dumps({"datasets": runs}, indent=2))
    print(f"wrote {len(runs)} runs -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
