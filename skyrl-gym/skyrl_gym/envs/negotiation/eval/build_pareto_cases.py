#!/usr/bin/env python3
"""Build a curated 'Pareto case studies' run for the visualizer.

For each dnd scenario we show FOUR same-scenario cards in matched same-protocol
pairs: frontier-dual vs Qwen3.5-9B-dual, and frontier-single vs Qwen3.5-9B-single.
Under either protocol the frontier reaches the Pareto frontier while Qwen3.5-9B
either fails to close or *agrees* on a clearly inefficient split (off-frontier,
joint value left on the table). The point: agreement-rate and own-outcome reward
can both look fine while the deal is integratively bad — exactly what a Pareto /
joint-efficiency reward would catch. Run after export_to_viz.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import export_to_viz as E

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parent / "visualizer" / "public" / "data" / "eval"

FRONTIER_DUAL = {
    "gpt-5.5": "openai_gpt-5.5_dnd_val_dual_n12.json",
    "opus-4.8": "anthropic_claude-opus-4.8_dnd_val_dual_n12.json",
    "gemini-3.1": "google_gemini-3.1-pro-preview_dnd_val_dual_n12.json",
}
FRONTIER_SINGLE = {
    "gpt-5.5": "openai_gpt-5.5_dnd_val_single_n12.json",
    "opus-4.8": "anthropic_claude-opus-4.8_dnd_val_single_n12.json",
    "gemini-3.1": "google_gemini-3.1-pro-preview_dnd_val_single_n12.json",
}
QWEN_DUAL = "qwen_qwen3.5-9b_dnd_val_dual-nothink_n12.json"
QWEN_SINGLE = "qwen_qwen3.5-9b_dnd_val_single-nothink_n20.json"

# (case#, frontier_key, frontier_pretty, scenario_sig). All four cards play the
# SAME scenario, in matched same-protocol pairs: frontier-dual vs qwen-9b-dual,
# then frontier-single vs qwen-9b-single.
CASES = [
    # scenarios where qwen-9b *agreed under dual* but off-frontier (the cleanest same-protocol contrast)
    (1, "gemini-3.1", "Gemini-3.1-Pro", ((1, 2, 2), (2, 0, 4), (0, 2, 3))),  # qwen-dual 8/14, partner gets 0
    (2, "gpt-5.5",    "GPT-5.5",        ((1, 2, 2), (0, 2, 3), (4, 0, 3))),  # qwen-dual 10/14
    # scenario where qwen-9b's dual deal IS on-frontier but its single deal is not (honest nuance)
    (3, "gpt-5.5",    "GPT-5.5",        ((3, 3, 1), (2, 0, 4), (1, 2, 1))),  # qwen-dual 14/16 Pareto, single 13/16
    # scenarios where under dual qwen-9b can't even close (no-deal) while frontier hits the frontier
    (4, "gpt-5.5",    "GPT-5.5",        ((1, 1, 3), (9, 1, 0), (2, 2, 2))),  # qwen-dual no-deal; single 6/17
    (5, "opus-4.8",   "Claude-Opus-4.8", ((2, 1, 2), (3, 2, 1), (0, 0, 5))),  # qwen-dual no-deal; single 11/18
    (6, "opus-4.8",   "Claude-Opus-4.8", ((3, 1, 1), (2, 3, 1), (0, 8, 2))),  # qwen-dual no-deal; single 9/16
]


def sig(sc):
    return (tuple(sc["counts"]), tuple(sc["you_values"]), tuple(sc["them_values"]))


def index_by_sig(fname):
    out = {}
    for r in json.loads((RESULTS / fname).read_text())["results"]:
        if "scenario" in r and r.get("transcript"):
            out[sig(r["scenario"])] = r
    return out


def joint_max_for(g):
    # max_joint is present on agreed-non-Pareto games; otherwise compute it.
    if g.get("max_joint"):
        return g["max_joint"]
    mj, _, _ = E.efficient_split(g["counts"], g["you_values"], g["them_values"])
    return mj


def tag_for(g, mj):
    if not g.get("agreed"):
        return f"NO DEAL \u2014 {g.get('reason','')} \u2192 0/0"
    if g.get("pareto_optimal"):
        return f"Pareto-optimal ({g['joint_score']}/{mj})"
    return f"off-frontier, {mj - g['joint_score']} joint pts lost ({g['joint_score']}/{mj})"


def main():
    qd_idx = index_by_sig(QWEN_DUAL)
    qs_idx = index_by_sig(QWEN_SINGLE)
    fd_idx = {k: index_by_sig(v) for k, v in FRONTIER_DUAL.items()}
    fs_idx = {k: index_by_sig(v) for k, v in FRONTIER_SINGLE.items()}

    games = []
    for case_no, fkey, fpretty, s in CASES:
        s = (tuple(s[0]), tuple(s[1]), tuple(s[2]))
        fd, qd = fd_idx[fkey].get(s), qd_idx.get(s)
        fs, qs = fs_idx[fkey].get(s), qs_idx.get(s)
        if any(x is None for x in (fd, qd, fs, qs)):
            print(f"!! case {case_no}: missing (f-dual={fd is not None}, q-dual={qd is not None}, "
                  f"f-single={fs is not None}, q-single={qs is not None})")
            continue
        fdg, qdg, fsg, qsg = (E.to_viz_game(fd), E.to_viz_game(qd),
                              E.to_viz_game(fs), E.to_viz_game(qs))
        mj = joint_max_for(fdg)
        # Matched same-protocol pairs: dual pair first, then single pair.
        fdg["model_label"] = f"Case {case_no} \u00b7 {fpretty} \u00b7 dual  ({tag_for(fdg, mj)})"
        qdg["model_label"] = f"Case {case_no} \u00b7 Qwen3.5-9B \u00b7 dual  ({tag_for(qdg, mj)})"
        fsg["model_label"] = f"Case {case_no} \u00b7 {fpretty} \u00b7 single  ({tag_for(fsg, mj)})"
        qsg["model_label"] = f"Case {case_no} \u00b7 Qwen3.5-9B \u00b7 single  ({tag_for(qsg, mj)})"
        for g in (fdg, qdg, fsg, qsg):
            g["case_no"] = case_no
        games += [fdg, qdg, fsg, qsg]

    stats = E.summarize(games, 10)
    runid = "pareto_cases__dnd__curated"
    (OUT / f"{runid}.json").write_text(json.dumps(
        {"split": "all", "dataset": "eval", "model": "Pareto case studies", "stats": stats, "games": games}))

    # prepend to manifest
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text())
    man["datasets"] = [r for r in man["datasets"] if r["id"] != runid]
    man["datasets"].insert(0, {
        "id": runid,
        "name": "\u2b50 Pareto case studies \u00b7 frontier vs Qwen3.5-9B (matched protocols)",
        "blurb": ("Same dnd scenarios, four cards each in matched same-protocol pairs: frontier vs "
                  "Qwen3.5-9B on dual, then frontier vs Qwen3.5-9B on single. Under either protocol the "
                  "frontier hits the Pareto frontier while qwen-9b fails to close or agrees off-frontier "
                  "\u2014 a gap only a Pareto / joint-efficiency reward captures."),
        "splits": ["all"],
        "items": games[0]["item_names"] if games else ["book", "hat", "ball"],
        "protocol": "curated",
        "agreement_rate": stats["agreement_rate"],
        "avg_outcome_reward": stats["avg_outcome_reward"],
    })
    man_path.write_text(json.dumps(man, indent=2))
    ncases = len({g["case_no"] for g in games})
    print(f"wrote {runid}: {len(games)} games across {ncases} cases -> manifest updated")


if __name__ == "__main__":
    main()
