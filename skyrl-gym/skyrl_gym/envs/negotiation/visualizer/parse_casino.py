#!/usr/bin/env python3
"""Parse the CaSiNo (CampSite Negotiations) ConvoKit corpus into the visualizer's JSON schema.

Source: https://convokit.cornell.edu/documentation/casino-corpus.html
        (https://github.com/kushalchawla/CaSiNo)

Two campers negotiate over three packages: Food, Water, Firewood. There are 3 units of
each. Each negotiator privately ranks the issues High / Medium / Low priority, worth
5 / 4 / 3 points per unit respectively (max 3*(5+4+3) = 36 per agent). They chat freely,
then one submits a deal (Submit-Deal) which the other Accepts / Rejects / Walks-Away from.

We emit the same per-game schema as the Deal-or-No-Deal parser so the web app can render
both datasets, adding CaSiNo-only fields (per-item reasons, strategy annotations,
satisfaction, opponent likeness).
"""

import json
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "casino-corpus"
OUT_DIR = Path(__file__).resolve().parent / "public" / "data" / "casino"

ITEM_NAMES = ["food", "water", "firewood"]
ISSUE_ORDER = ["Food", "Water", "Firewood"]
PREF_VALUE = {"High": 5, "Medium": 4, "Low": 3}
UNITS_PER_ITEM = 3
SCORE_MAX = UNITS_PER_ITEM * sum(PREF_VALUE.values())  # 36

CONTROL_TOKENS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
SATISFACTION_SCALE = {
    "Extremely dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Undecided": 3,
    "Slightly satisfied": 4,
    "Extremely satisfied": 5,
}


def agent_values(participant):
    """value2issue maps preference -> issue. Return per-item value in ISSUE_ORDER."""
    issue2pref = {issue: pref for pref, issue in participant["value2issue"].items()}
    return [PREF_VALUE[issue2pref[issue]] for issue in ISSUE_ORDER]


def agent_reasons(participant):
    issue2pref = {issue: pref for pref, issue in participant["value2issue"].items()}
    reasons = participant.get("value2reason", {})
    return [reasons.get(issue2pref[issue], "") for issue in ISSUE_ORDER]


def uid(utt_id):
    m = re.search(r"(\d+)$", utt_id)
    return int(m.group(1)) if m else 0


def load_utterances():
    by_conv = {}
    with open(CORPUS_DIR / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            by_conv.setdefault(u["conversation_id"], []).append(u)
    for conv_id in by_conv:
        by_conv[conv_id].sort(key=lambda u: uid(u["id"]))
    return by_conv


def extract_deal(utts):
    """Find the Submit-Deal that is immediately followed by Accept-Deal.

    issue2youget is from the *submitter's* perspective. Returns
    (agreed, you_alloc, them_alloc) where 'you' == mturk_agent_1.
    """
    for i, u in enumerate(utts):
        if u["text"].strip() != "Accept-Deal":
            continue
        # Walk back to the preceding Submit-Deal.
        for j in range(i - 1, -1, -1):
            if utts[j]["text"].strip() == "Submit-Deal":
                sub = utts[j]
                submitter = sub["meta"]["speaker_internal_id"]
                youget = {k: int(v) for k, v in sub["meta"]["issue2youget"].items()}
                theyget = {k: int(v) for k, v in sub["meta"]["issue2theyget"].items()}
                if submitter == "mturk_agent_1":
                    a1, a2 = youget, theyget
                else:
                    a1, a2 = theyget, youget
                you_alloc = [a1.get(issue, 0) for issue in ISSUE_ORDER]
                them_alloc = [a2.get(issue, 0) for issue in ISSUE_ORDER]
                return True, you_alloc, them_alloc
        break
    return False, None, None


def build_turns(utts):
    turns = []
    for u in utts:
        text = u["text"].strip()
        if text in CONTROL_TOKENS:
            continue
        speaker = "you" if u["meta"]["speaker_internal_id"] == "mturk_agent_1" else "them"
        ann = u["meta"].get("annotations")
        annotations = [a for a in ann.split(",") if a] if ann else []
        turns.append({"speaker": speaker, "text": text, "annotations": annotations})
    return turns


def parse_corpus():
    conversations = json.load(open(CORPUS_DIR / "conversations.json"))
    utts_by_conv = load_utterances()
    games = []
    for conv_id, conv in conversations.items():
        pinfo = conv["meta"]["participant_info"]
        a1, a2 = pinfo["mturk_agent_1"], pinfo["mturk_agent_2"]
        utts = utts_by_conv.get(conv_id, [])
        if not utts:
            continue

        you_values = agent_values(a1)
        them_values = agent_values(a2)
        turns = build_turns(utts)
        agreed, you_alloc, them_alloc = extract_deal(utts)

        you_score = them_score = None
        valid_alloc = False
        if agreed:
            valid_alloc = all(you_alloc[i] + them_alloc[i] == UNITS_PER_ITEM for i in range(3))
            you_score = sum(you_alloc[i] * you_values[i] for i in range(3))
            them_score = sum(them_alloc[i] * them_values[i] for i in range(3))

        games.append(
            {
                "dataset": "casino",
                "item_names": ITEM_NAMES,
                "counts": [UNITS_PER_ITEM] * 3,
                "you_values": you_values,
                "them_values": them_values,
                "you_max": SCORE_MAX,
                "them_max": SCORE_MAX,
                "turns": turns,
                "num_turns": len(turns),
                "agreed": agreed,
                "valid_alloc": valid_alloc,
                "you_alloc": you_alloc,
                "them_alloc": them_alloc,
                "you_score": you_score,
                "them_score": them_score,
                "first_speaker": turns[0]["speaker"] if turns else None,
                "meta": {
                    "you": {
                        "reasons": agent_reasons(a1),
                        "points_scored": a1["outcomes"].get("points_scored"),
                        "satisfaction": a1["outcomes"].get("satisfaction"),
                        "likeness": a1["outcomes"].get("opponent_likeness"),
                    },
                    "them": {
                        "reasons": agent_reasons(a2),
                        "points_scored": a2["outcomes"].get("points_scored"),
                        "satisfaction": a2["outcomes"].get("satisfaction"),
                        "likeness": a2["outcomes"].get("opponent_likeness"),
                    },
                },
            }
        )
    return games


def summarize(games):
    n = len(games)
    if n == 0:
        return {}
    agreed = [g for g in games if g["agreed"] and g["valid_alloc"]]
    n_agreed = len(agreed)
    turns_hist = {}
    for g in games:
        turns_hist[g["num_turns"]] = turns_hist.get(g["num_turns"], 0) + 1
    score_hist = {i: 0 for i in range(SCORE_MAX + 1)}
    for g in agreed:
        score_hist[g["you_score"]] += 1
        score_hist[g["them_score"]] += 1
    you_scores = [g["you_score"] for g in agreed]
    them_scores = [g["them_score"] for g in agreed]
    joint = [g["you_score"] + g["them_score"] for g in agreed]

    # Avg self-reported satisfaction across both agents (1-5 scale).
    sats = []
    for g in games:
        for side in ("you", "them"):
            s = g["meta"][side]["satisfaction"]
            if s in SATISFACTION_SCALE:
                sats.append(SATISFACTION_SCALE[s])
    annotated = sum(1 for g in games for t in g["turns"] if t.get("annotations"))
    total_turns = sum(g["num_turns"] for g in games)

    return {
        "num_games": n,
        "num_agreed": n_agreed,
        "agreement_rate": round(n_agreed / n, 4),
        "avg_turns": round(total_turns / n, 2),
        "avg_you_score": round(sum(you_scores) / n_agreed, 2) if n_agreed else 0,
        "avg_them_score": round(sum(them_scores) / n_agreed, 2) if n_agreed else 0,
        "avg_joint_score": round(sum(joint) / n_agreed, 2) if n_agreed else 0,
        "score_max": SCORE_MAX,
        "joint_max": 2 * SCORE_MAX,
        "turns_hist": turns_hist,
        "score_hist": score_hist,
        "extra_cards": [
            {"label": "Avg satisfaction", "value": (f"{round(sum(sats) / len(sats), 2)} / 5" if sats else "n/a")},
            {
                "label": "Strategy-annotated turns",
                "value": f"{round(annotated / total_turns * 100, 1)}%" if total_turns else "0%",
            },
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = parse_corpus()
    stats = summarize(games)
    with open(OUT_DIR / "all.json", "w") as f:
        json.dump({"split": "all", "dataset": "casino", "stats": stats, "games": games}, f)
    print(f"[casino] all: {len(games)} games, agreement {stats['agreement_rate']:.1%}")
    print(f"wrote CaSiNo JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
