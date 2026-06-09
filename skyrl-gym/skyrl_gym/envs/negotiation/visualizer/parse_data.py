#!/usr/bin/env python3
"""Parse the FAIR "Deal or No Deal" negotiation dataset into JSON for the web visualizer.

Source: https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate

Raw line format (one perspective of one game per line):
    <input> c0 v0 c1 v1 c2 v2 </input>
    <dialogue> YOU: ... <eos> THEM: ... <eos> ... <selection> </dialogue>
    <output> item0=a item1=b item2=c item0=d item1=e item2=f </output>
    <partner_input> c0 v0' c1 v1' c2 v2' </partner_input>

Items are: 0=book, 1=hat, 2=ball. Counts are shared; values are private and sum to 10 per agent.
Each game appears twice (once per perspective) on consecutive lines; we dedupe into a single game.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "public" / "data" / "dnd"

ITEM_NAMES = ["book", "hat", "ball"]
SCORE_MAX = 10  # values sum to 10 per agent


def _between(line: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", line)
    return m.group(1).strip() if m else ""


def parse_input(text: str):
    """'c0 v0 c1 v1 c2 v2' -> (counts[3], values[3])."""
    nums = [int(x) for x in text.split()]
    counts = nums[0::2]
    values = nums[1::2]
    return counts, values


def parse_dialogue(text: str):
    """Split into ordered turns: [{speaker, text}]. Drops the trailing <selection>."""
    body = text.replace("<selection>", "").strip()
    turns = []
    for chunk in body.split("<eos>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("YOU:"):
            speaker, msg = "you", chunk[len("YOU:"):].strip()
        elif chunk.startswith("THEM:"):
            speaker, msg = "them", chunk[len("THEM:"):].strip()
        else:
            speaker, msg = "them", chunk.strip()
        if msg:
            turns.append({"speaker": speaker, "text": msg})
    return turns


def parse_output(text: str):
    """Returns (agreed: bool, you_alloc[3], them_alloc[3])."""
    if "item0=" not in text:
        return False, None, None
    nums = re.findall(r"item\d+=(\d+)", text)
    if len(nums) < 6:
        return False, None, None
    nums = [int(x) for x in nums[:6]]
    return True, nums[0:3], nums[3:6]


def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    counts, you_values = parse_input(_between(line, "input"))
    _, them_values = parse_input(_between(line, "partner_input"))
    turns = parse_dialogue(_between(line, "dialogue"))
    agreed, you_alloc, them_alloc = parse_output(_between(line, "output"))

    you_score = them_score = None
    valid_alloc = False
    if agreed and you_alloc and them_alloc:
        # An allocation is consistent if the two sides exactly partition every item pool.
        valid_alloc = all(you_alloc[i] + them_alloc[i] == counts[i] for i in range(3))
        you_score = sum(you_alloc[i] * you_values[i] for i in range(3))
        them_score = sum(them_alloc[i] * them_values[i] for i in range(3))

    # Who speaks first determines the canonical "first mover".
    first_speaker = turns[0]["speaker"] if turns else None

    return {
        "dataset": "dnd",
        "item_names": ITEM_NAMES,
        "counts": counts,
        "you_values": you_values,
        "them_values": them_values,
        "you_max": sum(counts[i] * you_values[i] for i in range(3)),
        "them_max": sum(counts[i] * them_values[i] for i in range(3)),
        "turns": turns,
        "num_turns": len(turns),
        "agreed": agreed,
        "valid_alloc": valid_alloc,
        "you_alloc": you_alloc,
        "them_alloc": them_alloc,
        "you_score": you_score,
        "them_score": them_score,
        "first_speaker": first_speaker,
    }


def dedupe_key(game):
    """Two perspectives of the same game share a sorted signature."""
    persp = (tuple(game["counts"]), tuple(game["you_values"]), tuple(game["them_values"]))
    swapped = (tuple(game["counts"]), tuple(game["them_values"]), tuple(game["you_values"]))
    return tuple(sorted([persp, swapped])) + (game["num_turns"],)


def parse_file(path: Path):
    games = []
    seen = set()
    with open(path) as f:
        for line in f:
            g = parse_line(line)
            if g is None:
                continue
            # Prefer the perspective where "you" moves first for a stable view.
            key = dedupe_key(g)
            if key in seen:
                continue
            seen.add(key)
            games.append(g)
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
    you_scores = [g["you_score"] for g in agreed]
    them_scores = [g["them_score"] for g in agreed]
    joint = [g["you_score"] + g["them_score"] for g in agreed]
    pareto_count = 0
    score_hist = {i: 0 for i in range(11)}
    for g in agreed:
        score_hist[g["you_score"]] = score_hist.get(g["you_score"], 0) + 1
        score_hist[g["them_score"]] = score_hist.get(g["them_score"], 0) + 1
        if g["you_score"] + g["them_score"] >= 10:
            pareto_count += 1
    return {
        "num_games": n,
        "num_agreed": n_agreed,
        "agreement_rate": round(n_agreed / n, 4),
        "avg_turns": round(sum(g["num_turns"] for g in games) / n, 2),
        "avg_you_score": round(sum(you_scores) / n_agreed, 2) if n_agreed else 0,
        "avg_them_score": round(sum(them_scores) / n_agreed, 2) if n_agreed else 0,
        "avg_joint_score": round(sum(joint) / n_agreed, 2) if n_agreed else 0,
        "score_max": SCORE_MAX,
        "joint_max": 2 * SCORE_MAX,
        "turns_hist": turns_hist,
        "score_hist": score_hist,
        "extra_cards": [
            {"label": "Efficient deals", "value": f"{round(pareto_count / n_agreed * 100, 1)}%" if n_agreed else "0%"},
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.txt"
        if not path.exists():
            print(f"skip {split}: {path} missing")
            continue
        games = parse_file(path)
        stats = summarize(games)
        with open(OUT_DIR / f"{split}.json", "w") as f:
            json.dump({"split": split, "dataset": "dnd", "stats": stats, "games": games}, f)
        print(f"[dnd] {split}: {len(games)} games, agreement {stats['agreement_rate']:.1%}")
    print(f"wrote DND JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
