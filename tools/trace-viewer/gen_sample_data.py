#!/usr/bin/env python3
"""Synthesize a realistic multi-step negotiation run for the trace viewer demo.

Writes JSONL files matching the on-disk schema produced by
``skyrl.train.utils.trainer_utils.dump_training_trajectories`` — one file per
training step at ``public/data/<run>/global_step_<N>.jsonl`` — so the viewer can be
exercised without a live training run. The same scenarios recur across every step
(as with a fixed dataset), which is what powers the "Track one prompt" view.

This is demo data only. For real runs use ``build_manifest.py``.
"""
import argparse
import json
import random
from pathlib import Path

# A handful of Deal-or-No-Deal style scenarios reused across every step.
SCENARIOS = [
    {"id": "dnd_0", "items": ["book", "hat", "ball"], "counts": [1, 2, 3],
     "you": [4, 2, 0], "them": [1, 1, 2]},
    {"id": "dnd_1", "items": ["book", "hat", "ball"], "counts": [3, 1, 1],
     "you": [1, 3, 4], "them": [2, 2, 2]},
    {"id": "dnd_2", "items": ["book", "hat", "ball"], "counts": [2, 2, 2],
     "you": [3, 1, 1], "them": [0, 3, 2]},
    {"id": "dnd_3", "items": ["book", "hat", "ball"], "counts": [1, 4, 1],
     "you": [6, 1, 0], "them": [2, 1, 3]},
]

SYS = ("You are a skilled negotiator playing a item-division game. Split the pool to "
       "maximize YOUR total value. End your offer with a <propose>{json}</propose> tag "
       "stating what you keep, or <accept> to accept the opponent's offer.")


def user_prompt(sc):
    pool = ", ".join(f"{c} {n}" for c, n in zip(sc["counts"], sc["items"]))
    vals = ", ".join(f"{n}={v}" for n, v in zip(sc["items"], sc["you"]))
    return (f"Pool: {pool}. Your private values: {vals}. The opponent has different, "
            f"hidden values. Make your opening offer.")


def chatml_prompt(sc):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt(sc)}<|im_end|>\n<|im_start|>assistant\n")


def gen_trajectory(sc, step, total_steps, rng, thinking):
    """Produce one (text, reward, turns, stop_reason) sample.

    Later steps: the policy learns to reach agreements faster with higher self-score.
    """
    frac = step / max(1, total_steps - 1)
    agree_p = 0.15 + 0.7 * frac          # agreement rate climbs over training
    greedy = 0.55 + 0.35 * frac          # learns to claim its high-value items
    agreed = rng.random() < agree_p

    you_take = [int(round(c * greedy)) if v >= max(sc["you"]) else int(round(c * (1 - greedy) * 0.5))
                for c, v in zip(sc["counts"], sc["you"])]
    you_take = [max(0, min(c, x)) for c, x in zip(sc["counts"], you_take)]
    you_score = sum(t * v for t, v in zip(you_take, sc["you"]))
    you_max = sum(c * v for c, v in zip(sc["counts"], sc["you"])) or 1

    def think(s):
        return f"<think>{s}</think>\n" if thinking else ""

    propose = json.dumps({n: t for n, t in zip(sc["items"], you_take)})
    turns_text = []
    # opening offer
    turns_text.append(
        f"{think('book is my highest-value item; anchor hard and keep it.')}"
        f"I value the {sc['items'][0]} most. I propose I keep the high-value pieces and "
        f"you take the rest.\n<propose>{propose}</propose>")
    # opponent reply
    turns_text.append("<|im_end|>\n<|im_start|>user\nThat's lopsided. I counter: we split the "
                      "balls evenly and I keep a hat.<|im_end|>\n<|im_start|>assistant\n")
    if agreed:
        turns_text.append(f"{think('their counter still leaves me my book; good enough.')}"
                          f"Deal — that works for me.\n<accept>")
        stop_reason = "stop"
        reward = round(you_score / you_max, 3)
        turns = 2
    else:
        # drags on / never converges
        turns_text.append(f"{think('hold firm, do not concede the book.')}"
                          f"I can't accept that, the {sc['items'][0]} is non-negotiable.\n"
                          f"<propose>{propose}</propose>")
        if rng.random() < 0.5:
            stop_reason = "length"          # ran out of turns/tokens
            reward = round(0.05 + 0.15 * rng.random(), 3)
        else:
            stop_reason = "stop"            # no_deal
            reward = round(-0.1 + 0.1 * rng.random(), 3)
        turns = 3
    text = "".join(turns_text)
    tokens = 60 + len(text) // 4 + rng.randint(0, 40)
    return text, reward, turns, stop_reason, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "public/data/sample-negotiation-run"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thinking", action="store_true", help="include <think> blocks")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("global_step_*.jsonl"):
        f.unlink()

    ts0 = 1_750_000_000.0
    for s in range(0, args.steps * 5, 5):          # steps 0,5,10,... like real ckpt cadence
        idx = s // 5
        path = out / f"global_step_{s}.jsonl"
        with open(path, "w") as fh:
            for sc in SCENARIOS:
                for _ in range(args.samples_per_prompt):
                    text, reward, turns, stop, tokens = gen_trajectory(
                        sc, idx, args.steps, rng, args.thinking)
                    entry = {
                        "step": s,
                        "env_key": "negotiation",
                        "data_source": "dnd",
                        "stop_reason": stop,
                        "reward": reward,
                        "turns": turns,
                        "tokens": tokens,
                        "prompt": chatml_prompt(sc),
                        "text": text,
                        "timestamp": ts0 + s * 600,
                    }
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"wrote {path}")
    print(f"\nDone. {args.steps} steps in {out}")


if __name__ == "__main__":
    main()
