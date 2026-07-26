"""Prepare leaky-poker self-play RLVR data as parquet for SkyRL.

Each row is one heads-up match SEED. The policy plays the hero seat; LeakyPokerEnv deals the match
(seed -> deck) at init and drives the villain (a served policy snapshot). The reward is verifiable
+ dense (leaky_reward), so there is no natural-language ground truth to bake in -- the row just
carries the match config in reward_spec.ground_truth.

Usage:
  python prepare_dataset.py --output_dir ~/data/fleet/leaky_poker \
      --n_train 4096 --n_val 256 --num_rounds 4
"""
import argparse
from pathlib import Path

import datasets

from skyrl_gym.envs.leaky_poker.env import POKER_SYSTEM


def make_row(idx, split, seed, num_rounds, starting_chips, sb, bb, eval_opponent=None):
    # Eval is pinned to a STATIONARY opponent (the frozen exploiter) so win-rate is interpretable
    # regardless of the training arm; extra_info.opponent_mode overrides env_config in the env.
    extra = {"split": split, "index": idx, "seed": int(seed), "max_turns": 4 * num_rounds * 8}
    if eval_opponent:
        extra["opponent_mode"] = eval_opponent
    return {
        "data_source": "leaky_poker",
        "env_class": "leaky_poker",
        # The env replaces the user turn at init() with the actual first board state (the deal
        # depends on the seed), so a placeholder opener is fine here.
        "prompt": [
            {"role": "system", "content": POKER_SYSTEM},
            {"role": "user", "content": "A new heads-up match begins. Wait for the board."},
        ],
        "reward_spec": {
            "method": "rule",
            "ground_truth": {
                "seed": int(seed),
                "num_rounds": int(num_rounds),
                "starting_chips": int(starting_chips),
                "small_blind": int(sb),
                "big_blind": int(bb),
                # hero alternates seat by parity for position balance
                "hero": int(idx % 2),
            },
        },
        "extra_info": extra,
    }


def build(n, split, base_seed, args, eval_opponent=None):
    return [make_row(i, split, base_seed + i, args.num_rounds, args.starting_chips,
                     args.small_blind, args.big_blind, eval_opponent) for i in range(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="~/data/fleet/leaky_poker")
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--num_rounds", type=int, default=4)
    ap.add_argument("--starting_chips", type=int, default=1000)
    ap.add_argument("--small_blind", type=int, default=10)
    ap.add_argument("--big_blind", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--eval_opponent", default="exploiter",
                    help="opponent_mode pinned on val rows for a stationary eval (exploiter|scripted|'' to disable)")
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train = build(args.n_train, "train", args.seed, args)
    val = build(args.n_val, "val", args.seed + 10_000_000, args, eval_opponent=(args.eval_opponent or None))
    tp, vp = out / "train.parquet", out / "validation.parquet"
    datasets.Dataset.from_list(train).to_parquet(str(tp))
    datasets.Dataset.from_list(val).to_parquet(str(vp))
    print(f"wrote {len(train)} train -> {tp}\nwrote {len(val)} val -> {vp}")
