"""Prepare negotiation RLVR training data as parquet files for SkyRL.

Converts negotiation scenarios into train/validation parquet datasets consumed
by the SkyRL trainer. Each row encodes one game setup. During training, the
policy plays the "you" (first-mover) side; the environment plays "them".

Usage:
    python prepare_dataset.py [--output_dir DIR] [--dataset dnd|casino] \
        [--train_split SPLIT] [--val_split SPLIT] [--protocol single|dual] \
        [--max_turns N] [--max_train N] [--seed S]
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Allow direct execution from the negotiation package dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402

import datasets  # noqa: E402


def make_row(sc, idx: int, dataset: str, split: str, protocol: str, max_turns: int) -> dict:
    """Build a single RLVR dataset row from a Scenario."""
    item_names = list(sc.item_names)
    counts = list(sc.counts)
    you_values = list(sc.you_values)
    them_values = list(sc.them_values)

    you_system_prompt = prompts.build_system_prompt(
        item_names, counts, you_values, max_turns, protocol=protocol
    )
    them_system_prompt = prompts.build_system_prompt(
        item_names, counts, them_values, max_turns, protocol=protocol
    )

    return {
        "data_source": f"negotiation_{dataset}",
        "env_class": "negotiation",
        "prompt": [
            {"role": "system", "content": you_system_prompt},
            {"role": "user", "content": prompts.OPENING_USER_MSG},
        ],
        "reward_spec": {
            "method": "rule",
            "ground_truth": {
                "item_names": item_names,
                "counts": counts,
                "you_values": you_values,
                "them_values": them_values,
            },
        },
        "extra_info": {
            "dataset": dataset,
            "split": split,
            "index": idx,
            "protocol": protocol,
            "max_turns": max_turns,
            "them_system_prompt": them_system_prompt,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare negotiation RLVR parquet datasets for SkyRL."
    )
    parser.add_argument("--output_dir", default="~/data/fleet/negotiation")
    parser.add_argument("--dataset", default="dnd", choices=["dnd", "casino"])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument(
        "--protocol",
        default="single",
        choices=["single", "dual"],
        help="Negotiation protocol: 'single' (propose/accept) or 'dual' (each submits own deal).",
    )
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Cap on training scenarios (0 = use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for shuffling train scenarios before optional truncation.",
    )
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)

    # --- Load scenarios ---
    train_scenarios = scenarios_mod.load_scenarios(args.dataset, args.train_split)
    rng = random.Random(args.seed)
    rng.shuffle(train_scenarios)
    if args.max_train and args.max_train > 0:
        train_scenarios = train_scenarios[: args.max_train]

    val_scenarios = scenarios_mod.load_scenarios(args.dataset, args.val_split)

    # --- Build rows ---
    train_rows = [
        make_row(sc, idx, args.dataset, args.train_split, args.protocol, args.max_turns)
        for idx, sc in enumerate(train_scenarios)
    ]
    val_rows = [
        make_row(sc, idx, args.dataset, args.val_split, args.protocol, args.max_turns)
        for idx, sc in enumerate(val_scenarios)
    ]

    # --- Write parquet ---
    os.makedirs(output_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_rows)
    val_ds = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(
        f"dataset={args.dataset}  protocol={args.protocol}  max_turns={args.max_turns}\n"
        f"  train rows : {len(train_rows):>5}  -> {train_path}\n"
        f"  val   rows : {len(val_rows):>5}  -> {val_path}"
    )
