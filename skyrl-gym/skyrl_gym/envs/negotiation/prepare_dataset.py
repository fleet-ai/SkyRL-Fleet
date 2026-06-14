"""Prepare negotiation RLVR training data as parquet files for SkyRL.

Converts negotiation scenarios into train/validation parquet datasets consumed
by the SkyRL trainer. Each row encodes one game setup. During training, the
policy plays the "you" (first-mover) side; the environment plays "them".

Usage:
    python prepare_dataset.py [--output_dir DIR] [--dataset dnd|casino] \
        [--train_split SPLIT] [--val_split SPLIT] [--protocol single|dual] \
        [--max_turns N] [--max_train N] [--seed S] \
        [--extra_val_dataset dnd|casino] [--extra_val_split SPLIT] \
        [--max_extra_val N]

    Pass --extra_val_dataset to emit an additional held-out eval parquet
    (validation_<dataset>.parquet) from a different dataset, e.g. to measure
    transfer/memorisation when training on dnd and evaluating on casino.
    The trainer logs metrics for each data_source separately, so the extra
    parquet will appear as eval/negotiation_<extra_val_dataset>/*.
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


def make_row(sc, idx: int, dataset: str, split: str, protocol: str, max_turns: int, elicit: str = "none") -> dict:
    """Build a single RLVR dataset row from a Scenario.

    elicit: 'none' | 'two_sided' (mutual ask-and-tell in both prompts) |
            'one_sided' (policy probes only; opponent gets no elicitation block).
    """
    item_names = list(sc.item_names)
    counts = list(sc.counts)
    you_values = list(sc.you_values)
    them_values = list(sc.them_values)

    if elicit == "two_sided":
        you_block, them_block = prompts.PROACTIVE_BLOCK, prompts.PROACTIVE_BLOCK
    elif elicit == "one_sided":
        you_block, them_block = prompts.ASK_ONLY_BLOCK, ""
    else:
        you_block, them_block = "", ""

    you_system_prompt = prompts.build_system_prompt(
        item_names, counts, you_values, max_turns, protocol=protocol, elicit_block=you_block
    )
    them_system_prompt = prompts.build_system_prompt(
        item_names, counts, them_values, max_turns, protocol=protocol, elicit_block=them_block
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
            "elicit": elicit,
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
        "--proactive",
        action="store_true",
        help="Back-compat alias for --elicit two_sided (mutual ask-and-tell in both prompts).",
    )
    parser.add_argument(
        "--elicit",
        default="none",
        choices=["none", "two_sided", "one_sided"],
        help="Preference-elicitation arm: none | two_sided (both ask+tell) | one_sided (policy probes only).",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Cap on training scenarios (0 = use all).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=0,
        help="Cap on validation scenarios (0 = use all). Subsampled with the same seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for shuffling train scenarios before optional truncation.",
    )
    parser.add_argument(
        "--extra_val_dataset",
        default=None,
        choices=["dnd", "casino", "synthetic"],
        help=(
            "Additional held-out dataset to emit as a second eval parquet "
            "(written to validation_<dataset>.parquet; logged separately by "
            "the trainer as eval/negotiation_<dataset>/*). 'synthetic' is "
            "procedurally generated (4-6 items, asymmetric/zero-value/conflict "
            "structure the train set never shows) — fully verifiable, fixed seed."
        ),
    )
    parser.add_argument(
        "--extra_val_split",
        default="all",
        help="Split file of the extra eval dataset (casino only ships all.json).",
    )
    parser.add_argument(
        "--max_extra_val",
        type=int,
        default=0,
        help="Cap on extra eval scenarios (0 = use all; casino has 36 unique scenarios after dedupe).",
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
    if args.max_val and args.max_val > 0:
        rng.shuffle(val_scenarios)
        val_scenarios = val_scenarios[: args.max_val]

    # --- Build rows ---
    elicit_mode = "two_sided" if args.proactive else args.elicit  # --proactive is a back-compat alias
    train_rows = [
        make_row(sc, idx, args.dataset, args.train_split, args.protocol, args.max_turns, elicit_mode)
        for idx, sc in enumerate(train_scenarios)
    ]
    val_rows = [
        make_row(sc, idx, args.dataset, args.val_split, args.protocol, args.max_turns, elicit_mode)
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

    extra_val_rows = []
    extra_val_path = None
    if args.extra_val_dataset is not None:
        extra_val_scenarios = scenarios_mod.load_scenarios(args.extra_val_dataset, args.extra_val_split)
        if args.max_extra_val and args.max_extra_val > 0:
            extra_rng = random.Random(args.seed)
            extra_rng.shuffle(extra_val_scenarios)
            extra_val_scenarios = extra_val_scenarios[: args.max_extra_val]
        extra_val_rows = [
            make_row(sc, idx, args.extra_val_dataset, args.extra_val_split, args.protocol, args.max_turns, elicit_mode)
            for idx, sc in enumerate(extra_val_scenarios)
        ]
        extra_val_path = os.path.join(output_dir, f"validation_{args.extra_val_dataset}.parquet")
        datasets.Dataset.from_list(extra_val_rows).to_parquet(extra_val_path)

    summary = (
        f"dataset={args.dataset}  protocol={args.protocol}  max_turns={args.max_turns}  proactive={args.proactive}\n"
        f"  train rows : {len(train_rows):>5}  -> {train_path}\n"
        f"  val   rows : {len(val_rows):>5}  -> {val_path}"
    )
    if extra_val_path is not None:
        summary += f"\n  extra val  : {len(extra_val_rows):>5}  -> {extra_val_path}"
    print(summary)
