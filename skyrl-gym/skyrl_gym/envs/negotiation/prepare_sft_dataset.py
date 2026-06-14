"""Prepare negotiation SFT warm-start data as parquet files.

Converts human negotiation dialogues (CaSiNo and/or DnD) into a conversational
SFT dataset with assistant-only supervision. Each row encodes one full dialogue
from the perspective of one speaker (the "assistant" side). The downstream trainer
is responsible for masking loss to assistant turns only.

Usage:
    python prepare_sft_dataset.py [--output_dir DIR] [--dataset casino|dnd] \\
        [--both_sides true|false] [--min_turns N] [--max_chars_per_turn N] \\
        [--val_frac F] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets  # noqa: E402

VIZ_DATA = Path(__file__).resolve().parent / "visualizer" / "public" / "data"

# Generic user priming message used when the assistant speaker opens the game.
_PRIMING_USER_MSG = "Let's negotiate. Go ahead with your opening message."

# DnD/CaSiNo artifact patterns to strip from turn text (pre-split by visualizer,
# but keep the regex in case residual tokens survive future re-exports).
_ARTIFACT_RE = re.compile(
    r"(<selection>|<eos>|<disagree>|THEM:|YOU:)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# System prompt builder (lightweight — no structured-tag machinery)
# ---------------------------------------------------------------------------


def build_sft_system_prompt(
    item_names: List[str],
    counts: List[int],
    values: List[int],
) -> str:
    """Build a plain-language negotiation system prompt for SFT training."""
    pool_lines = "\n".join(
        f"  - {c} x {name}" for name, c in zip(item_names, counts)
    )
    value_lines = "\n".join(
        f"  - {name}: {v} points each" for name, v in zip(item_names, values)
    )
    return textwrap.dedent(f"""\
        You are negotiating with another person to divide a shared pool of items between the two of you.

        Pool:
        {pool_lines}

        Your private point values (the other person has their own different, hidden values):
        {value_lines}

        Negotiate by exchanging short, natural messages to reach an agreement on who gets what. \
Aim to maximize your own points while still reaching a deal the other person will accept.""")


# ---------------------------------------------------------------------------
# Turn cleaning / filtering
# ---------------------------------------------------------------------------


def _clean_turn_text(text: str) -> str:
    """Strip whitespace and known artifact tokens from a turn."""
    text = _ARTIFACT_RE.sub("", text)
    return text.strip()


def _is_valid_turn(text: str, max_chars: int) -> bool:
    """Return True if the cleaned turn should be kept."""
    return bool(text) and len(text) <= max_chars


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------


def build_conversation(
    game: dict,
    perspective: str,  # "you" | "them"
    min_turns: int,
    max_chars_per_turn: int,
) -> Optional[List[dict]]:
    """Convert a game dict into an alternating [system, user, assistant, ...] message list.

    Returns None if the game should be dropped for this perspective.
    """
    item_names = game["item_names"]
    counts = game["counts"]
    values = game["you_values"] if perspective == "you" else game["them_values"]
    assistant_speaker = perspective       # "you" or "them"
    user_speaker = "them" if perspective == "you" else "you"

    system_prompt = build_sft_system_prompt(item_names, counts, values)

    # --- Filter and clean raw turns ---
    raw_turns = []
    for t in game["turns"]:
        text = _clean_turn_text(t["text"])
        if not _is_valid_turn(text, max_chars_per_turn):
            continue
        raw_turns.append({"speaker": t["speaker"], "text": text})

    # Drop game if too few real dialogue turns.
    if len(raw_turns) < min_turns:
        return None

    # --- Merge consecutive same-speaker turns ---
    merged: List[dict] = []
    for t in raw_turns:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["text"] += " " + t["text"]
        else:
            merged.append({"speaker": t["speaker"], "text": t["text"]})

    # --- Build alternating user/assistant sequence ---
    # Map speakers to roles.
    role_map = {assistant_speaker: "assistant", user_speaker: "user"}

    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    # If the assistant speaks first, prepend a synthetic user priming turn.
    if merged and merged[0]["speaker"] == assistant_speaker:
        messages.append({"role": "user", "content": _PRIMING_USER_MSG})

    # Walk merged turns, enforcing strict alternation.
    expected_role: Optional[str] = None  # will be set after first real turn
    for t in merged:
        role = role_map[t["speaker"]]
        if expected_role is None:
            expected_role = role
        if role != expected_role:
            # Merge into previous message if roles collide (shouldn't happen
            # after same-speaker merging, but guard defensively).
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += " " + t["text"]
                continue
            # Otherwise skip to maintain alternation.
            continue
        messages.append({"role": role, "content": t["text"]})
        expected_role = "user" if role == "assistant" else "assistant"

    # Ensure at least one assistant turn.
    if not any(m["role"] == "assistant" for m in messages):
        return None

    # The last message must not be a user turn with no following assistant
    # (the trainer can handle it, but trim for cleanliness).
    while messages and messages[-1]["role"] == "user":
        messages.pop()

    # Must still have at least one assistant turn after trimming.
    if not any(m["role"] == "assistant" for m in messages):
        return None

    return messages


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_rows(
    game: dict,
    game_index: int,
    dataset: str,
    both_sides: bool,
    min_turns: int,
    max_chars_per_turn: int,
) -> List[dict]:
    """Build one or two SFT rows for a game (one per perspective)."""
    rows = []
    perspectives = ["you", "them"] if both_sides else ["you"]
    for perspective in perspectives:
        messages = build_conversation(
            game,
            perspective=perspective,
            min_turns=min_turns,
            max_chars_per_turn=max_chars_per_turn,
        )
        if messages is None:
            continue
        rows.append(
            {
                "messages": messages,
                "data_source": f"sft_{dataset}",
                "extra_info": {
                    "dataset": dataset,
                    "perspective": perspective,
                    "game_index": game_index,
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare negotiation SFT warm-start parquet datasets."
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/fleet/negotiation_sft",
        help="Directory to write train.parquet (and validation.parquet if --val_frac > 0).",
    )
    parser.add_argument(
        "--dataset",
        default="casino",
        choices=["casino", "dnd"],
        help="Source dataset to process.",
    )
    parser.add_argument(
        "--both_sides",
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
        help="Emit both 'you' and 'them' perspective rows per game (default: true).",
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        default=2,
        help="Minimum number of real dialogue turns; games with fewer are dropped (default: 2).",
    )
    parser.add_argument(
        "--max_chars_per_turn",
        type=int,
        default=2000,
        help="Drop turns longer than this many characters (default: 2000).",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.0,
        help="Fraction of games to hold out as validation.parquet (default: 0.0 = no val split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for shuffling / val split (default: 1).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val_frac must be in [0.0, 1.0)")

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load games ---
    # CaSiNo ships one file (all.json); DnD ships train.json.
    split = "all" if args.dataset == "casino" else "train"
    data_path = VIZ_DATA / args.dataset / f"{split}.json"
    if not data_path.exists():
        sys.exit(
            f"ERROR: data file not found at {data_path}.\n"
            "Ensure the visualizer data has been built (visualizer/build.py)."
        )

    with open(data_path) as f:
        raw = json.load(f)
    all_games = raw["games"]
    print(f"Loaded {len(all_games)} games from {data_path}")

    # --- Shuffle and split into train/val game lists ---
    rng = random.Random(args.seed)
    indices = list(range(len(all_games)))
    rng.shuffle(indices)

    if args.val_frac > 0.0:
        n_val = max(1, int(len(indices) * args.val_frac))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
    else:
        val_indices = []
        train_indices = indices

    # --- Build rows ---
    def _build_rows_for_split(game_indices: List[int]) -> List[dict]:
        rows = []
        for original_idx in game_indices:
            game = all_games[original_idx]
            rows.extend(
                build_rows(
                    game=game,
                    game_index=original_idx,
                    dataset=args.dataset,
                    both_sides=args.both_sides,
                    min_turns=args.min_turns,
                    max_chars_per_turn=args.max_chars_per_turn,
                )
            )
        return rows

    train_rows = _build_rows_for_split(train_indices)
    val_rows = _build_rows_for_split(val_indices) if val_indices else []

    # --- Write parquet ---
    train_path = os.path.join(output_dir, "train.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)

    val_path = None
    if val_rows:
        val_path = os.path.join(output_dir, "validation.parquet")
        datasets.Dataset.from_list(val_rows).to_parquet(val_path)

    # --- Summary ---
    print(
        f"\n{'='*60}\n"
        f"dataset          : {args.dataset}\n"
        f"source file      : {data_path}\n"
        f"games loaded     : {len(all_games)}\n"
        f"both_sides       : {args.both_sides}\n"
        f"min_turns        : {args.min_turns}\n"
        f"max_chars_per_turn: {args.max_chars_per_turn}\n"
        f"val_frac         : {args.val_frac}\n"
        f"seed             : {args.seed}\n"
        f"{'='*60}\n"
        f"train rows       : {len(train_rows):>6}  -> {train_path}"
    )
    if val_path:
        print(f"val   rows       : {len(val_rows):>6}  -> {val_path}")
    print(f"{'='*60}")

    # --- Print two example conversations (truncated) ---
    print("\n--- Example conversations (truncated) ---\n")
    for i, row in enumerate(train_rows[:2]):
        print(f"[Example {i+1}]  data_source={row['data_source']}  "
              f"extra_info={row['extra_info']}")
        for msg in row["messages"]:
            snippet = msg["content"][:200].replace("\n", " ")
            ellipsis = "..." if len(msg["content"]) > 200 else ""
            print(f"  [{msg['role']:9s}] {snippet}{ellipsis}")
        print()


if __name__ == "__main__":
    main()
