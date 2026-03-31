"""
Prepare training dataset for Witness GRPO training.

Generates a parquet file with prompts for each game/level combination.
Each row contains a system prompt + initial observation that the
SkyRLGymGenerator will use to start an episode.

Usage:
  python examples/train_integrations/witness/prepare_witness_dataset.py \
    --game_ids tw10 tw04 tw07 \
    --reward_mode shaped \
    --output_dir $HOME/data/witness
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _get_witness_repo() -> str:
    return os.environ.get(
        "WITNESS_ENVS_DIR",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "arc-witness-envs")),
    )


def _load_baselines(witness_repo: str, game_id: str) -> list:
    meta_path = os.path.join(witness_repo, "environment_files", game_id, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("baseline_actions", [])
    return []


def make_prompt(game_id: str, total_levels: int) -> list:
    """Create the initial prompt for a game."""
    return [
        {
            "role": "user",
            "content": (
                f"You are starting puzzle game '{game_id}' "
                f"with {total_levels} levels. "
                f"Explore the grid, discover the rules, and solve all levels. "
                f"Respond with your action: <action>NUMBER</action>"
            ),
        }
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_ids", nargs="+", default=["tw10"],
                        help="Games to include (e.g., tw10 tw04 tw07)")
    parser.add_argument("--reward_mode", default="shaped",
                        choices=["sparse", "shaped", "arc_score"])
    parser.add_argument("--max_levels", type=int, default=5,
                        help="Max levels per game (default 5)")
    parser.add_argument("--obs_mode", default="grid",
                        choices=["grid", "ascii"])
    parser.add_argument("--rules_mode", default="rules_unknown",
                        choices=["rules_given", "rules_unknown"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    witness_repo = _get_witness_repo()
    if witness_repo not in sys.path:
        sys.path.insert(0, witness_repo)

    rows = []
    for game_id in args.game_ids:
        baselines = _load_baselines(witness_repo, game_id)
        total_levels = len(baselines) if baselines else 5
        total_levels = min(total_levels, args.max_levels)

        # One row per game: each episode plays from level 0 through max_levels
        rows.append({
            "data_source": "witness",
            "prompt": make_prompt(game_id, total_levels),
            "env_class": "witness",
            "game_id": game_id,
            "seed": args.seed,
            "reward_mode": args.reward_mode,
            "obs_mode": args.obs_mode,
            "rules_mode": args.rules_mode,
            "max_levels": total_levels,
            "max_steps_multiplier": 3,
        })

    # For online RL, train and eval use the same games.
    train_rows = rows.copy()
    val_rows = rows.copy()

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Created {len(train_rows)} train + {len(val_rows)} val samples")
    print(f"  Games: {args.game_ids}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Obs mode: {args.obs_mode}")
    print(f"  Rules mode: {args.rules_mode}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")


if __name__ == "__main__":
    main()
