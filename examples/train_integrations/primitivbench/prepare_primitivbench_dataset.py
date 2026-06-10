"""
Prepare the PrimitivBench mini-flywheel dataset (Sprint 2).

Does two things:
  1. Copies the portfolio games' game.py from the pilot repo into ./games/
     (self-contained for cluster deployment; env loads from here or
     $PRIMITIVBENCH_GAMES_DIR).
  2. Emits parquet rows — one per (game × seed) — with env_class
     "primitivbench". Per-game `data_source` gives per-game metric splits in
     SkyRL eval for free (per-game Score attribution from a single run).

Arms (per D-15, 2026-06-10; headline = Δ(B−C)):
  A  witness-13 baseline                → existing witness parquet (no work here)
  B  witness-13 + top-12 portfolio      → --portfolio portfolio_v1.json --merge-witness ...
  C  witness-13 + placebo-12 (matched)  → --portfolio portfolio_placebo_v1.json --merge-witness ...
                                          (same --seeds-per-game as B; use --merged-output armC_mixed.parquet)

Usage:
  python3 prepare_primitivbench_dataset.py \
      --portfolio /path/to/05-pilot-study/orchestrator/curated_v2/portfolio_v1.json \
      --games-src /path/to/05-pilot-study/orchestrator/curated_v2/games \
      --seeds-per-game 64 \
      --output data/pb_train.parquet \
      [--merge-witness data/witness_train.parquet --merged-output data/armB_mixed.parquet]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def make_prompt(game_name: str) -> list:
    # Placeholder — PrimitivBenchEnv.init() builds the real chat. Mirrors the
    # witness integration convention.
    return [{"role": "user", "content": f"Start PrimitivBench game {game_name}."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", required=True, help="portfolio json (in-repo portfolios/ or pilot repo)")
    ap.add_argument("--games-src", default=os.path.join(_HERE, "games"),
                    help="curated_v2/games dir in the pilot repo; default = in-repo games/ "
                         "(cluster: games ship pre-vendored, vendoring becomes a no-op)")
    ap.add_argument("--seeds-per-game", type=int, default=64,
                    help="instance multiplication: distinct reset seeds per game")
    ap.add_argument("--seed-offset", type=int, default=1000,
                    help="train seeds start here (proxy battery used 100-204; keep disjoint)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--reward-mode", default="shaped")
    ap.add_argument("--output", default=os.path.join(_HERE, "data", "pb_train.parquet"))
    ap.add_argument("--val-seeds-per-game", type=int, default=8)
    ap.add_argument("--val-output", default=None, help="default: <output dir>/pb_val.parquet")
    ap.add_argument("--merge-witness", default=None, help="existing witness train parquet for arm B")
    ap.add_argument("--merged-output", default=None)
    args = ap.parse_args()

    portfolio = json.load(open(args.portfolio))
    names = [g["game"] for g in portfolio["games"]]

    # 1. vendor the game files (no-op per game when source == destination or the
    #    game ships pre-vendored and the pilot source tree is absent, e.g. cluster)
    games_dst = os.path.join(_HERE, "games")
    os.makedirs(games_dst, exist_ok=True)
    vendored = 0
    for name in names:
        src = os.path.join(args.games_src, name, "game.py")
        dst_dir = os.path.join(games_dst, name)
        if os.path.abspath(os.path.dirname(src)) == os.path.abspath(dst_dir):
            if not os.path.exists(src):
                raise FileNotFoundError(f"{name}: not vendored and no source given ({src})")
            continue
        if not os.path.exists(src):
            if os.path.exists(os.path.join(dst_dir, "game.py")):
                continue
            raise FileNotFoundError(src)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, "game.py"))
        cur = os.path.join(args.games_src, name, "curation.json")
        if os.path.exists(cur):
            shutil.copy(cur, os.path.join(dst_dir, "curation.json"))
        vendored += 1
    print(f"vendored {vendored}/{len(names)} portfolio games → {games_dst}")

    # 2. emit rows
    def rows(seed_lo: int, n_seeds: int) -> list:
        out = []
        for name in names:
            for s in range(seed_lo, seed_lo + n_seeds):
                out.append({
                    "data_source": f"primitivbench/{name}",
                    "prompt": make_prompt(name),
                    "env_class": "primitivbench",
                    "game_name": name,
                    "seed": s,
                    "max_turns": args.max_turns,
                    "reward_mode": args.reward_mode,
                })
        return out

    train = pd.DataFrame(rows(args.seed_offset, args.seeds_per_game))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train.to_parquet(args.output, index=False)
    print(f"train: {len(train)} rows ({len(names)} games × {args.seeds_per_game} seeds) → {args.output}")

    val_output = args.val_output or os.path.join(os.path.dirname(args.output), "pb_val.parquet")
    val = pd.DataFrame(rows(args.seed_offset + args.seeds_per_game, args.val_seeds_per_game))
    val.to_parquet(val_output, index=False)
    print(f"val:   {len(val)} rows → {val_output}")

    # 3. arm B merge
    if args.merge_witness:
        wt = pd.read_parquet(args.merge_witness)
        merged = pd.concat([wt, train], ignore_index=True)
        merged_out = args.merged_output or os.path.join(os.path.dirname(args.output), "armB_mixed.parquet")
        merged.to_parquet(merged_out, index=False)
        print(f"arm B: {len(wt)} witness + {len(train)} pb = {len(merged)} rows → {merged_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
