#!/usr/bin/env python3
"""Build the JSON payload for the SFT-dataset viewer webapp.

Reads the SFT parquet datasets produced by `prepare_sft_dataset.py` (the exact
rows that will be fed to `negotiation_sft_trainer.py`), computes corpus-level
stats over ALL rows, samples a subset of conversations for display, and writes
one JSON file per dataset plus a manifest into `public/data/`.

Each parquet row is a conversation:
    {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...],
     "data_source": "sft_<dataset>",
     "extra_info": {"dataset", "perspective", "game_index"}}

Supervision is on ASSISTANT turns only (that is what the trainer masks), so the
viewer flags assistant messages as "supervised".

Usage:
    python build.py                      # regenerate parquets if missing, then build
    python build.py --sample 500         # cap displayed conversations per dataset
    python build.py --casino_parquet P --dnd_parquet P   # use explicit parquet paths
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
NEG_DIR = HERE.parent  # .../envs/negotiation
PREP = NEG_DIR / "prepare_sft_dataset.py"
OUT_DIR = HERE / "public" / "data"

DEFAULT_PARQUETS = {
    "casino": os.path.expanduser("~/data/fleet/negotiation_sft_casino/train.parquet"),
    "dnd": os.path.expanduser("~/data/fleet/negotiation_sft_dnd/train.parquet"),
}

DATASET_LABEL = {
    "casino": "CaSiNo (campsite: food / water / firewood)",
    "dnd": "Deal or No Deal (books / hats / balls)",
}


def ensure_parquet(dataset: str, path: str) -> str:
    """Generate the SFT parquet via prepare_sft_dataset.py if it does not exist."""
    if os.path.exists(path):
        return path
    out_dir = os.path.dirname(path)
    print(f"[build] {dataset} parquet missing; generating -> {out_dir}")
    subprocess.run(
        [
            sys.executable,
            str(PREP),
            "--dataset",
            dataset,
            "--output_dir",
            out_dir,
            "--both_sides",
            "true",
        ],
        check=True,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"prepare_sft_dataset.py did not write {path}")
    return path


def row_to_conv(row: dict) -> dict:
    msgs = row.get("messages") or []
    # pyarrow returns list[dict]; normalise to plain {role, content, supervised}
    out_msgs = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        out_msgs.append(
            {"role": role, "content": content, "supervised": role == "assistant"}
        )
    info = row.get("extra_info") or {}
    return {
        "perspective": info.get("perspective", "?"),
        "game_index": info.get("game_index", -1),
        "messages": out_msgs,
    }


def compute_stats(convs: list[dict]) -> dict:
    n = len(convs)
    persp = Counter(c["perspective"] for c in convs)
    turns_hist = Counter()
    tot_msgs = tot_asst = tot_user = 0
    tot_sup_chars = tot_sup_words = 0
    sup_turn_lens = []  # words per supervised (assistant) turn
    for c in convs:
        asst = [m for m in c["messages"] if m["role"] == "assistant"]
        user = [m for m in c["messages"] if m["role"] == "user"]
        tot_msgs += len(c["messages"])
        tot_asst += len(asst)
        tot_user += len(user)
        turns_hist[len(asst)] += 1
        for m in asst:
            w = len(m["content"].split())
            tot_sup_chars += len(m["content"])
            tot_sup_words += w
            sup_turn_lens.append(w)

    def avg(x):
        return round(x / n, 2) if n else 0

    sup_turn_lens.sort()

    def pct(p):
        if not sup_turn_lens:
            return 0
        i = min(len(sup_turn_lens) - 1, int(p * len(sup_turn_lens)))
        return sup_turn_lens[i]

    return {
        "n_rows": n,
        "n_games": max((c["game_index"] for c in convs), default=-1) + 1,
        "perspectives": dict(persp),
        "avg_msgs": avg(tot_msgs),
        "avg_assistant_msgs": avg(tot_asst),
        "avg_user_msgs": avg(tot_user),
        "avg_supervised_chars": avg(tot_sup_chars),
        "avg_supervised_words": avg(tot_sup_words),
        "supervised_turn_word_p50": pct(0.50),
        "supervised_turn_word_p90": pct(0.90),
        # histogram of assistant (supervised) turns per conversation, capped bucket 12+
        "assistant_turns_hist": _bucket_hist(turns_hist, cap=12),
    }


def _bucket_hist(counter: Counter, cap: int) -> dict:
    out = {}
    for k, v in counter.items():
        key = str(k) if k < cap else f"{cap}+"
        out[key] = out.get(key, 0) + v
    # ordered by numeric key
    return dict(
        sorted(out.items(), key=lambda kv: (kv[0].endswith("+"), int(kv[0].rstrip("+"))))
    )


def build_dataset(dataset: str, parquet_path: str, sample: int, seed: int) -> dict:
    print(f"[build] reading {dataset}: {parquet_path}")
    rows = pq.read_table(parquet_path).to_pylist()
    convs = [row_to_conv(r) for r in rows]
    stats = compute_stats(convs)

    rng = random.Random(seed)
    if sample and sample < len(convs):
        display = rng.sample(convs, sample)
        # keep a stable, readable order: by game_index then perspective
        display.sort(key=lambda c: (c["game_index"], c["perspective"]))
        truncated = True
    else:
        display = sorted(convs, key=lambda c: (c["game_index"], c["perspective"]))
        truncated = False

    payload = {
        "dataset": dataset,
        "label": DATASET_LABEL.get(dataset, dataset),
        "stats": stats,
        "displayed": len(display),
        "truncated": truncated,
        "conversations": display,
    }
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casino_parquet", default=DEFAULT_PARQUETS["casino"])
    ap.add_argument("--dnd_parquet", default=DEFAULT_PARQUETS["dnd"])
    ap.add_argument(
        "--sample",
        type=int,
        default=500,
        help="Max conversations to embed per dataset for display (0 = all).",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = {"casino": args.casino_parquet, "dnd": args.dnd_parquet}
    manifest = {"datasets": []}
    for ds, path in sources.items():
        path = ensure_parquet(ds, path)
        payload = build_dataset(ds, path, args.sample, args.seed)
        out_path = OUT_DIR / f"{ds}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)
        size_kb = out_path.stat().st_size / 1024
        manifest["datasets"].append(
            {
                "id": ds,
                "label": payload["label"],
                "n_rows": payload["stats"]["n_rows"],
                "displayed": payload["displayed"],
            }
        )
        print(
            f"[build] wrote {out_path.name}  rows={payload['stats']['n_rows']}  "
            f"displayed={payload['displayed']}  ({size_kb:.0f} KB)"
        )

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build] wrote manifest.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
