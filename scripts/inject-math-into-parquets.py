"""Inject math-QA sandwich into the user message of fleet density training prompts.

Reads existing ${DATA_DIR}/train.parquet (built by prepare_dataset) and rewrites
each row's prompt[0]['content'] to:

  For reference:

  <math QA start, deterministic from hash(task_key, "start") into 220 pool>

  Focus on the task — this is your main objective:

  <original task query>

  For reference:

  <math QA end, deterministic from hash(task_key, "end") into 220 pool>

Eval (validation.parquet) is left UNTOUCHED so eval prompts match the
baseline density runs — apples-to-apples comparison on Fleet pass@3.

Single-variable change vs baseline density: only the user message of
training rows changes. System prompt (tool defs, persona, env context)
is built by the env at inference time and is untouched.

Math pool: the 220 problems Qwen3.5-9B failed on HMMT Feb 25 / HMMT Nov 25 /
PolyMath EN. See s3://skyrl-trajectories/math-injection/qwen35_9b_failed/v1/.

Required env vars: DATA_DIR (where train/validation.parquet live), AWS keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

BLOCKS_S3 = "s3://skyrl-trajectories/math-injection/qwen35_9b_failed/v1/math_injection_blocks_clean.jsonl"
LOCAL_BLOCKS = "/tmp/math_injection_blocks.jsonl"

FOCUS_LINE = "Focus on the task — this is your main objective:"


def fetch_blocks() -> list[str]:
    Path(LOCAL_BLOCKS).parent.mkdir(parents=True, exist_ok=True)
    if not Path(LOCAL_BLOCKS).exists():
        subprocess.run(["aws", "s3", "cp", BLOCKS_S3, LOCAL_BLOCKS], check=True)
    blocks = [json.loads(l)["block"] for l in open(LOCAL_BLOCKS)]
    if len(blocks) != 220:
        print(f"WARN: expected 220 blocks, got {len(blocks)}", file=sys.stderr)
    return blocks


def pick_pair(task_key: str, n: int) -> tuple[int, int]:
    """Hash task_key to two distinct indices in [0, n)."""
    start = int(hashlib.md5((task_key + "|start").encode()).hexdigest(), 16) % n
    end = int(hashlib.md5((task_key + "|end").encode()).hexdigest(), 16) % n
    if end == start:
        end = (end + 1) % n
    return start, end


def sandwich(original: str, blk_start: str, blk_end: str) -> str:
    return (
        f"For reference:\n\n{blk_start}\n\n"
        f"{FOCUS_LINE}\n\n"
        f"{original}\n\n"
        f"For reference:\n\n{blk_end}"
    )


def inject_row(row: dict, blocks: list[str]) -> dict:
    prompt = row["prompt"]
    # prompt is a numpy array of dicts from parquet — normalize to list
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    original_user_msg = prompt[0]["content"]
    s, e = pick_pair(row["task_key"], len(blocks))
    new_content = sandwich(original_user_msg, blocks[s], blocks[e])
    new_prompt = [{"role": "user", "content": new_content}]
    row["prompt"] = new_prompt
    return row


def transform_parquet(path: Path, blocks: list[str]) -> tuple[int, int]:
    df = pd.read_parquet(path)
    n_before = len(df)
    df = df.apply(lambda r: pd.Series(inject_row(r.to_dict(), blocks)), axis=1)
    df.to_parquet(path)
    return n_before, len(df)


def main() -> None:
    data_dir = Path(os.environ["DATA_DIR"])
    print(f"DATA_DIR = {data_dir}")

    blocks = fetch_blocks()
    print(f"loaded {len(blocks)} math blocks")

    # Only transform train.parquet — validation.parquet stays clean so eval
    # matches the baseline density runs (apples-to-apples Fleet pass@3).
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found", file=sys.stderr)
        sys.exit(1)
    n_before, n_after = transform_parquet(train_path, blocks)
    print(f"transformed train.parquet: {n_before} -> {n_after} rows")
    print("validation.parquet intentionally untouched (eval stays clean)")

    # Echo a sample for sanity
    sample = pd.read_parquet(data_dir / "train.parquet").iloc[0]
    head = sample["prompt"][0]["content"][:400]
    print(f"\n=== sample (first 400 chars of train.parquet row 0 user content) ===\n{head}\n=== /sample ===")


if __name__ == "__main__":
    main()
