"""Prepare the Hendrycks MATH dataset (levels 3-5) for SkyRL training.

Produces ``train.parquet`` / ``validation.parquet`` in the *exact* schema used
by the DAPO/AIME math parquets so the rows can be graded by the existing
``aime`` env (``skyrl_gym.envs.aime``), which performs Hendrycks-MATH answer
matching (boxed / ``Answer:`` extraction + normalization). No new env code is
needed.

Source dataset: ``DigitalLearningGmbH/MATH-lighteval`` -- the community-
maintained, ungated, drop-in replacement for the original (DMCA'd)
``hendrycks/competition_math``. Same 7,500 train / 5,000 test problems, same
``level`` ("Level 1".."Level 5") and ``type`` fields, answer boxed in the
``solution``. Override with ``--dataset-name`` if you have a different mirror.

Each output row:
    {
      "data_source": <dataset name>,
      "prompt": [{"role": "user", "content": <templated problem>}],
      "env_class": "aime",                         # Hendrycks-MATH grader
      "ability": "MATH",
      "reward_model": {"ground_truth": <answer>, "style": "rule-lighteval/MATH_v2"},
      "extra_info": {"index", "raw_problem", "level", "type", "split"},
    }

The prompt template is identical to the DAPO/AIME parquets so the model is asked
to end with ``Answer: $Answer`` -- which is what the ``aime`` grader extracts.

Usage:
    bash examples/train/math/prepare_math_data.sh
    # or directly:
    uv run --isolated --extra fsdp -m examples.train.math.math_dataset \
        --output-dir ~/data/math --levels 3 4 5
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import datasets


# Identical to the DAPO/AIME parquet prompt so the `aime` grader's `Answer:`
# extraction pattern matches the model output.
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) substring, or None."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        idx = string.rfind("\\fbox{")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Strip the ``\\boxed{...}`` / ``\\fbox{...}`` wrapper, returning the content."""
    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left) : -1]
    return s


def extract_answer(solution: str) -> Optional[str]:
    """Pull the final boxed answer out of a MATH solution string."""
    boxed = last_boxed_only_string(solution or "")
    if boxed is None:
        return None
    answer = remove_boxed(boxed).strip()
    return answer or None


def _level_int(level: str) -> Optional[int]:
    """'Level 3' -> 3; tolerant of plain ints / missing values."""
    if level is None:
        return None
    s = str(level).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def build_split(
    ds: "datasets.Dataset",
    split_name: str,
    data_source: str,
    keep_levels: set,
) -> "datasets.Dataset":
    rows = []
    dropped_level = 0
    dropped_no_answer = 0
    for i, ex in enumerate(ds):
        lvl = _level_int(ex.get("level"))
        if keep_levels and lvl not in keep_levels:
            dropped_level += 1
            continue
        problem = ex.get("problem", "")
        gt = extract_answer(ex.get("solution", ""))
        if not gt:
            dropped_no_answer += 1
            continue
        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}
                ],
                "env_class": "aime",
                "ability": "MATH",
                "reward_model": {"ground_truth": gt, "style": "rule-lighteval/MATH_v2"},
                "extra_info": {
                    "index": i,
                    "raw_problem": problem,
                    "level": lvl,
                    "type": ex.get("type"),
                    "split": split_name,
                },
            }
        )
    print(
        f"[{split_name}] kept {len(rows)} | dropped {dropped_level} (level filter) "
        f"+ {dropped_no_answer} (no boxed answer)"
    )
    return datasets.Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Hendrycks MATH levels 3-5 for SkyRL")
    p.add_argument("--output-dir", default="~/data/math")
    p.add_argument(
        "--dataset-name",
        default="DigitalLearningGmbH/MATH-lighteval",
        help="HF dataset id (ungated Hendrycks MATH mirror with a `level` field)",
    )
    p.add_argument("--config", default="default", help="HF dataset config/subset name")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="test")
    p.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=[3, 4, 5],
        help="Difficulty levels to keep (default: 3 4 5). Pass nothing to keep all.",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on number of training examples after filtering.",
    )
    p.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional cap on number of validation examples after filtering.",
    )
    args = p.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    keep_levels = set(args.levels or [])

    print(f"Loading {args.dataset_name} (config={args.config}) ...")
    dsd = datasets.load_dataset(args.dataset_name, args.config)

    train_ds = build_split(dsd[args.train_split], "train", args.dataset_name, keep_levels)
    val_ds = build_split(dsd[args.val_split], "validation", args.dataset_name, keep_levels)

    if args.max_train is not None and len(train_ds) > args.max_train:
        train_ds = train_ds.select(range(args.max_train))
        print(f"[train] truncated to {len(train_ds)}")
    if args.max_val is not None and len(val_ds) > args.max_val:
        val_ds = val_ds.select(range(args.max_val))
        print(f"[validation] truncated to {len(val_ds)}")

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "validation.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"\nWrote {len(train_ds)} train rows -> {train_path}")
    print(f"Wrote {len(val_ds)} val rows   -> {val_path}")
    if len(train_ds):
        print("\nExample train row:")
        ex = train_ds[0]
        print("  ground_truth:", ex["reward_model"]["ground_truth"])
        print("  level:", ex["extra_info"]["level"], "| type:", ex["extra_info"]["type"])
        print("  prompt[:160]:", ex["prompt"][0]["content"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
