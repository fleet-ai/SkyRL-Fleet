"""Generate multi-rollout math traces for credit-assignment / judge calibration.

Produces the same JSONL schema as the existing DAPO / AIME trace files
(``problem_index``, ``raw_problem``, ``ground_truth``, ``completion``, ``pred``,
``acc``, ``reward`` ∈ {±1}, ``roll_idx``, ``chat_history`` ...) so the output can
be fed directly to ``analyze_traces.py --format dapo-math`` and viewed in the
trace-viewer.

Samples are answered single-turn by an OpenRouter model (default
``qwen/qwen-2.5-72b-instruct``) at temperature 0.7, K rollouts per problem.

Supported datasets:
  * ``gsm8k``      — HuggingFace ``openai/gsm8k`` (config ``main``), answer is the
                     integer after ``####``.
  * ``parquet``    — any local parquet with a ``prompt`` (chat list or string) and
                     a ``reward_model.ground_truth`` field (e.g. dapo-math-17k).

Usage:
    export OPENROUTER_API_KEY=...
    python -m integrations.fleet.entrypoints.generate_math_rollouts \\
        --dataset gsm8k --split test \\
        --n-problems 1000 --rollouts-per-problem 4 \\
        --output ~/Work/data/gsm8k/gsm8k_traces_qwen2.5-72b.jsonl

Outputs:
    {output}                         — one JSON trace per line
    {output%.jsonl}_summary.json     — run-level metrics (pass@1, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the repo's canonical Hendrycks-MATH grader (boxed extraction + Minerva
# normalization — the same rule-lighteval/MATH_v2 style used for the DAPO traces).
_SKYRL_GYM = _REPO_ROOT / "skyrl-gym"
if str(_SKYRL_GYM) not in sys.path:
    sys.path.insert(0, str(_SKYRL_GYM))
try:
    from skyrl_gym.envs.aime.utils import (  # type: ignore
        is_correct_minerva,
        last_boxed_only_string,
        normalize_final_answer,
        remove_boxed,
    )

    _HAS_MINERVA = True
except Exception as _e:  # noqa: BLE001
    logger.warning(f"Minerva grader unavailable ({_e}); falling back to numeric grading")
    _HAS_MINERVA = False

# Prompt template — identical to the existing DAPO/AIME traces so rollouts are
# directly comparable.
_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


# ---------------------------------------------------------------------------
# Dataset loading -> list of {"problem", "ground_truth"}
# ---------------------------------------------------------------------------


def _gsm8k_gt(answer_raw: str) -> str:
    """Ground-truth integer for a GSM8k example (text after ``####``)."""
    m = re.search(r"####\s*(-?[0-9.,]+)", answer_raw)
    if not m:
        return ""
    return m.group(1).strip().replace(",", "")


def _math_gt(example: Dict[str, Any]) -> str:
    """Ground-truth answer for a Hendrycks-MATH example.

    Prefers the pre-extracted ``answer`` column; falls back to the last boxed
    expression in the solution.
    """
    ans = (example.get("answer") or "").strip()
    if not ans:
        sol = example.get("solution") or ""
        boxed = last_boxed_only_string(sol) if _HAS_MINERVA else None
        ans = remove_boxed(boxed) if boxed else ""
    return ans


def load_problems(
    dataset: str,
    split: str,
    parquet_path: Optional[str],
    levels: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if dataset == "gsm8k":
        import datasets

        ds = datasets.load_dataset("openai/gsm8k", "main")[split]
        problems = [
            {"problem": ex["question"], "ground_truth": _gsm8k_gt(ex["answer"])}
            for ex in ds
        ]
        logger.info(f"Loaded {len(problems)} GSM8k '{split}' problems")
        return problems

    if dataset == "math":
        import datasets

        ds = datasets.load_dataset("nlile/hendrycks-MATH-benchmark")[split]
        problems = []
        for ex in ds:
            lvl = ex.get("level")
            if levels and lvl not in levels:
                continue
            problems.append({
                "problem": ex["problem"],
                "ground_truth": _math_gt(ex),
                "level": lvl,
                "subject": ex.get("subject"),
            })
        lvl_msg = f" (levels {levels})" if levels else ""
        logger.info(f"Loaded {len(problems)} Hendrycks-MATH '{split}' problems{lvl_msg}")
        return problems

    if dataset == "parquet":
        if not parquet_path:
            raise ValueError("--parquet-path required for --dataset parquet")
        import pyarrow.parquet as pq

        t = pq.read_table(os.path.expanduser(parquet_path))
        rows = t.to_pylist()
        problems = []
        for r in rows:
            prompt = r.get("prompt")
            if isinstance(prompt, list):  # chat format; take the user content verbatim
                text = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
            else:
                text = str(prompt)
            gt = ""
            rm = r.get("reward_model") or {}
            if isinstance(rm, dict):
                gt = str(rm.get("ground_truth", ""))
            problems.append({"problem": text, "ground_truth": gt, "_prebuilt_prompt": isinstance(prompt, list)})
        logger.info(f"Loaded {len(problems)} problems from {parquet_path}")
        return problems

    raise ValueError(f"unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Answer extraction + grading
# ---------------------------------------------------------------------------

_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_pred(completion: str) -> str:
    """Pull the predicted answer from the last ``Answer:`` line."""
    matches = _ANSWER_LINE_RE.findall(completion or "")
    raw = matches[-1].strip() if matches else (completion or "").strip().splitlines()[-1:] or [""]
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    # Strip common wrappers: \boxed{}, $, **, trailing punctuation.
    raw = re.sub(r"\\boxed\{(.*?)\}", r"\1", raw)
    raw = raw.replace("$", "").replace("**", "").strip().rstrip(".")
    return raw


def _normalize_num(s: str) -> Optional[float]:
    m = _NUM_RE.search(s or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def grade(pred: str, ground_truth: str) -> bool:
    """True if pred matches ground_truth (numeric where possible, else string)."""
    gt = (ground_truth or "").strip()
    pr = (pred or "").strip()
    if not gt:
        return False
    pn, gn = _normalize_num(pr), _normalize_num(gt)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return pr.replace(",", "") == gt.replace(",", "")


def _sanitize_math_delims(text: str) -> str:
    """Convert LaTeX inline/display delimiters to ``$`` so the normalizer's
    ``$...$`` handling applies (models often emit ``Answer: \\(\\frac12\\)``)."""
    return (
        (text or "")
        .replace("\\(", "$").replace("\\)", "$")
        .replace("\\[", "$").replace("\\]", "$")
        .replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    )


def grade_completion(completion: str, ground_truth: str) -> Tuple[bool, str]:
    """Grade a completion against ground truth, returning (acc, normalized_pred).

    Uses the repo's Minerva grader when available (handles LaTeX fractions,
    \\sqrt, units, etc.); otherwise falls back to surface numeric/string match.
    """
    clean = _sanitize_math_delims(completion)
    if _HAS_MINERVA:
        gt_norm = normalize_final_answer(_sanitize_math_delims(ground_truth or ""))
        acc, pred = is_correct_minerva(clean, gt_norm)
        return bool(acc), pred
    pred = extract_pred(clean)
    return grade(pred, ground_truth), pred


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def make_client(base_url: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY (or OPENAI_API_KEY) not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def run_one(
    client,
    model: str,
    problem: Dict[str, str],
    problem_index: int,
    roll_idx: int,
    temperature: float,
    max_tokens: int,
    dataset: str,
    retries: int = 6,
) -> Optional[Dict[str, Any]]:
    raw_problem = problem["problem"]
    gt = problem["ground_truth"]
    prompt = raw_problem if problem.get("_prebuilt_prompt") else _PROMPT_TEMPLATE.format(problem=raw_problem)

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not getattr(resp, "choices", None):
                raise RuntimeError(f"no choices in response: {resp}")
            completion = resp.choices[0].message.content or ""
            usage = resp.usage
            acc, pred = grade_completion(completion, gt)
            rec = {
                "problem_index": problem_index,
                "raw_problem": raw_problem,
                "ground_truth": gt,
                "dataset": dataset,
                "ability": "MATH",
                "model": model,
                "roll_idx": roll_idx,
                "temperature": temperature,
                "reward": 1.0 if acc else -1.0,
                "acc": acc,
                "pred": pred,
                "completion": completion,
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "chat_history": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
            if problem.get("level") is not None:
                rec["level"] = problem["level"]
            if problem.get("subject") is not None:
                rec["subject"] = problem["subject"]
            return rec
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Exponential backoff with jitter to ride out upstream 429s.
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.5))
    logger.warning(f"problem {problem_index} roll {roll_idx} failed: {last_err}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Generate multi-rollout math traces")
    p.add_argument("--dataset", choices=["gsm8k", "math", "parquet"], default="gsm8k")
    p.add_argument("--split", default="test", help="HF split (gsm8k: test|train; math: train|test)")
    p.add_argument("--parquet-path", default=None, help="Path for --dataset parquet")
    p.add_argument(
        "--levels",
        default="3,4,5",
        help="Comma-separated Hendrycks-MATH difficulty levels to keep (1-5). math only.",
    )
    p.add_argument("--n-problems", type=int, default=1000)
    p.add_argument("--rollouts-per-problem", type=int, default=4)
    p.add_argument("--model", default="qwen/qwen-2.5-72b-instruct")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="~/Work/data/gsm8k/gsm8k_traces_qwen2.5-72b.jsonl")
    args = p.parse_args()

    levels = None
    if args.dataset == "math" and args.levels:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
    problems = load_problems(args.dataset, args.split, args.parquet_path, levels)
    rng = random.Random(args.seed)
    if args.n_problems and args.n_problems < len(problems):
        idxs = rng.sample(range(len(problems)), args.n_problems)
        idxs.sort()
        problems = [problems[i] for i in idxs]
    logger.info(f"Targeting {len(problems)} problems x {args.rollouts_per_problem} rollouts")

    client = make_client(args.base_url)

    jobs: List[Tuple[int, int]] = [
        (pi, ri) for pi in range(len(problems)) for ri in range(args.rollouts_per_problem)
    ]
    total = len(jobs)
    results: List[Dict[str, Any]] = []
    errors = 0
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def _work(job: Tuple[int, int]):
        pi, ri = job
        return run_one(
            client, args.model, problems[pi], pi, ri,
            args.temperature, args.max_tokens, args.dataset,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_work, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                done += 1
                if rec is None:
                    errors += 1
                else:
                    results.append(rec)
                if done % 50 == 0 or done == total:
                    rate = done / max(time.time() - t0, 1e-6)
                    logger.info(f"  {done}/{total} ({errors} errors) | {rate:.1f}/s")

    results.sort(key=lambda r: (r["problem_index"], r["roll_idx"]))

    out_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    by_problem: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_problem.setdefault(r["problem_index"], []).append(r)
    pass_at_1 = (sum(1 for r in results if r["acc"]) / len(results)) if results else 0.0
    solved_any = sum(1 for recs in by_problem.values() if any(r["acc"] for r in recs))
    mixed = sum(
        1 for recs in by_problem.values()
        if len({r["acc"] for r in recs}) > 1
    )
    # per-level pass@1 (math)
    by_level: Dict[Any, List[bool]] = {}
    for r in results:
        if "level" in r:
            by_level.setdefault(r["level"], []).append(r["acc"])
    per_level = {
        str(k): {"traces": len(v), "pass_at_1": round(sum(v) / len(v), 4)}
        for k, v in sorted(by_level.items(), key=lambda kv: str(kv[0]))
    }

    ds_label = {"gsm8k": f"gsm8k-{args.split}", "math": f"hendrycks-math-{args.split}"}.get(
        args.dataset, args.dataset
    )
    summary = {
        "model": args.model,
        "dataset": ds_label,
        "levels": levels,
        "sample_seed": args.seed,
        "problems_targeted": len(problems),
        "rollouts_per_problem": args.rollouts_per_problem,
        "temperature": args.temperature,
        "traces_saved": len(results),
        "distinct_questions_with_traces": len(by_problem),
        "errors_this_run": errors,
        "pass_at_1": round(pass_at_1, 4),
        "problems_solved_any": solved_any,
        "problems_mixed_reward": mixed,
        "per_level": per_level,
        "path": out_path,
    }
    summary_path = re.sub(r"\.jsonl$", "", out_path) + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(results)} traces -> {out_path}")
    logger.info(f"Summary -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
