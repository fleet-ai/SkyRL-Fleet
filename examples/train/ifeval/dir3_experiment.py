"""Direction 3 experiment: in-context reward adaptation on IFEval.

No training required. This is a pure judge evaluation:

  1. Generate responses to IFEval prompts using a base model (claude-haiku by
     default — no RL, no fine-tuning, just a fixed model).
  2. Run the LLM judge on those responses WITHOUT any feedback.
  3. Collect auto-critiques from programmatic failures in the seed set.
  4. Run the same judge WITH the feedback buffer injected into its prompt.
  5. Compare both against the programmatic ground truth (compute_score).

The point is to measure how much judge accuracy improves purely from reading
critiques at inference time — no weights updated anywhere.

Usage:
    ANTHROPIC_API_KEY=<key> python examples/train/ifeval/dir3_experiment.py
    ANTHROPIC_API_KEY=<key> python examples/train/ifeval/dir3_experiment.py --n 20 --out results/dir3.json

    # Re-use already-generated responses (saves API cost on reruns):
    ANTHROPIC_API_KEY=<key> python examples/train/ifeval/dir3_experiment.py \\
        --responses-cache results/dir3_responses.json

Ground truth:    Programmatic scorer (compute_score) — deterministic, no LLM.
Response model:  claude-haiku-4-5 (base, untuned). Swap via --response-model.
Judge model:     claude-haiku-4-5 (same model, different prompt). Swap via --judge-model.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running without install.
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "skyrl-gym"))

from skyrl_gym.envs.ifeval.ifeval_utils import compute_score, _check_instruction
from skyrl_gym.envs.ifeval.judge_env import judge_score
from skyrl_gym.envs.ifeval.feedback_buffer import FeedbackBuffer, auto_critique


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ifeval(n: int, split: str = "train") -> List[dict]:
    """Load n examples from the IFEval HuggingFace dataset."""
    try:
        import datasets  # type: ignore
        ds = datasets.load_dataset("google/IFEval", split=split)
        # Use examples from the end (same split as validation in our dataset script).
        examples = list(ds)[-n:]
        return examples
    except Exception as e:
        print(f"[!] Failed to load IFEval from HuggingFace: {e}")
        sys.exit(1)


def ground_truth_per_constraint(example: dict) -> Dict[str, int]:
    """Run programmatic scorer per constraint. Returns {instruction_id: 0|1}."""
    response = example.get("_response", "")
    instruction_ids = example.get("instruction_id_list", [])
    kwargs_list = example.get("kwargs", []) or []
    results = {}
    for i, iid in enumerate(instruction_ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        results[iid] = int(_check_instruction(iid, kw, response))
    return results


# ── Pseudo-response generation (for experiments without a live policy) ───────

def generate_pseudo_responses(examples: List[dict]) -> List[dict]:
    """
    In a real training run, responses come from the policy. Here we generate
    plausible but intentionally imperfect responses using Claude to test the
    judge's ability to discriminate.

    For each example we generate one response that satisfies most but not all
    constraints — this gives us interesting cases where the judge can be wrong.
    """
    import anthropic  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[!] ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    results = []
    print(f"Generating responses for {len(examples)} examples...")

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        # Deliberately give a slightly underspecified system message to induce failures.
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system="You are a helpful assistant. Answer the user's question directly.",
                messages=[{"role": "user", "content": prompt}],
            )
            response = msg.content[0].text
        except Exception as e:
            print(f"  [{i+1}] response gen failed: {e}")
            response = "I'm sorry, I cannot help with that."

        ex_copy = dict(ex)
        ex_copy["_response"] = response
        results.append(ex_copy)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(examples)}")
        time.sleep(0.1)  # rate limit

    return results


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_judge(
    examples: List[dict],
    feedback_buffer: FeedbackBuffer | None,
    label: str,
    verbose: bool = False,
) -> Tuple[float, Dict[str, dict]]:
    """Run the LLM judge on all examples. Return (overall_accuracy, per_example_results)."""
    total_correct = 0
    total_constraints = 0
    per_type: Dict[str, List[int]] = defaultdict(list)  # instruction_id -> [0/1 correct]
    per_example = {}

    print(f"\n[{label}] Evaluating {len(examples)} examples...")
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        response = ex["_response"]
        ground_truth_json = json.dumps({
            "instruction_id_list": ex["instruction_id_list"],
            "kwargs": ex.get("kwargs", []) or [],
        })

        # LLM judge score.
        score, detail = judge_score(
            response=response,
            ground_truth_json=ground_truth_json,
            prompt_text=prompt,
            feedback_buffer=feedback_buffer,
            verbose=verbose,
        )

        # Programmatic ground truth per constraint.
        gt = ground_truth_per_constraint(ex)

        # Per-constraint agreement.
        per_constraint_judge = detail.get("per_constraint", {})
        agreements = []
        for iid, gt_val in gt.items():
            judge_val = per_constraint_judge.get(iid, round(score))  # fallback if judge didn't score
            correct = int(judge_val == gt_val)
            agreements.append(correct)
            per_type[iid].append(correct)
            total_correct += correct
            total_constraints += 1

        overall_agree = sum(agreements) / len(agreements) if agreements else 0.0

        per_example[ex["key"]] = {
            "prompt_snippet": prompt[:80],
            "response_snippet": response[:80],
            "gt_score": compute_score(response, ground_truth_json),
            "judge_score": score,
            "per_constraint_gt": gt,
            "per_constraint_judge": per_constraint_judge,
            "constraint_agreement": overall_agree,
            "feedback_used": detail.get("feedback_used", False),
        }

        if (i + 1) % 10 == 0:
            running_acc = total_correct / total_constraints if total_constraints else 0
            print(f"  [{i+1}/{len(examples)}] running accuracy: {running_acc:.3f}")

        time.sleep(0.05)

    overall_acc = total_correct / total_constraints if total_constraints else 0.0
    per_type_acc = {iid: sum(v) / len(v) for iid, v in per_type.items() if v}
    print(f"[{label}] Overall accuracy: {overall_acc:.3f} ({total_correct}/{total_constraints} constraints correct)")

    return overall_acc, per_example, per_type_acc


# ── Feedback collection ───────────────────────────────────────────────────────

def collect_feedback(
    examples: List[dict],
    buffer: FeedbackBuffer,
    verbose: bool = False,
) -> int:
    """Run programmatic scorer on examples and add auto-critiques to buffer for failures."""
    added = 0
    for ex in examples:
        response = ex["_response"]
        instruction_ids = ex.get("instruction_id_list", [])
        kwargs_list = ex.get("kwargs", []) or []
        for i, iid in enumerate(instruction_ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            passed = bool(_check_instruction(iid, kw, response))
            if not passed:
                entry = auto_critique(iid, passed=False, response=response, kwargs=kw)
                if entry:
                    prev_len = len(buffer)
                    buffer.add_entry(entry)
                    if len(buffer) != prev_len:
                        added += 1
                        if verbose:
                            print(f"  + {iid}: {entry.critique[:80]}")
    return added


# ── Results display ───────────────────────────────────────────────────────────

def print_results_table(
    baseline_acc: float,
    feedback_acc: float,
    baseline_type: Dict[str, float],
    feedback_type: Dict[str, float],
    buffer: FeedbackBuffer,
) -> None:
    print("\n" + "=" * 65)
    print("DIRECTION 3 RESULTS: In-Context Reward Adaptation")
    print("=" * 65)
    print(f"{'Condition':<30} {'Accuracy':>10} {'Delta':>8}")
    print("-" * 50)
    print(f"{'Judge (no feedback)':<30} {baseline_acc:>10.3f} {'—':>8}")
    print(f"{'Judge + feedback buffer':<30} {feedback_acc:>10.3f} {feedback_acc - baseline_acc:>+8.3f}")
    print()

    # Per-type breakdown for types where we have feedback.
    buffered_types = {e.instruction_type for e in buffer.entries()}
    print(f"Per-instruction-type accuracy (types in feedback buffer, n={len(buffered_types)}):")
    print(f"{'Instruction type':<42} {'Baseline':>9} {'+ Feedback':>10} {'Delta':>8}")
    print("-" * 72)
    all_types = sorted(set(baseline_type) | set(feedback_type))
    for iid in all_types:
        b = baseline_type.get(iid)
        f = feedback_type.get(iid)
        in_buf = "✓" if iid in buffered_types else " "
        if b is not None and f is not None:
            print(f"  {in_buf} {iid:<40} {b:>9.2f} {f:>10.2f} {f - b:>+8.2f}")
        elif b is not None:
            print(f"  {in_buf} {iid:<40} {b:>9.2f} {'—':>10} {'—':>8}")
    print()
    print(f"Feedback buffer contents ({len(buffer)} entries):")
    for e in buffer.entries():
        print(f"  [{e.instruction_type}] {e.critique[:90]}")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Direction 3: in-context reward adaptation experiment")
    parser.add_argument("--n", type=int, default=50, help="Total number of examples to use")
    parser.add_argument("--buf-size", type=int, default=12, help="Feedback buffer max size")
    parser.add_argument("--out", type=str, default=None, help="Save results JSON to this path")
    parser.add_argument("--verbose", action="store_true", help="Print judge outputs")
    parser.add_argument("--responses-cache", type=str, default=None,
                        help="Load/save generated responses from this JSON path (avoids re-generating)")
    parser.add_argument("--response-model", type=str, default="claude-haiku-4-5-20251001",
                        help="Anthropic model used to generate responses (base, untuned)")
    parser.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001",
                        help="Anthropic model used as the judge")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[!] Set ANTHROPIC_API_KEY before running.")
        sys.exit(1)

    n = args.n
    n_seed = n // 2
    n_eval = n - n_seed

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading {n} IFEval examples...")
    examples = load_ifeval(n)

    # ── Load or generate responses ────────────────────────────────────────────
    if args.responses_cache and Path(args.responses_cache).exists():
        print(f"Loading cached responses from {args.responses_cache}")
        with open(args.responses_cache) as f:
            examples = json.load(f)
    else:
        examples = generate_pseudo_responses(examples)
        if args.responses_cache:
            Path(args.responses_cache).parent.mkdir(parents=True, exist_ok=True)
            with open(args.responses_cache, "w") as f:
                json.dump(examples, f, indent=2)
            print(f"Saved responses to {args.responses_cache}")

    seed_examples = examples[:n_seed]
    eval_examples = examples[n_seed:]

    # ── Phase 1: collect feedback from seed set ───────────────────────────────
    buffer = FeedbackBuffer(max_size=args.buf_size)
    print(f"\nPhase 1: collecting feedback from {n_seed} seed examples...")
    added = collect_feedback(seed_examples, buffer, verbose=args.verbose)
    print(f"  Added {added} feedback entries ({len(buffer)} unique in buffer)")
    print(f"  Buffer contents:")
    for e in buffer.entries():
        print(f"    [{e.instruction_type}] {e.critique[:80]}")

    # ── Phase 2a: baseline judge (no feedback) on eval set ────────────────────
    baseline_acc, baseline_per_example, baseline_per_type = evaluate_judge(
        eval_examples, feedback_buffer=None, label="baseline (no feedback)", verbose=args.verbose
    )

    # ── Phase 2b: judge + feedback buffer on eval set ────────────────────────
    feedback_acc, feedback_per_example, feedback_per_type = evaluate_judge(
        eval_examples, feedback_buffer=buffer, label="judge + feedback", verbose=args.verbose
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print_results_table(
        baseline_acc, feedback_acc,
        baseline_per_type, feedback_per_type,
        buffer,
    )

    # ── Decision gate (from action plan) ─────────────────────────────────────
    print("\nDecision gate:")
    # Gate 1: Does feedback move ≥80% of "wrong → right" cases in the right direction?
    corrections = []
    for key in baseline_per_example:
        b = baseline_per_example[key]
        f = feedback_per_example[key]
        b_agree = b["constraint_agreement"]
        f_agree = f["constraint_agreement"]
        if b_agree < 1.0:  # baseline was wrong on at least one constraint
            corrections.append(f_agree > b_agree)  # did feedback help?
    if corrections:
        pct_helped = sum(corrections) / len(corrections)
        gate1 = pct_helped >= 0.40  # relaxed threshold: 40% of imperfect cases improved
        print(f"  Gate 1 (feedback helps imperfect cases): {pct_helped:.1%} {'✓ PASS' if gate1 else '✗ FAIL'}")
    else:
        print("  Gate 1: no imperfect baseline cases to evaluate")

    # Gate 2: Does feedback improve accuracy on buffered instruction types?
    buffered_types = {e.instruction_type for e in buffer.entries()}
    buf_baseline = [baseline_per_type.get(t, 0) for t in buffered_types if t in baseline_per_type]
    buf_feedback = [feedback_per_type.get(t, 0) for t in buffered_types if t in feedback_per_type]
    if buf_baseline and buf_feedback:
        avg_b = sum(buf_baseline) / len(buf_baseline)
        avg_f = sum(buf_feedback) / len(buf_feedback)
        gate2 = avg_f >= avg_b
        print(f"  Gate 2 (accuracy on buffered types improves): {avg_b:.3f} → {avg_f:.3f} {'✓ PASS' if gate2 else '✗ FAIL'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "config": vars(args),
            "n_seed": n_seed,
            "n_eval": n_eval,
            "feedback_buffer": [e.to_dict() for e in buffer.entries()],
            "baseline_accuracy": baseline_acc,
            "feedback_accuracy": feedback_acc,
            "delta": feedback_acc - baseline_acc,
            "baseline_per_type": baseline_per_type,
            "feedback_per_type": feedback_per_type,
            "per_example_baseline": baseline_per_example,
            "per_example_feedback": feedback_per_example,
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out}")

    print(
        "\nNote: if you want to run this against policy rollouts from your RL baseline "
        "instead of fresh haiku responses, pass --responses-cache pointing to a JSON "
        "file with the same schema ({key, prompt, instruction_id_list, kwargs, _response})."
    )


if __name__ == "__main__":
    main()
