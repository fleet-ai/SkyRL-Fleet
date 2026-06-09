"""Direction 3 mock experiment — runs without an API key.

Simulates a realistic LLM judge that:
  - Is PERFECT on easy constraint types (json_format, english_capital, no_comma)
  - Makes calibrated errors on ambiguous types (number_words: ±20% tolerance,
    number_sentences: ±1 tolerance, keywords:existence: misses 30% of cases)
  - Shows that in-context feedback reduces errors on the affected types

This is NOT a substitute for a real run but lets you validate the full pipeline
and see representative output before spending API credits.

Run:
    python examples/train/ifeval/dir3_mock_run.py
    python examples/train/ifeval/dir3_mock_run.py --out results/dir3_mock.json
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "skyrl-gym"))

from skyrl_gym.envs.ifeval.ifeval_utils import compute_score, _check_instruction
from skyrl_gym.envs.ifeval.feedback_buffer import FeedbackBuffer, auto_critique


# ── Simulated judge with realistic error patterns ─────────────────────────────

# Error rates per instruction type for the baseline judge (no feedback).
# 0.0 = perfect; 0.3 = wrong 30% of the time.
_BASELINE_ERROR_RATES = {
    "keywords:existence":                    0.28,  # misses case/synonym edge cases
    "keywords:frequency":                    0.22,  # miscounts occurrences
    "keywords:forbidden_words":              0.10,
    "keywords:letter_frequency":             0.30,  # letter counting is hard
    "length_constraints:number_words":       0.25,  # off-by-a-few errors
    "length_constraints:number_sentences":   0.20,  # what counts as a sentence?
    "length_constraints:number_paragraphs":  0.12,
    "length_constraints:nth_paragraph_first_word": 0.18,
    "detectable_content:number_placeholders": 0.08,
    "detectable_content:postscript":         0.12,
    "detectable_format:number_bullet_lists": 0.15,
    "detectable_format:constrained_response": 0.05,
    "detectable_format:number_highlighted_sections": 0.18,
    "detectable_format:multiple_sections":   0.12,
    "detectable_format:json_format":         0.03,
    "detectable_format:title":               0.08,
    "combination:two_responses":             0.10,
    "combination:repeat_prompt":             0.15,
    "startend:end_checker":                  0.20,
    "change_case:capital_word_frequency":    0.25,
    "change_case:english_capital":           0.04,
    "change_case:english_lowercase":         0.04,
    "punctuation:no_comma":                  0.06,
}

# After feedback, error rates drop substantially for buffered types.
# Types not in the buffer see no change.
_FEEDBACK_ERROR_REDUCTION = 0.60  # 60% reduction in error rate for buffered types


def mock_judge_score(
    instruction_id: str,
    ground_truth: int,
    with_feedback: bool,
    feedback_buffer: FeedbackBuffer | None,
    rng: random.Random,
) -> int:
    """Simulate a judge score. Returns 0 or 1."""
    base_err = _BASELINE_ERROR_RATES.get(instruction_id, 0.10)

    # Reduce error rate if this type is in the feedback buffer.
    if with_feedback and feedback_buffer:
        buffered = {e.instruction_type for e in feedback_buffer.entries()}
        if instruction_id in buffered:
            base_err = base_err * (1 - _FEEDBACK_ERROR_REDUCTION)

    # Flip the ground truth with probability = error_rate.
    if rng.random() < base_err:
        return 1 - ground_truth  # wrong
    return ground_truth           # correct


# ── Data ──────────────────────────────────────────────────────────────────────

def load_ifeval(n: int) -> List[dict]:
    """Load IFEval examples. Falls back to synthetic generation if HF is unavailable."""
    try:
        import datasets  # type: ignore
        ds = datasets.load_dataset("google/IFEval", split="train")
        return list(ds)[-n:]
    except Exception:
        print("  HuggingFace unavailable — generating synthetic IFEval examples.")
        return _generate_synthetic_ifeval(n)


def _generate_synthetic_ifeval(n: int) -> List[dict]:
    """Generate synthetic IFEval-format examples from the constraint taxonomy."""
    rng = random.Random(0)

    templates = [
        ("Write a short explanation of {topic}.", [
            ("keywords:existence", {"keywords": ["{kw1}", "{kw2}"]}),
            ("length_constraints:number_words", {"num_words": rng.randint(50, 150), "relation": "at least"}),
        ]),
        ("Describe {topic} in exactly {n} sentences.", [
            ("length_constraints:number_sentences", {"num_sentences": rng.randint(3, 8), "relation": "exactly"}),
            ("keywords:forbidden_words", {"forbidden_words": ["however", "moreover"]}),
        ]),
        ("Write about {topic} without using any commas.", [
            ("punctuation:no_comma", {}),
            ("length_constraints:number_words", {"num_words": 40, "relation": "at least"}),
        ]),
        ("Explain {topic} in all lowercase letters.", [
            ("change_case:english_lowercase", {}),
            ("keywords:existence", {"keywords": ["{kw1}"]}),
        ]),
        ("Write about {topic}. End with the phrase 'I hope this helps.'", [
            ("startend:end_checker", {"end_phrase": "I hope this helps."}),
            ("keywords:existence", {"keywords": ["{kw1}", "{kw2}"]}),
        ]),
        ("Provide {n} bullet points about {topic}.", [
            ("detectable_format:number_bullet_lists", {"num_bullets": rng.randint(3, 6)}),
            ("length_constraints:number_words", {"num_words": 80, "relation": "at least"}),
        ]),
        ("Write a JSON object describing {topic}.", [
            ("detectable_format:json_format", {}),
        ]),
        ("Write about {topic}. Add a postscript at the end.", [
            ("detectable_content:postscript", {"postscript_marker": "P.S."}),
            ("keywords:forbidden_words", {"forbidden_words": ["sorry", "apologize"]}),
        ]),
        ("Write about {topic} in {n} or more words, including the word '{kw1}'.", [
            ("length_constraints:number_words", {"num_words": rng.randint(60, 120), "relation": "at least"}),
            ("keywords:existence", {"keywords": ["{kw1}"]}),
        ]),
        ("Describe {topic}. Use the word '{kw1}' at least {freq} times.", [
            ("keywords:frequency", {"keyword": "{kw1}", "frequency": rng.randint(2, 4), "relation": "at least"}),
        ]),
    ]

    topics = ["machine learning", "climate change", "renewable energy", "neural networks",
              "data privacy", "quantum computing", "space exploration", "genetic engineering",
              "economic inequality", "artificial intelligence", "blockchain", "cybersecurity",
              "urban planning", "biodiversity", "supply chains", "remote work", "education reform"]
    keywords = ["important", "significant", "critical", "essential", "fundamental",
                "process", "system", "impact", "challenge", "solution", "approach",
                "development", "research", "data", "technology", "analysis"]

    examples = []
    for i in range(n):
        template_prompt, template_constraints = rng.choice(templates)
        topic = rng.choice(topics)
        kw1 = rng.choice(keywords)
        kw2 = rng.choice([k for k in keywords if k != kw1])
        freq = rng.randint(2, 3)
        n_val = rng.randint(3, 8)

        def sub(s):
            return (str(s).replace("{topic}", topic).replace("{kw1}", kw1)
                    .replace("{kw2}", kw2).replace("{freq}", str(freq)).replace("{n}", str(n_val)))

        prompt = sub(template_prompt)
        instruction_ids = []
        kwargs_list = []
        for iid, kw_tmpl in template_constraints:
            instruction_ids.append(iid)
            resolved_kw = {k: sub(v) if isinstance(v, str) else
                          ([sub(x) for x in v] if isinstance(v, list) else v)
                          for k, v in kw_tmpl.items()}
            kwargs_list.append(resolved_kw)

        examples.append({
            "key": f"synthetic_{i:04d}",
            "prompt": prompt,
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs_list,
        })

    return examples


def make_programmatic_response(example: dict) -> str:
    """Build a response that satisfies ~70% of constraints on average."""
    rng_local = random.Random(hash(example["prompt"]))
    prompt = example["prompt"]
    instruction_ids = example.get("instruction_id_list", [])
    kwargs_list = example.get("kwargs", []) or []

    # Start with the prompt echoed + a plausible paragraph.
    base = f"Here is my response to your request: {prompt[:100]}.\n\n"
    base += "I will do my best to fulfill the requirements you've outlined. "
    base += "Please let me know if this meets your expectations."

    # Apply easy constraints with high probability.
    for i, iid in enumerate(instruction_ids):
        kw = kwargs_list[i] if i < len(kwargs_list) else {}
        if rng_local.random() < 0.65:
            base = _try_satisfy(base, iid, kw, rng_local)

    return base


def _try_satisfy(response: str, iid: str, kwargs: dict, rng: random.Random) -> str:
    if iid == "punctuation:no_comma":
        return response.replace(",", " —")
    if iid == "keywords:existence":
        kws = kwargs.get("keywords", []) or []
        return response + " " + " ".join(str(k) for k in kws)
    if iid == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", []) or []
        for w in forbidden:
            response = re.sub(r"\b" + re.escape(str(w)) + r"\b", "[removed]", response, flags=re.IGNORECASE)
        return response
    if iid == "change_case:english_lowercase":
        return response.lower()
    if iid == "change_case:english_capital":
        return response.upper()
    if iid == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        return response + f"\n\n{marker} Thank you."
    if iid == "startend:end_checker":
        phrase = kwargs.get("end_phrase", "")
        if phrase:
            return response + " " + phrase
    if iid == "detectable_format:number_bullet_lists":
        n = kwargs.get("num_bullets", 0)
        bullets = "\n".join(f"- Item {j+1}" for j in range(n))
        return response + "\n\n" + bullets
    if iid == "length_constraints:number_words":
        n = int(kwargs.get("num_words", 100))
        relation = kwargs.get("relation", "at least")
        words = re.findall(r"\b\w+\b", response)
        if relation in ("at least", "exactly") and len(words) < n:
            return response + " " + " ".join(["word"] * (n - len(words)))
    return response


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    examples: List[dict],
    feedback_buffer: FeedbackBuffer | None,
    label: str,
    rng: random.Random,
) -> Tuple[float, Dict[str, float]]:
    total_correct = 0
    total_constraints = 0
    per_type: Dict[str, List[int]] = defaultdict(list)
    with_feedback = feedback_buffer is not None and len(feedback_buffer) > 0

    for ex in examples:
        instruction_ids = ex.get("instruction_id_list", [])
        kwargs_list = ex.get("kwargs", []) or []
        response = ex["_response"]

        for i, iid in enumerate(instruction_ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            gt = int(_check_instruction(iid, kw, response))
            judge = mock_judge_score(iid, gt, with_feedback, feedback_buffer, rng)
            correct = int(judge == gt)
            per_type[iid].append(correct)
            total_correct += correct
            total_constraints += 1

    overall = total_correct / total_constraints if total_constraints else 0.0
    per_type_acc = {iid: sum(v) / len(v) for iid, v in per_type.items() if v}
    print(f"[{label}] accuracy: {overall:.3f}  ({total_correct}/{total_constraints} constraints)")
    return overall, per_type_acc


# ── Results display ───────────────────────────────────────────────────────────

def print_results(
    baseline_acc: float,
    feedback_acc: float,
    baseline_type: Dict[str, float],
    feedback_type: Dict[str, float],
    buffer: FeedbackBuffer,
) -> None:
    buffered_types = {e.instruction_type for e in buffer.entries()}
    W = 65

    print("\n" + "=" * W)
    print("DIRECTION 3 RESULTS: In-Context Reward Adaptation (mock)")
    print("=" * W)
    print(f"{'Condition':<35} {'Accuracy':>10} {'Delta':>8}")
    print("-" * 55)
    print(f"{'Judge (no feedback)':<35} {baseline_acc:>10.3f} {'—':>8}")
    print(f"{'Judge + feedback buffer':<35} {feedback_acc:>10.3f} {feedback_acc - baseline_acc:>+8.3f}")
    print()

    print(f"Per-type breakdown (✓ = type in feedback buffer):")
    print(f"{'':>2} {'Instruction type':<42} {'Base':>7} {'+ FB':>7} {'Δ':>6}")
    print("-" * 68)
    all_types = sorted(set(baseline_type) | set(feedback_type))
    for iid in all_types:
        b = baseline_type.get(iid)
        f = feedback_type.get(iid)
        mark = "✓" if iid in buffered_types else " "
        if b is not None and f is not None:
            delta = f - b
            highlight = " ◄" if abs(delta) >= 0.05 and iid in buffered_types else ""
            print(f"  {mark} {iid:<42} {b:>7.3f} {f:>7.3f} {delta:>+6.3f}{highlight}")
    print()

    print(f"Feedback buffer ({len(buffer)} entries):")
    for e in buffer.entries():
        print(f"  [{e.instruction_type}]")
        print(f"    {e.critique[:100]}")
    print()

    # Summary stats for buffered vs. unbuffered types.
    buf_b = [baseline_type[t] for t in buffered_types if t in baseline_type]
    buf_f = [feedback_type[t] for t in buffered_types if t in feedback_type]
    unbuf_b = [baseline_type[t] for t in baseline_type if t not in buffered_types]
    unbuf_f = [feedback_type[t] for t in feedback_type if t not in buffered_types]

    if buf_b and buf_f:
        print(f"Buffered types avg:   {sum(buf_b)/len(buf_b):.3f} → {sum(buf_f)/len(buf_f):.3f}  "
              f"(Δ {sum(buf_f)/len(buf_f) - sum(buf_b)/len(buf_b):+.3f})")
    if unbuf_b and unbuf_f:
        print(f"Unbuffered types avg: {sum(unbuf_b)/len(unbuf_b):.3f} → {sum(unbuf_f)/len(unbuf_f):.3f}  "
              f"(Δ {sum(unbuf_f)/len(unbuf_f) - sum(unbuf_b)/len(unbuf_b):+.3f}  — should be ~0)")

    print()
    print("Decision gate:")
    delta = feedback_acc - baseline_acc
    buf_delta = (sum(buf_f)/len(buf_f) - sum(buf_b)/len(buf_b)) if buf_b and buf_f else 0
    unbuf_delta = (sum(unbuf_f)/len(unbuf_f) - sum(unbuf_b)/len(unbuf_b)) if unbuf_b and unbuf_f else 0
    g1 = buf_delta >= 0.02
    g2 = abs(unbuf_delta) <= 0.02  # feedback shouldn't hurt unbuffered types
    print(f"  G1 accuracy improves on buffered types:   Δ={buf_delta:+.3f}  {'✓ PASS' if g1 else '✗ FAIL'}")
    print(f"  G2 accuracy stable on unbuffered types:   Δ={unbuf_delta:+.3f}  {'✓ PASS' if g2 else '✗ FAIL'}")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Total examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buf-size", type=int, default=12)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.n
    n_seed = n // 2
    n_eval = n - n_seed

    print(f"Loading {n} IFEval examples...")
    examples = load_ifeval(n)

    # Generate pseudo-responses.
    print("Generating pseudo-responses (programmatic, no API)...")
    for ex in examples:
        ex["_response"] = make_programmatic_response(ex)
    print(f"  Done. Average programmatic score: "
          f"{sum(compute_score(ex['_response'], json.dumps({'instruction_id_list': ex['instruction_id_list'], 'kwargs': ex.get('kwargs',[])})) for ex in examples) / len(examples):.3f}")

    seed_examples = examples[:n_seed]
    eval_examples = examples[n_seed:]

    # Phase 1: collect feedback from seed failures.
    buffer = FeedbackBuffer(max_size=args.buf_size)
    print(f"\nPhase 1: collecting feedback from {n_seed} seed examples...")
    for ex in seed_examples:
        instruction_ids = ex.get("instruction_id_list", [])
        kwargs_list = ex.get("kwargs", []) or []
        response = ex["_response"]
        for i, iid in enumerate(instruction_ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            passed = bool(_check_instruction(iid, kw, response))
            if not passed:
                entry = auto_critique(iid, passed=False, response=response, kwargs=kw)
                if entry:
                    buffer.add_entry(entry)
    print(f"  Buffer: {len(buffer)} unique instruction types with critiques")

    # Phase 2a: baseline (no feedback).
    print(f"\nPhase 2a: baseline judge on {n_eval} examples...")
    baseline_acc, baseline_type = evaluate(eval_examples, None, "no feedback", rng)

    # Phase 2b: judge + feedback.
    print(f"Phase 2b: judge + feedback buffer...")
    rng2 = random.Random(args.seed)  # same seed so only the buffer changes behavior
    feedback_acc, feedback_type = evaluate(eval_examples, buffer, "+ feedback", rng2)

    # Display.
    print_results(baseline_acc, feedback_acc, baseline_type, feedback_type, buffer)

    # Save.
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results = {
            "config": vars(args),
            "n_seed": n_seed, "n_eval": n_eval,
            "baseline_accuracy": baseline_acc,
            "feedback_accuracy": feedback_acc,
            "delta": feedback_acc - baseline_acc,
            "baseline_per_type": baseline_type,
            "feedback_per_type": feedback_type,
            "buffer": [e.to_dict() for e in buffer.entries()],
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
