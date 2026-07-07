#!/usr/bin/env python3
"""FanToM dataset loader for the offline Theory-of-Mind eval harness.

Loads and parses the FANToM benchmark (github.com/skywalker023/fantom) into a flat
list of per-question eval items conforming to the shared ToM harness interface.

Coverage
--------
- beliefq_choice       : MCQ belief questions (2-option; scoring: "mcq").
- answerability_binary : per-character binary knowability of a target fact (scoring: "binary").
- answerability_list   : list of all characters who can answer the target fact (scoring: "list").
- infoaccess_binary    : per-character binary access to stated information (scoring: "binary").
- infoaccess_list      : list of all characters who have access to the information (scoring: "list").
- fact                 : factual comprehension, F1-rule-scorable (scoring: "exact").

Excluded
--------
- beliefq_dist (free-response belief questions): correct requires cosine-similarity / LLM-judge
  scoring; only the MCQ variant is retained for fully rule-based evaluation.
- Binary questions whose gold is "no:long": dropped for the short-context setting because the
  long context is required to verify the extended negation (matches the official eval convention
  documented in eval_fantom.py: short_context input type drops these items).

Context variant
---------------
Each question set ships with both a *short* context (the final segment of the conversation,
immediately before the missing-information character re-joins) and the *full* context (the
entire conversation).  This loader defaults to **short** context, which is the standard
benchmark scoring setup and the easiest model input.  To switch to the full context, change
`_CONTEXT_KEY = "short_context"` → `"full_context"` below.
"""

from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"
_TARBALL = "fantom.tar.gz"
_JSON_NAME = "fantom_v1.json"

# Switch to "full_context" for the harder evaluation setting.
_CONTEXT_KEY = "short_context"

_PROMPT_HEADER = (
    "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, "
    "based on the following in-person conversation between individuals who have just met."
)

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(data_dir: Path) -> None:
    """Download and extract FanToM into *data_dir* if the JSON is not already present.

    The official tarball extracts a single flat file (fantom_v1.json) directly into the
    target directory; the archive is deleted after successful extraction.
    """
    json_path = data_dir / _JSON_NAME
    if json_path.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / _TARBALL
    print(f"[fantom_loader] Downloading FanToM from {_DATA_URL} …", flush=True)
    urllib.request.urlretrieve(_DATA_URL, tar_path)

    print("[fantom_loader] Extracting …", flush=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(data_dir)
    tar_path.unlink()
    print(f"[fantom_loader] Dataset ready at {json_path}", flush=True)


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------


def _mcq_prompt_and_answer(
    context: str,
    question: str,
    correct: str,
    wrong: str,
    correct_first: bool,
) -> tuple[str, str, list[str]]:
    """Return (prompt_str, gold_letter, choices_list) for a 2-option MCQ item."""
    if correct_first:
        choices = [correct, wrong]
        gold = "A"
    else:
        choices = [wrong, correct]
        gold = "B"
    choices_block = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    prompt = (
        f"{_PROMPT_HEADER}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"{choices_block}\n\n"
        "Answer with only the letter of the correct option."
    )
    return prompt, gold, choices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_items(data_dir, max_samples: int = 0, seed: int = 0) -> list[dict]:
    """Load FanToM items for offline ToM evaluation.

    Args:
        data_dir: str or Path.  Raw data is auto-downloaded into Path(data_dir) if missing.
        max_samples: if > 0, the returned list is shuffled and truncated to this length.
            Shuffling happens at the **question** level after all items are built.
        seed: RNG seed for the question-level shuffle.

    Returns:
        List of eval item dicts with keys:
            id, task, subtype, system, prompt, answer, choices, scoring, meta.
    """
    data_dir = Path(data_dir)
    _download(data_dir)

    with open(data_dir / _JSON_NAME, encoding="utf-8") as f:
        sets: list[dict] = json.load(f)

    items: list[dict] = []

    for set_entry in sets:
        sid = set_entry["set_id"]
        part_id = set_entry["part_id"]
        conv_id = set_entry["conv_id"]
        context = set_entry[_CONTEXT_KEY].strip()
        fact_q = set_entry["factQA"]["question"]
        fact_a = set_entry["factQA"]["correct_answer"]

        base_meta = {"set_id": sid, "part_id": part_id, "conv_id": conv_id}

        # ------------------------------------------------------------------ #
        # Fact question  (scoring: exact / F1-rule-scorable)
        # ------------------------------------------------------------------ #
        items.append(
            {
                "id": f"fantom/{sid}/fact",
                "task": "fantom",
                "subtype": "fact",
                "system": None,
                "prompt": (
                    f"{_PROMPT_HEADER}\n\n"
                    f"{context}\n\n"
                    f"Question: {fact_q}\n\n"
                    "Answer with a single short phrase."
                ),
                "answer": fact_a,
                "choices": None,
                "scoring": "exact",
                "meta": {**base_meta, "question_type": "fact"},
            }
        )

        # ------------------------------------------------------------------ #
        # Belief questions: MCQ variant only
        # (beliefq_dist with free-response answers is excluded — see module
        # docstring; the MCQ wrapper makes these fully rule-scorable.)
        # Option ordering is deterministic per question to ensure reproducibility
        # across runs: correct_first = (hash(set_id) + belief_idx) % 2 == 0.
        # ------------------------------------------------------------------ #
        for bq_i, bqa in enumerate(set_entry["beliefQAs"]):
            correct_first = (hash(sid) + bq_i) % 2 == 0
            prompt, gold, choices = _mcq_prompt_and_answer(
                context,
                bqa["question"],
                bqa["correct_answer"],
                bqa["wrong_answer"],
                correct_first,
            )
            items.append(
                {
                    "id": f"fantom/{sid}/beliefq_{bq_i}",
                    "task": "fantom",
                    "subtype": "beliefq_choice",
                    "system": None,
                    "prompt": prompt,
                    "answer": gold,
                    "choices": choices,
                    "scoring": "mcq",
                    "meta": {
                        **base_meta,
                        "question_type": bqa["question_type"],
                        "tom_type": bqa["tom_type"],
                        "missed_info_accessibility": bqa["missed_info_accessibility"],
                    },
                }
            )

        # ------------------------------------------------------------------ #
        # Answerability — list variant
        # ------------------------------------------------------------------ #
        al = set_entry.get("answerabilityQA_list")
        if al is not None:
            items.append(
                {
                    "id": f"fantom/{sid}/answerability_list",
                    "task": "fantom",
                    "subtype": "answerability_list",
                    "system": None,
                    "prompt": (
                        f"{_PROMPT_HEADER}\n\n"
                        f"{context}\n\n"
                        f"Target question: {fact_q}\n"
                        f"Question: {al['question']}\n\n"
                        "Answer with only the names, separated by commas."
                    ),
                    "answer": ", ".join(al["correct_answer"]),
                    "choices": None,
                    "scoring": "list",
                    "meta": {
                        **base_meta,
                        "question_type": al["question_type"],
                        "missed_info_accessibility": al["missed_info_accessibility"],
                        "wrong_names": al["wrong_answer"],
                    },
                }
            )

        # ------------------------------------------------------------------ #
        # Answerability — binary variant
        # "no:long" items are dropped for the short-context setting: the model
        # receives only the short excerpt, so questions whose correct negation
        # depends on the full conversation are unanswerable in this setup.
        # ------------------------------------------------------------------ #
        for ab_i, abqa in enumerate(set_entry["answerabilityQAs_binary"]):
            if abqa["correct_answer"] == "no:long":
                continue
            items.append(
                {
                    "id": f"fantom/{sid}/answerability_binary_{ab_i}",
                    "task": "fantom",
                    "subtype": "answerability_binary",
                    "system": None,
                    "prompt": (
                        f"{_PROMPT_HEADER}\n\n"
                        f"{context}\n\n"
                        f"Target question: {fact_q}\n"
                        f"Question: {abqa['question']}\n\n"
                        "Answer with only 'yes' or 'no'."
                    ),
                    "answer": abqa["correct_answer"],  # "yes" or "no"
                    "choices": None,
                    "scoring": "binary",
                    "meta": {
                        **base_meta,
                        "question_type": abqa["question_type"],
                        "missed_info_accessibility": abqa["missed_info_accessibility"],
                    },
                }
            )

        # ------------------------------------------------------------------ #
        # Info-accessibility — list variant
        # ------------------------------------------------------------------ #
        il = set_entry.get("infoAccessibilityQA_list")
        if il is not None:
            items.append(
                {
                    "id": f"fantom/{sid}/infoaccess_list",
                    "task": "fantom",
                    "subtype": "infoaccess_list",
                    "system": None,
                    "prompt": (
                        f"{_PROMPT_HEADER}\n\n"
                        f"{context}\n\n"
                        f"Information: {fact_q} {fact_a}\n"
                        f"Question: {il['question']}\n\n"
                        "Answer with only the names, separated by commas."
                    ),
                    "answer": ", ".join(il["correct_answer"]),
                    "choices": None,
                    "scoring": "list",
                    "meta": {
                        **base_meta,
                        "question_type": il["question_type"],
                        "missed_info_accessibility": il["missed_info_accessibility"],
                        "wrong_names": il["wrong_answer"],
                    },
                }
            )

        # ------------------------------------------------------------------ #
        # Info-accessibility — binary variant  (no:long also dropped)
        # ------------------------------------------------------------------ #
        for ib_i, ibqa in enumerate(set_entry["infoAccessibilityQAs_binary"]):
            if ibqa["correct_answer"] == "no:long":
                continue
            items.append(
                {
                    "id": f"fantom/{sid}/infoaccess_binary_{ib_i}",
                    "task": "fantom",
                    "subtype": "infoaccess_binary",
                    "system": None,
                    "prompt": (
                        f"{_PROMPT_HEADER}\n\n"
                        f"{context}\n\n"
                        f"Information: {fact_q} {fact_a}\n"
                        f"Question: {ibqa['question']}\n\n"
                        "Answer with only 'yes' or 'no'."
                    ),
                    "answer": ibqa["correct_answer"],  # "yes" or "no"
                    "choices": None,
                    "scoring": "binary",
                    "meta": {
                        **base_meta,
                        "question_type": ibqa["question_type"],
                        "missed_info_accessibility": ibqa["missed_info_accessibility"],
                    },
                }
            )

    # Shuffle at the question level (after all items are built), then truncate.
    rng = random.Random(seed)
    rng.shuffle(items)
    if max_samples > 0:
        items = items[:max_samples]

    return items


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    data_dir = Path(__file__).parent / "data"
    print(f"Loading items from {data_dir} …\n")
    items = load_items(data_dir)

    print(f"Total items: {len(items)}\n")

    by_subtype = Counter(it["subtype"] for it in items)
    print("Items per subtype:")
    for st, cnt in sorted(by_subtype.items()):
        print(f"  {st:30s}: {cnt}")

    # One sample per scoring type (prompt truncated to 300 chars).
    seen: dict[str, dict] = {}
    for it in items:
        if it["scoring"] not in seen:
            seen[it["scoring"]] = it

    print("\nSample item per scoring type:")
    for scoring in sorted(seen):
        s = seen[scoring]
        print(f"\n{'─'*60}")
        print(f"  scoring  : {scoring!r}")
        print(f"  id       : {s['id']}")
        print(f"  subtype  : {s['subtype']}")
        print(f"  answer   : {s['answer']!r}")
        print(f"  choices  : {s['choices']}")
        print(f"  prompt   : {s['prompt'][:300]!r}")
