"""Tier 1 static evaluation for B6 SFT — teacher-forced CE + format validity.

Compares SFT'd model against base Qwen3.5-9B on:
  1. Teacher-forced CE on the v3 training set (judge_score >= 2)
     → expected: SFT << base. Confirms training really updated weights.
  2. Teacher-forced CE on the v3 filtered set (judge_score < 2)
     → expected: SFT mildly < base, but NOT dramatically. If SFT
       drops as much as on training set, that's overfitting / leakage.
  3. Per-game and per-tag breakdown.
  4. (Optional) Format validity via sample generation:
     - XML structure (<meta>, <add>, <delete>, <plan>) presence
     - <plan> integers in [1,5]
     - <add> non-empty
     - Response length distribution

Usage (typically launched via tasks/witness-sft-static-eval.yaml):
    python sft_static_eval.py \\
        --sft_model_path /path/to/merged/ \\
        --base_model_name Qwen/Qwen3.5-9B \\
        --eval_pairs /path/to/sft_pairs_judged.jsonl \\
        --output_json eval_results.json \\
        [--do_generation] \\
        [--max_pairs N]

Output: eval_results.json with all per-set metrics + summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer


# ── Model loading helpers ──────────────────────────────────────────────


def load_model(path_or_name: str, device: str = "cuda"):
    """Load Qwen3.5 VLM in bf16. Tries AutoModelForImageTextToText first
    (Qwen3.5 is a VLM), falls back to AutoModelForCausalLM."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path_or_name, trust_remote_code=True)
    is_vlm = hasattr(cfg, "vision_config") and getattr(cfg, "vision_config") is not None
    if is_vlm:
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
    else:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    print(f"  Loading {path_or_name} (cls={model_cls.__name__}, dtype=bf16)...")
    t0 = time.time()
    model = model_cls.from_pretrained(
        path_or_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


# ── Teacher-forced CE ───────────────────────────────────────────────────


@torch.no_grad()
def teacher_force_ce(model, tokenizer, pair: dict, max_length: int = 4096) -> Optional[Dict]:
    """Compute teacher-forced cross-entropy on the assistant tokens only.

    Returns dict with num_actions / nll_sum / ce_per_token, or None if
    tokenization is degenerate.
    """
    msgs = pair["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        return None

    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True,
                         max_length=max_length, add_special_tokens=False)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length,
                           add_special_tokens=False)

    prompt_len = len(prompt_ids["input_ids"])
    full_len = full_ids["input_ids"].shape[1]
    num_actions = full_len - prompt_len
    if num_actions <= 0:
        return None

    full_ids = {k: v.to(model.device) for k, v in full_ids.items()}

    out = model(**full_ids)
    logits = out.logits  # [1, full_len, V]

    # logits[:, t-1] predicts token at position t. So for assistant tokens
    # at positions [prompt_len, full_len), predicting logits index
    # [prompt_len-1, full_len-1).
    shift_logits = logits[:, prompt_len - 1 : full_len - 1, :].contiguous().float()
    shift_labels = full_ids["input_ids"][:, prompt_len:full_len].contiguous()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).item()

    return {
        "num_actions": num_actions,
        "nll_sum": nll_sum,
        "ce_per_token": nll_sum / num_actions,
    }


# ── Format validity (generation-based) ─────────────────────────────────

XML_TAG_RE = re.compile(r"<(meta|add|delete|keep|plan)>(.*?)</\1>", re.DOTALL)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
ADD_RE = re.compile(r"<add>(.*?)</add>", re.DOTALL)
META_RE = re.compile(r"<meta>(.*?)</meta>", re.DOTALL)


def check_response_format(response_text: str) -> Dict:
    """Parse XML structure from a generated response."""
    has_any_tag = bool(XML_TAG_RE.search(response_text))
    metas = META_RE.findall(response_text)
    adds = ADD_RE.findall(response_text)
    plans = PLAN_RE.findall(response_text)

    plan_valid = False
    plan_ints = []
    if plans:
        ints_str = re.findall(r"\d+", plans[0])
        plan_ints = [int(s) for s in ints_str]
        plan_valid = bool(plan_ints) and all(1 <= n <= 5 for n in plan_ints)

    add_nonempty = bool(adds) and any(a.strip() for a in adds)
    meta_nonempty = bool(metas) and any(m.strip() for m in metas)

    return {
        "has_any_tag": has_any_tag,
        "has_meta": meta_nonempty,
        "has_add": add_nonempty,
        "has_plan": bool(plans),
        "plan_valid": plan_valid,
        "plan_ints": plan_ints,
        "n_adds": len(adds),
        "response_chars": len(response_text),
    }


@torch.no_grad()
def generate_response(model, tokenizer, pair: dict, max_new_tokens: int = 512) -> str:
    msgs = pair["messages"]
    prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy for reproducibility
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    response_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# ── Main eval loop ──────────────────────────────────────────────────────


def aggregate(records: List[Dict], filter_fn=None) -> Dict:
    """Aggregate per-pair CE records into summary stats."""
    if filter_fn:
        records = [r for r in records if filter_fn(r)]
    if not records:
        return {"n": 0}
    nll_sum = sum(r["nll_sum"] for r in records)
    actions = sum(r["num_actions"] for r in records)
    per_token_list = [r["ce_per_token"] for r in records]
    return {
        "n": len(records),
        "total_assistant_tokens": actions,
        "ce_per_token_micro": nll_sum / max(actions, 1),     # token-weighted avg
        "ce_per_token_macro": float(np.mean(per_token_list)), # pair-weighted avg
        "ce_per_token_p50": float(np.median(per_token_list)),
        "ce_per_token_p95": float(np.percentile(per_token_list, 95)) if len(per_token_list) > 1 else per_token_list[0],
    }


def aggregate_format(records: List[Dict]) -> Dict:
    if not records:
        return {"n": 0}
    n = len(records)
    return {
        "n": n,
        "has_any_tag_pct": 100 * sum(r["has_any_tag"] for r in records) / n,
        "has_meta_pct": 100 * sum(r["has_meta"] for r in records) / n,
        "has_add_pct": 100 * sum(r["has_add"] for r in records) / n,
        "has_plan_pct": 100 * sum(r["has_plan"] for r in records) / n,
        "plan_valid_pct": 100 * sum(r["plan_valid"] for r in records) / n,
        "median_response_chars": int(np.median([r["response_chars"] for r in records])),
        "p95_response_chars": int(np.percentile([r["response_chars"] for r in records], 95)) if n > 1 else records[0]["response_chars"],
    }


def evaluate_model(model, tokenizer, pairs: List[Dict], do_generation: bool, gen_n: int) -> Dict:
    """Run teacher-forced CE on all pairs + (optional) format check on a sample."""
    print(f"\n  Teacher-forced CE on {len(pairs)} pairs...")
    ce_records = []
    for i, p in enumerate(pairs):
        if i % 20 == 0:
            print(f"    [{i}/{len(pairs)}]")
        rec = teacher_force_ce(model, tokenizer, p)
        if rec is not None:
            rec["meta"] = p["metadata"]
            ce_records.append(rec)

    result = {
        "ce_records": ce_records,
        "ce_n_evaluated": len(ce_records),
    }

    if do_generation:
        sample_pairs = pairs[:gen_n] if gen_n > 0 else pairs
        print(f"\n  Format check (greedy generation) on {len(sample_pairs)} prompts...")
        format_records = []
        for i, p in enumerate(sample_pairs):
            if i % 5 == 0:
                print(f"    [{i}/{len(sample_pairs)}]")
            try:
                resp = generate_response(model, tokenizer, p, max_new_tokens=512)
                fmt = check_response_format(resp)
                fmt["game_id"] = p["metadata"].get("game_id", "?")
                format_records.append(fmt)
            except Exception as e:
                print(f"    WARN: gen failed on pair {i}: {e}")
        result["format_records"] = format_records

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_model_path", required=True, help="path or HF name of SFT'd model")
    ap.add_argument("--base_model_name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--eval_pairs", required=True, help="path to sft_pairs_judged.jsonl")
    ap.add_argument("--output_json", default="eval_results.json")
    ap.add_argument("--do_generation", action="store_true",
                    help="run greedy generation on a sample for format validity check (slower)")
    ap.add_argument("--gen_n", type=int, default=20, help="number of prompts to generate for format check")
    ap.add_argument("--max_pairs", type=int, default=0, help="cap eval set size for quick smoke (0=all)")
    ap.add_argument("--judge_score_min_for_train", type=int, default=2,
                    help="pairs with score >= this go into 'train' set (matches training filter)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ── Load + bucket pairs ──
    print(f"[load] reading {args.eval_pairs}")
    all_pairs = [json.loads(l) for l in open(args.eval_pairs)]
    print(f"[load] {len(all_pairs)} total pairs")

    train_pairs = [p for p in all_pairs
                   if (p["metadata"].get("judge_score") or -1) >= args.judge_score_min_for_train]
    filtered_pairs = [p for p in all_pairs
                      if (p["metadata"].get("judge_score") or -1) < args.judge_score_min_for_train
                      and (p["metadata"].get("judge_score") or -1) >= 0]
    print(f"[load] train (judge>={args.judge_score_min_for_train}): {len(train_pairs)}")
    print(f"[load] filtered (judge<{args.judge_score_min_for_train}): {len(filtered_pairs)}")

    if args.max_pairs > 0:
        train_pairs = train_pairs[: args.max_pairs]
        filtered_pairs = filtered_pairs[: args.max_pairs]
        print(f"[load] capped to {args.max_pairs} per set")

    # Shared tokenizer (use SFT model's — has chat template from training)
    print(f"\n[tok] loading tokenizer from {args.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Eval each model ──
    results: Dict[str, Dict] = {}
    for label, model_path in [("sft", args.sft_model_path), ("base", args.base_model_name)]:
        print(f"\n{'=' * 60}")
        print(f"== Evaluating: {label}  ({model_path})")
        print(f"{'=' * 60}")
        model = load_model(model_path, device=args.device)

        results[label] = {
            "train": evaluate_model(model, tokenizer, train_pairs,
                                     args.do_generation, args.gen_n),
            "filtered": evaluate_model(model, tokenizer, filtered_pairs,
                                        args.do_generation, args.gen_n),
        }

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── Aggregate + report ──
    summary = {"sft_model_path": args.sft_model_path,
               "base_model_name": args.base_model_name}
    for arm in ("sft", "base"):
        for split in ("train", "filtered"):
            recs = results[arm][split].get("ce_records", [])
            summary[f"{arm}.{split}.overall"] = aggregate(recs)
            summary[f"{arm}.{split}.solid_solve"] = aggregate(
                recs, lambda r: "solid_solve" in r["meta"].get("tags", [])
            )
            summary[f"{arm}.{split}.negative"] = aggregate(
                recs, lambda r: "negative" in r["meta"].get("tags", [])
            )
            for game in ("tw01", "tw07", "tw09"):
                summary[f"{arm}.{split}.{game}"] = aggregate(
                    recs, lambda r: r["meta"].get("game_id") == game
                )
            if "format_records" in results[arm][split]:
                summary[f"{arm}.{split}.format"] = aggregate_format(
                    results[arm][split]["format_records"]
                )

    # Print key comparisons
    print(f"\n{'=' * 60}")
    print(f"{'TIER 1 SUMMARY — teacher-forced CE per token':^60}")
    print(f"{'=' * 60}")
    print(f"{'set':<25}{'sft':<12}{'base':<12}{'Δ (sft-base)':<15}")
    print(f"{'-' * 64}")
    for split in ("train", "filtered"):
        for tag in ("overall", "solid_solve", "negative", "tw01", "tw07", "tw09"):
            sft = summary.get(f"sft.{split}.{tag}", {}).get("ce_per_token_micro")
            base = summary.get(f"base.{split}.{tag}", {}).get("ce_per_token_micro")
            n = summary.get(f"sft.{split}.{tag}", {}).get("n", 0)
            if n == 0 or sft is None or base is None:
                continue
            delta = sft - base
            print(f"{split + '/' + tag:<25}{sft:<12.4f}{base:<12.4f}{delta:<+15.4f}  (n={n})")

    if args.do_generation:
        print(f"\n{'=' * 60}")
        print(f"{'FORMAT VALIDITY (greedy generation, train pairs)':^60}")
        print(f"{'=' * 60}")
        print(f"{'metric':<25}{'sft':<12}{'base':<12}{'Δ':<10}")
        print(f"{'-' * 59}")
        for k in ("has_any_tag_pct", "has_meta_pct", "has_add_pct",
                  "has_plan_pct", "plan_valid_pct"):
            sft_v = summary.get("sft.train.format", {}).get(k)
            base_v = summary.get("base.train.format", {}).get(k)
            if sft_v is None or base_v is None:
                continue
            print(f"{k:<25}{sft_v:<12.1f}{base_v:<12.1f}{sft_v - base_v:<+10.1f}")

    # Persist
    out = {
        "summary": summary,
        # Don't save per-pair records to keep file small; flip to True for debugging
        # "details": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
