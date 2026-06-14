#!/usr/bin/env python3
"""Offline Theory-of-Mind evaluation harness for negotiation checkpoints.

This runner evaluates model responses on FanToM (multiparty, information-asymmetry
ToM) items through an OpenAI-compatible chat endpoint (vLLM local serve, OpenRouter, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


NO_THINK_TOKEN = "/no_think"
NO_THINK_BODY = {"reasoning": {"enabled": False}}
TASK_TO_LOADER = {
    "fantom": "fantom_loader",
}


def make_client(base_url: str, timeout: float = 90.0, max_retries: int = 2) -> "AsyncOpenAI":
    if AsyncOpenAI is None:
        raise RuntimeError("`openai` package required for endpoint calls (pip install openai).")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY).")
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


async def chat(client, model, messages, temperature, max_tokens=256, retries=4, extra_body=None):
    """Robust chat call that adapts to provider parameter quirks."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if temperature is not None and temperature >= 0:
        kwargs["temperature"] = temperature

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            if not getattr(resp, "choices", None):
                return ""
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
            elif "max_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif ("reasoning" in msg or "extra_body" in msg) and "extra_body" in kwargs:
                kwargs.pop("extra_body")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return ""


def _maybe_no_think(text: str, no_think: bool) -> str:
    if not no_think:
        return text
    return text + "\n\n" + NO_THINK_TOKEN


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_name(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\b(?:the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_exact(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return text


def _set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def parse_mcq_prediction(response: str, choices: list[str] | None) -> str | None:
    if not response:
        return None
    n_choices = len(choices or [])
    if n_choices <= 0:
        return None
    valid_letters = [chr(ord("A") + i) for i in range(n_choices)]
    valid_set = set(valid_letters)
    text = response.strip()

    preferred_patterns = [
        r"^\s*\(?([A-Z])\)?\b",
        r"\banswer(?:\s+is)?\s*[:\-]?\s*\(?([A-Z])\)?\b",
        r"\boption\s*\(?([A-Z])\)?\b",
        r"\b\(?([A-Z])\)\b",
    ]
    for pattern in preferred_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter in valid_set:
                return letter

    fallback = [
        m.group(1).upper()
        for m in re.finditer(r"\(?([A-Z])\)?", text, flags=re.IGNORECASE)
        if m.group(1).upper() in valid_set
    ]
    if fallback:
        return fallback[-1]

    norm_resp = _normalize_text(text)
    if norm_resp:
        for idx, choice in enumerate(choices or []):
            norm_choice = _normalize_text(choice)
            if not norm_choice:
                continue
            if norm_choice in norm_resp or norm_resp in norm_choice:
                return chr(ord("A") + idx)
    return None


def score_mcq(response: str, gold: str, choices: list[str] | None) -> tuple[str | None, bool]:
    pred = parse_mcq_prediction(response, choices)
    return pred, (pred is not None and pred.upper() == (gold or "").strip().upper())


def parse_binary_prediction(response: str) -> str | None:
    norm = _normalize_text(response)
    for match in re.finditer(r"\b(yes|no)\b", norm):
        return match.group(1)
    return None


def score_binary(response: str, gold: str) -> tuple[str | None, bool]:
    pred = parse_binary_prediction(response)
    gold_norm = _normalize_text(gold)
    return pred, (pred is not None and pred == gold_norm)


def parse_list_prediction(response: str) -> set[str]:
    pieces = re.split(r",|\n|\band\b", response or "", flags=re.IGNORECASE)
    names = {_normalize_name(piece) for piece in pieces}
    return {name for name in names if name}


def score_list(response: str, gold: str) -> tuple[list[str], bool, float]:
    pred_set = parse_list_prediction(response)
    gold_set = parse_list_prediction(gold)
    correct = pred_set == gold_set
    f1 = _set_f1(pred_set, gold_set)
    return sorted(pred_set), correct, f1


def score_exact(response: str, gold: str) -> tuple[str, bool]:
    pred_norm = _normalize_exact(response)
    gold_norm = _normalize_exact(gold)
    correct = bool(gold_norm) and (pred_norm == gold_norm or gold_norm in pred_norm)
    return pred_norm, correct


def score_item(item: dict, response: str) -> dict:
    scoring = item.get("scoring")
    gold = item.get("answer", "")
    if scoring == "mcq":
        pred, correct = score_mcq(response, gold, item.get("choices"))
        return {"prediction": pred, "correct": correct}
    if scoring == "binary":
        pred, correct = score_binary(response, gold)
        return {"prediction": pred, "correct": correct}
    if scoring == "list":
        pred, correct, list_f1 = score_list(response, gold)
        return {"prediction": pred, "correct": correct, "list_f1": round(list_f1, 4)}
    if scoring == "exact":
        pred, correct = score_exact(response, gold)
        return {"prediction": pred, "correct": correct}
    raise ValueError(f"Unknown scoring type: {scoring}")


def aggregate_task_results(results: list[dict], elapsed_s: float) -> dict:
    total = len(results)
    errors = [r for r in results if "error" in r]
    scored = [r for r in results if "error" not in r]
    correct_count = sum(1 for r in scored if r.get("correct"))
    acc = (correct_count / len(scored)) if scored else 0.0

    by_subtype: dict[str, dict[str, int]] = {}
    list_f1_vals: list[float] = []
    for row in scored:
        subtype = row.get("subtype", "unknown")
        bucket = by_subtype.setdefault(subtype, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(row.get("correct")))
        if "list_f1" in row:
            list_f1_vals.append(float(row["list_f1"]))

    acc_by_subtype = {
        subtype: round((bucket["correct"] / bucket["n"]) if bucket["n"] else 0.0, 4)
        for subtype, bucket in sorted(by_subtype.items())
    }

    return {
        "n": total,
        "accuracy": round(acc, 4),
        "accuracy_by_subtype": acc_by_subtype,
        "mean_list_f1": round(sum(list_f1_vals) / len(list_f1_vals), 4) if list_f1_vals else None,
        "n_errors": len(errors),
        "elapsed_s": round(elapsed_s, 2),
    }


def print_task_report(cfg: dict, agg: dict) -> None:
    print("\n" + "=" * 60)
    print(f"TOM EVAL - {cfg['task'].upper()}")
    print("=" * 60)
    print(f"  model         : {cfg['model']}")
    print(f"  base_url      : {cfg['base_url']}")
    print(f"  n             : {agg['n']}   seed: {cfg['seed']}   temp: {cfg['temperature']}")
    print(f"  max_tokens    : {cfg['max_tokens']}   no_think: {cfg['no_think']}")
    print("-" * 60)
    print(f"  accuracy                : {agg['accuracy']:.1%}")
    if agg["mean_list_f1"] is not None:
        print(f"  mean list-F1            : {agg['mean_list_f1']:.4f}")
    print(f"  item errors             : {agg['n_errors']}")
    if agg["accuracy_by_subtype"]:
        print("  accuracy by subtype:")
        for subtype, value in agg["accuracy_by_subtype"].items():
            print(f"    - {subtype}: {value:.1%}")
    print("=" * 60 + "\n")


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def redact_base_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return url
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        return url


def parse_tasks(raw: str) -> list[str]:
    tasks = [t.strip().lower() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tasks if t not in TASK_TO_LOADER]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}. Valid tasks: {sorted(TASK_TO_LOADER)}")
    if not tasks:
        raise ValueError("No tasks requested.")
    return tasks


def load_task_items(task: str, data_dir: str, n: int, seed: int) -> list[dict]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    module_name = TASK_TO_LOADER[task]
    module = importlib.import_module(module_name)
    items = module.load_items(data_dir=data_dir, max_samples=n, seed=seed)
    if not isinstance(items, list):
        raise TypeError(f"{module_name}.load_items(...) must return list[dict]")
    return items


async def eval_task(
    client,
    model: str,
    task: str,
    items: list[dict],
    temperature: float,
    max_tokens: int,
    concurrency: int,
    no_think: bool,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = {"k": 0}
    total = len(items)
    body = NO_THINK_BODY if no_think else None

    async def run_one(item: dict) -> dict:
        async with sem:
            try:
                system_prompt = item.get("system")
                user_prompt = item.get("prompt", "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": _maybe_no_think(system_prompt, no_think)})
                    messages.append({"role": "user", "content": user_prompt})
                else:
                    messages.append({"role": "user", "content": _maybe_no_think(user_prompt, no_think)})

                raw = await chat(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=body,
                )
                scored = score_item(item, raw)
                row = {
                    "id": item.get("id"),
                    "subtype": item.get("subtype"),
                    "gold": item.get("answer"),
                    "response": raw,
                    "prediction": scored["prediction"],
                    "correct": bool(scored["correct"]),
                }
                if "list_f1" in scored:
                    row["list_f1"] = scored["list_f1"]
                return row
            except Exception as e:  # noqa: BLE001
                return {
                    "id": item.get("id"),
                    "subtype": item.get("subtype"),
                    "gold": item.get("answer"),
                    "error": str(e),
                }
            finally:
                done["k"] += 1
                if done["k"] % 20 == 0 or done["k"] == total:
                    print(f"  [{task}] {done['k']}/{total} items", flush=True)

    return await asyncio.gather(*[run_one(item) for item in items])


def _print_dry_run_summary(task_to_items: dict[str, list[dict]]) -> None:
    print("\nDRY RUN (no API calls)")
    print("=" * 60)
    for task, items in task_to_items.items():
        subtype_counts: dict[str, int] = {}
        for item in items:
            subtype = str(item.get("subtype", "unknown"))
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        print(f"\nTask: {task}  n={len(items)}")
        for subtype, count in sorted(subtype_counts.items()):
            print(f"  - {subtype}: {count}")
        for i, item in enumerate(items[:2], start=1):
            prompt = (item.get("prompt") or "").strip().replace("\n", " ")
            preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
            print(f"  sample_prompt_{i}: {preview}")

    fake_items = [
        {
            "id": "fake-mcq",
            "scoring": "mcq",
            "answer": "B",
            "choices": ["A. Alice", "B. Bob", "C. Carol"],
            "subtype": "sanity",
        },
        {
            "id": "fake-list",
            "scoring": "list",
            "answer": "anne, bob, carol",
            "choices": None,
            "subtype": "sanity",
        },
    ]
    fake_responses = ["The answer is (B).", "Anne, Bob and Carol"]
    print("\nScorer sanity checks:")
    for fake_item, fake_resp in zip(fake_items, fake_responses):
        scored = score_item(fake_item, fake_resp)
        print(
            f"  - {fake_item['id']}: pred={scored.get('prediction')} "
            f"correct={scored.get('correct')} list_f1={scored.get('list_f1', '-')}"
        )
    print("=" * 60 + "\n")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Offline ToM eval harness (FanToM)")
    p.add_argument("--model", required=True, help="Model name served at the endpoint")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--tasks", default="fantom")
    p.add_argument("--n", type=int, default=0, help="Max items per task (0=all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--out-dir", default=str(script_dir / "results"))
    p.add_argument("--data-dir", default=str(script_dir / "data"))
    p.add_argument("--dry-run", action="store_true", help="Load data and test scoring; no endpoint calls")
    return p.parse_args()


async def main_async(args):
    tasks = parse_tasks(args.tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_model_name(args.model)
    redacted_base_url = redact_base_url(args.base_url)

    task_to_items: dict[str, list[dict]] = {}
    for task in tasks:
        task_to_items[task] = load_task_items(task, data_dir=args.data_dir, n=args.n, seed=args.seed)

    if args.dry_run:
        _print_dry_run_summary(task_to_items)
        return {}

    client = make_client(args.base_url)
    all_aggregates: dict[str, dict] = {}

    for task in tasks:
        items = task_to_items[task]
        t0 = time.time()
        results = await eval_task(
            client=client,
            model=args.model,
            task=task,
            items=items,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            no_think=args.no_think,
        )
        elapsed = time.time() - t0
        agg = aggregate_task_results(results, elapsed_s=elapsed)
        cfg = {
            "model": args.model,
            "base_url": redacted_base_url,
            "task": task,
            "n": len(items),
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "no_think": args.no_think,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"config": cfg, "aggregate": agg, "results": results}
        out_path = out_dir / f"{safe_model}_{task}_n{len(items)}.json"
        out_path.write_text(json.dumps(payload, indent=2))

        print_task_report(cfg, agg)
        print(
            f"[summary] task={task} n={agg['n']} acc={agg['accuracy']:.1%} "
            f"errors={agg['n_errors']} elapsed_s={agg['elapsed_s']}"
        )
        all_aggregates[task] = agg

    summary_payload = {
        "config": {
            "model": args.model,
            "base_url": redacted_base_url,
            "tasks": tasks,
            "n_per_task": args.n,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "no_think": args.no_think,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "aggregates": all_aggregates,
    }
    (out_dir / f"{safe_model}_summary.json").write_text(json.dumps(summary_payload, indent=2))
    return all_aggregates


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
