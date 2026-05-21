#!/usr/bin/env python3
"""Run the taste LLM judge over a local SkyRL eval JSONL with a tqdm progress bar.

This script assumes the eval dump has already been downloaded locally:
  - ticketmaster.jsonl
  - images/...

It writes one JSON object per trajectory to --out and can resume by skipping
task_keys already present in that file.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from anthropic import AsyncAnthropic
from tqdm import tqdm


TEXT_SYSTEM = """You score one browser-use trajectory on execution quality. Return STRICT JSON only.

The agent used screenshots and mouse/keyboard actions to complete a task. A separate verifier already scored task success/failure. Score HOW the agent executed, not whether it succeeded.

Score each 1-5 integer:

D2 REDUNDANCY - repeated actions, looping, thrashing
  5 = every action moves to a new state
  3 = some repeats, but agent makes progress
  1 = severe thrashing

D4 CONSISTENCY - does stated intent match the action taken?
  5 = every announced action executes as described
  3 = half the actions lack narration, or 2-3 mismatches
  1 = narration unrelated to actions

D5 RECOVERY - when something fails, does the agent diagnose and adopt a new approach?
  5 = clean execution OR every recovery resolved within 2 turns
  3 = mixed; some slow recoveries
  1 = thrashes on the same failure without diagnosis

Output schema:
{"D2": <int>, "D4": <int>, "D5": <int>, "rationale": "<one or two sentences citing evidence>"}"""


VISUAL_SYSTEM = """You score visual grounding for a browser-use trajectory. Return STRICT JSON only.

You will see screenshots sampled from the trajectory and an agent transcript. Judge whether the agent's stated observations/actions are grounded in visible UI state.

Score D3 visual grounding 1-5:
  5 = visible UI evidence consistently supports the transcript
  3 = mixed or mostly vague grounding
  1 = frequent hallucinated, stale, or contradicted visual claims

Output schema:
{"D3": <int>, "rationale": "<one sentence>"}"""


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>|\bcomputer(?:_use)?\s*\(|\"action\"\s*:", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    return json.loads(match.group(0))


def clamp_dim(value: Any, default: int = 3) -> int:
    try:
        return max(1, min(5, int(value)))
    except Exception:
        return default


def task_text(row: dict[str, Any]) -> str:
    prompt = row.get("input_prompt") or ""
    if "<|im_start|>user" in prompt:
        prompt = prompt.split("<|im_start|>user", 1)[-1]
    return prompt[-2000:].strip() or "(task text unavailable)"


def count_actions(row: dict[str, Any]) -> int:
    text = row.get("output_response") or ""
    return max(1, len(TOOL_CALL_RE.findall(text)))


def efficiency_score(action_count: int) -> int:
    if action_count <= 10:
        return 5
    if action_count <= 20:
        return 4
    if action_count <= 35:
        return 3
    if action_count <= 55:
        return 2
    return 1


def verifier_score(raw_score: Any) -> float:
    if isinstance(raw_score, list):
        values = []
        for item in raw_score:
            try:
                values.append(float(item))
            except Exception:
                pass
        return 1.0 if values and max(values) >= 1.0 else 0.0
    try:
        return 1.0 if float(raw_score or 0.0) >= 1.0 else 0.0
    except Exception:
        return 0.0


def task_key_for(row: dict[str, Any], fallback: str) -> str:
    env_extras = row.get("env_extras") or {}
    base_key = env_extras.get("session_id") or row.get("session_id") or env_extras.get("task_key") or row.get("task_key") or "trajectory"
    return f"{base_key}::{fallback}"


def resolve_images(row: dict[str, Any], input_path: Path, limit: int) -> list[Path]:
    paths = []
    images_dir = input_path.parent / "images"
    for raw in row.get("image_paths") or []:
        p = Path(raw)
        candidates = [p]
        if not p.is_absolute():
            candidates.append(input_path.parent / p)
        candidates.append(images_dir / p.name)
        found = next((c for c in candidates if c.exists()), None)
        if found:
            paths.append(found)
    if len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def image_media_type(data: bytes, path: Optional[Path] = None) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if path is not None and path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def image_block(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image_media_type(data, path),
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


async def with_retries(coro_factory, retries: int) -> Any:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            await asyncio.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


async def json_with_retries(coro_factory, retries: int) -> dict[str, Any]:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = await coro_factory()
            return extract_json(resp.content[0].text)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            await asyncio.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


async def judge_one(
    client: AsyncAnthropic,
    row: dict[str, Any],
    input_path: Path,
    *,
    trajectory_key: str,
    text_model: str,
    visual_model: str,
    n_screenshots: int,
    skip_visual: bool,
    retries: int,
) -> dict[str, Any]:
    action_count = count_actions(row)
    d1 = efficiency_score(action_count)
    verifier = verifier_score(row.get("score"))
    env_extras = row.get("env_extras") or {}
    session_id = str(env_extras.get("session_id") or row.get("session_id") or "")
    original_task_key = str(env_extras.get("task_key") or row.get("task_key") or "")
    task_key = trajectory_key
    data_source = str(row.get("data_source") or env_extras.get("data_source") or "")
    transcript = (row.get("output_response") or "")[-24000:]

    text_prompt = f"""TASK
{task_text(row)}

METADATA
trajectory_key={task_key} task_key={original_task_key} env={data_source} verifier_score={verifier} estimated_actions={action_count} D1_efficiency={d1}/5

TRAJECTORY TRANSCRIPT
{transcript}

Score D2, D4, and D5 now. Return exactly one JSON object and no prose outside the JSON."""

    async def text_call() -> Any:
        return await client.messages.create(
            model=text_model,
            max_tokens=500,
            system=TEXT_SYSTEM,
            messages=[{"role": "user", "content": text_prompt}],
        )

    image_paths = [] if skip_visual else resolve_images(row, input_path, n_screenshots)
    if image_paths:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"TASK:\n{task_text(row)}\n\nTRANSCRIPT EXCERPT:\n{transcript[-6000:]}\n\nSampled screenshots:\n\nScore D3 now. Return exactly one JSON object and no prose outside the JSON.",
            }
        ]
        for path in image_paths:
            content.append(image_block(path))

        async def visual_call() -> Any:
            return await client.messages.create(
                model=visual_model,
                max_tokens=400,
                system=VISUAL_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )

        text_out, visual_out = await asyncio.gather(
            json_with_retries(text_call, retries),
            json_with_retries(visual_call, retries),
        )
    else:
        text_out = await json_with_retries(text_call, retries)
        visual_out = {"D3": 3, "rationale": "screenshots unavailable or visual judging skipped"}

    d2 = clamp_dim(text_out.get("D2"))
    d3 = clamp_dim(visual_out.get("D3"))
    d4 = clamp_dim(text_out.get("D4"))
    d5 = clamp_dim(text_out.get("D5"))

    taste_sum_all = d1 + d2 + d3 + d4 + d5
    taste_sum_rl = d2 + d3 + d4 + d5
    return {
        "task_key": task_key,
        "original_task_key": original_task_key,
        "session_id": session_id,
        "data_source": data_source,
        "verifier_score": verifier,
        "raw_score": row.get("score"),
        "estimated_actions": action_count,
        "n_images_used": len(image_paths),
        "D1_efficiency": d1,
        "D2_redundancy": d2,
        "D3_visual_grounding": d3,
        "D4_consistency": d4,
        "D5_recovery": d5,
        "taste_all_5": round(taste_sum_all / 25.0, 4),
        "taste_rl_4": round(taste_sum_rl / 20.0, 4),
        "rl_reward_formula": round(0.5 * verifier + taste_sum_rl / 40.0, 4),
        "text_rationale": text_out.get("rationale", ""),
        "visual_rationale": visual_out.get("rationale", ""),
        "text_model": text_model,
        "visual_model": visual_model if image_paths else None,
    }


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latest_by_key = {}
    for i, result in enumerate(results):
        key = str(result.get("task_key") or f"row_{i}")
        latest_by_key[key] = result
    latest = list(latest_by_key.values())
    good = [r for r in latest if "error" not in r]
    y = [float(r["verifier_score"]) for r in good]
    dims = ["D1_efficiency", "D2_redundancy", "D3_visual_grounding", "D4_consistency", "D5_recovery", "taste_all_5", "taste_rl_4"]
    assoc = {}
    for dim in dims:
        xs = [float(r[dim]) for r in good]
        corr = pearson(xs, y)
        pass_vals = [x for x, yy in zip(xs, y) if yy >= 1.0]
        fail_vals = [x for x, yy in zip(xs, y) if yy < 1.0]
        assoc[dim] = {
            "pearson_vs_verifier": None if corr is None else round(corr, 4),
            "pass_mean": None if not pass_vals else round(sum(pass_vals) / len(pass_vals), 4),
            "fail_mean": None if not fail_vals else round(sum(fail_vals) / len(fail_vals), 4),
            "pass_minus_fail": None if not pass_vals or not fail_vals else round(sum(pass_vals) / len(pass_vals) - sum(fail_vals) / len(fail_vals), 4),
        }
    return {
        "n": len(latest),
        "n_scored": len(good),
        "n_errors": len(latest) - len(good),
        "pass_rate": None if not good else round(sum(y) / len(y), 4),
        "association": assoc,
    }


def load_done(out_path: Path, retry_errors: bool) -> set[str]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("task_key")
            if key and ("error" not in obj or not retry_errors):
                done.add(str(key))
    return done


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else input_path.with_name("taste_scores.jsonl")
    summary_path = Path(args.summary) if args.summary else input_path.with_name("taste_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.open() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = load_done(out_path, retry_errors=args.retry_errors)
    pending = [(i, row) for i, row in enumerate(rows) if task_key_for(row, f"row_{i}") not in done]
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()

    async def run_row(i: int, row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            key = task_key_for(row, f"row_{i}")
            try:
                res = await judge_one(
                    client,
                    row,
                    input_path,
                    trajectory_key=key,
                    text_model=args.text_model,
                    visual_model=args.visual_model,
                    n_screenshots=args.num_screenshots,
                    skip_visual=args.skip_visual,
                    retries=args.retries,
                )
                if not res.get("task_key"):
                    res["task_key"] = key
            except Exception as exc:
                res = {"task_key": key, "error": str(exc)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
            return res

    print(f"Input: {input_path}")
    print(f"Output: {out_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(rows)} total, {len(done)} already done, {len(pending)} pending")

    tasks = [asyncio.create_task(run_row(i, row)) for i, row in pending]
    errors = 0
    with tqdm(total=len(pending), desc="taste judge", unit="traj") as pbar:
        for task in asyncio.as_completed(tasks):
            res = await task
            errors += int("error" in res)
            pbar.set_postfix(errors=errors)
            pbar.update(1)

    all_results = [json.loads(line) for line in out_path.open() if line.strip()]
    summary = summarize(all_results)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local eval JSONL, e.g. local_runs/.../ticketmaster.jsonl")
    parser.add_argument("--out", help="Incremental JSONL output. Defaults to taste_scores.jsonl next to input.")
    parser.add_argument("--summary", help="Summary JSON. Defaults to taste_summary.json next to input.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-screenshots", type=int, default=8)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="Retry rows that already have error records in --out.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--text-model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--visual-model", default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
