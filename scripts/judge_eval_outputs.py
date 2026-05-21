#!/usr/bin/env python3
"""Score SkyRL eval dumps with the taste judge rubric and summarize verifier association."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic


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


def image_media_type(data: bytes, path: Path | None = None) -> str:
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
    media_type = image_media_type(data, path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


async def judge_one(
    client: AsyncAnthropic,
    row: dict[str, Any],
    input_path: Path,
    *,
    text_model: str,
    visual_model: str,
    n_screenshots: int,
) -> dict[str, Any]:
    action_count = count_actions(row)
    d1 = efficiency_score(action_count)
    verifier = verifier_score(row.get("score"))
    env_extras = row.get("env_extras") or {}
    task_key = env_extras.get("task_key") or row.get("task_key") or ""
    data_source = row.get("data_source") or env_extras.get("data_source") or ""
    transcript = (row.get("output_response") or "")[-24000:]

    text_prompt = f"""TASK
{task_text(row)}

METADATA
task_key={task_key} env={data_source} verifier_score={verifier} estimated_actions={action_count} D1_efficiency={d1}/5

TRAJECTORY TRANSCRIPT
{transcript}

Score D2, D4, and D5 now. JSON only."""

    text_task = client.messages.create(
        model=text_model,
        max_tokens=500,
        system=TEXT_SYSTEM,
        messages=[{"role": "user", "content": text_prompt}],
    )

    image_paths = resolve_images(row, input_path, n_screenshots)
    if image_paths:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"TASK:\n{task_text(row)}\n\nTRANSCRIPT EXCERPT:\n{transcript[-6000:]}\n\nSampled screenshots:"}
        ]
        for p in image_paths:
            content.append(image_block(p))
        visual_task = client.messages.create(
            model=visual_model,
            max_tokens=400,
            system=VISUAL_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text_resp, visual_resp = await asyncio.gather(text_task, visual_task)
        visual_out = extract_json(visual_resp.content[0].text)
    else:
        text_resp = await text_task
        visual_out = {"D3": 3, "rationale": "screenshots unavailable"}

    text_out = extract_json(text_resp.content[0].text)
    d2 = clamp_dim(text_out.get("D2"))
    d3 = clamp_dim(visual_out.get("D3"))
    d4 = clamp_dim(text_out.get("D4"))
    d5 = clamp_dim(text_out.get("D5"))

    taste_sum_all = d1 + d2 + d3 + d4 + d5
    taste_sum_rl = d2 + d3 + d4 + d5
    return {
        "task_key": task_key,
        "data_source": data_source,
        "verifier_score": verifier,
        "raw_score": row.get("score"),
        "estimated_actions": action_count,
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


def pearson(xs: list[float], ys: list[float]) -> float | None:
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
    good = [r for r in results if "error" not in r]
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
        "n": len(results),
        "n_scored": len(good),
        "n_errors": len(results) - len(good),
        "pass_rate": None if not good else round(sum(y) / len(y), 4),
        "association": assoc,
    }


async def main_async(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.input) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out_path = Path(args.out)
    summary_path = Path(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                obj = json.loads(line)
                if obj.get("task_key"):
                    done.add(obj["task_key"])
            except Exception:
                pass

    sem = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()
    results: list[dict[str, Any]] = []

    async def run_row(row: dict[str, Any]) -> None:
        async with sem:
            try:
                res = await judge_one(
                    client,
                    row,
                    Path(args.input),
                    text_model=args.text_model,
                    visual_model=args.visual_model,
                    n_screenshots=args.num_screenshots,
                )
            except Exception as exc:
                env_extras = row.get("env_extras") or {}
                res = {"task_key": env_extras.get("task_key", ""), "error": str(exc)}
            async with lock:
                results.append(res)
                with open(out_path, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"[{len(results)}/{len(rows)}] {res.get('task_key','')[:32]} {'ERR' if 'error' in res else 'ok'}", flush=True)

    await asyncio.gather(*(run_row(row) for row in rows if ((row.get("env_extras") or {}).get("task_key") not in done)))

    all_results = [json.loads(line) for line in open(out_path) if line.strip()]
    summary = summarize(all_results)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-screenshots", type=int, default=8)
    parser.add_argument("--text-model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--visual-model", default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
