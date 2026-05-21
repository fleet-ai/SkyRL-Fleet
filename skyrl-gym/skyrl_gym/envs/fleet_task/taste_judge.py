"""Taste reward scorer for Fleet browser-use trajectories.

This mirrors the v1 rubric used by ``scripts/judge_v1.py`` and returns the
training reward:

    0.5 * verifier_score + (D2 + D3 + D4 + D5) / 40
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call


TEXT_SYSTEM = """You score one computer-use trajectory on three quality criteria. Return STRICT JSON only, no markdown.

The agent uses screenshots and mouse/keyboard actions to complete a natural-language task. A separate verifier already determined pass/fail. Score HOW the agent executed, not whether it succeeded.

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
{"D2": <int>, "D4": <int>, "D5": <int>, "rationale": "<one or two sentences citing specific turns>"}"""


VISUAL_SYSTEM = """You score visual grounding for a computer-use trajectory. Return STRICT JSON only.

You will see N (screenshot, agent_reasoning_after_screenshot) pairs. Judge whether the reasoning accurately reflects what is visible.

Rate each pair: "accurate" (1.0) / "vague" (0.5) / "wrong" (0.0).

Aggregate to D3 score 1-5:
  5 if mean >= 0.95
  4 if mean >= 0.85
  3 if mean >= 0.70
  2 if mean >= 0.50
  1 if mean  < 0.50

Output schema:
{"per_pair": [{"turn_index": <int>, "rating": "accurate"|"vague"|"wrong"}, ...], "D3": <int>, "rationale": "<one sentence>"}"""


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON found in judge output: {text[:200]!r}")
    return json.loads(match.group(0))


def _clamp_dim(value: Any, default: int = 3) -> int:
    try:
        return max(1, min(5, int(value)))
    except Exception:
        return default


def _task_text(chat_history: List[Dict[str, Any]]) -> str:
    for msg in chat_history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
    return "(no task text found)"


def _format_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "image_url":
                parts.append("[screenshot]")
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p).strip()
    return "" if content is None else str(content)


def _format_trajectory(chat_history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    turn_idx = 0
    for msg in chat_history:
        role = msg.get("role")
        if role == "system":
            continue
        content = _format_content(msg.get("content"))
        if role == "assistant":
            turn_idx += 1
            parsed = parse_tool_call(content)
            action = json.dumps(parsed, separators=(",", ":")) if parsed else "(none)"
            reason = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
            lines.append(f"[T{turn_idx}] reason: {reason or '(no narration)'}\n        action: {action}")
        elif role == "user" and content:
            if "[screenshot]" not in content:
                lines.append(f"        obs: {content[:300]}")
        elif role == "tool" and content:
            lines.append(f"        tool: {content[:300]}")
    return "\n".join(lines)


def _data_url_to_anthropic_source(url: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"data:(image/[^;]+);base64,(.+)$", url or "", re.DOTALL)
    if not match:
        return None
    # Validate base64 early so bad screenshots do not poison the judge request.
    data = base64.b64decode(match.group(2), validate=False)
    media_type = match.group(1)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        media_type = "image/webp"
    return {"type": "base64", "media_type": media_type, "data": match.group(2)}


def _sample_screenshot_pairs(chat_history: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for i, msg in enumerate(chat_history):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        url = None
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = block.get("image_url", {}).get("url")
                break
        if not url:
            continue
        for nxt in chat_history[i + 1 :]:
            if nxt.get("role") == "assistant":
                pairs.append({"turn_index": i, "url": url, "reasoning": _format_content(nxt.get("content"))[:600]})
                break
    if len(pairs) <= n:
        return pairs
    step = len(pairs) / n
    return [pairs[int(i * step)] for i in range(n)]


def _text_prompt(
    chat_history: List[Dict[str, Any]],
    *,
    task_key: str,
    env_key: str,
    verifier_score: float,
    num_turns: int,
) -> str:
    return f"""TASK
{_task_text(chat_history)}

METADATA
task_key={task_key} env={env_key} turns={num_turns} verifier_score={verifier_score}

TRAJECTORY
{_format_trajectory(chat_history)}

Score D2, D4, D5 now. JSON only."""


def _visual_messages(pairs: List[Dict[str, Any]], task_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"TASK: {task_text}\n\nScore these screenshot/reasoning pairs.\n"}
    ]
    for idx, pair in enumerate(pairs, start=1):
        source = _data_url_to_anthropic_source(pair["url"])
        content.append({"type": "text", "text": f"[PAIR {idx}, turn_index={pair['turn_index']}]"})
        if source:
            content.append({"type": "image", "source": source})
        else:
            content.append({"type": "text", "text": "(image unavailable; rate as vague by default)"})
        content.append({"type": "text", "text": f"Agent reasoning after this screenshot:\n{pair['reasoning']}\n"})
    content.append({"type": "text", "text": "Return the JSON now."})
    return [{"role": "user", "content": content}]


async def _score_text(
    client: Any,
    model: str,
    chat_history: List[Dict[str, Any]],
    task_key: str,
    env_key: str,
    verifier_score: float,
    num_turns: int,
) -> Dict[str, Any]:
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        system=TEXT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": _text_prompt(
                    chat_history,
                    task_key=task_key,
                    env_key=env_key,
                    verifier_score=verifier_score,
                    num_turns=num_turns,
                ),
            }
        ],
    )
    return _extract_json(resp.content[0].text)


async def _score_visual(
    client: Any,
    model: str,
    chat_history: List[Dict[str, Any]],
    n_screenshots: int,
) -> Dict[str, Any]:
    pairs = _sample_screenshot_pairs(chat_history, n_screenshots)
    if not pairs:
        return {"D3": 3, "rationale": "no screenshots available", "per_pair": []}
    resp = await client.messages.create(
        model=model,
        max_tokens=800,
        system=VISUAL_SYSTEM,
        messages=_visual_messages(pairs, _task_text(chat_history)),
    )
    return _extract_json(resp.content[0].text)


async def score_taste_reward_async(
    *,
    chat_history: List[Dict[str, Any]],
    task_key: str,
    env_key: str,
    verifier_score: float,
    num_turns: int,
    text_model: str = "claude-sonnet-4-5-20250929",
    visual_model: str = "claude-sonnet-4-5-20250929",
    skip_visual: bool = False,
    n_screenshots: int = 8,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Return reward and judge breakdown for one completed trajectory."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    verifier_binary = 1.0 if float(verifier_score) >= 1.0 else 0.0

    text_task = _score_text(client, text_model, chat_history, task_key, env_key, verifier_binary, num_turns)
    if skip_visual:
        text_out = await asyncio.wait_for(text_task, timeout=timeout)
        visual_out = {"D3": 3, "rationale": "visual scoring skipped", "per_pair": []}
    else:
        text_out, visual_out = await asyncio.wait_for(
            asyncio.gather(text_task, _score_visual(client, visual_model, chat_history, n_screenshots)),
            timeout=timeout,
        )

    d2 = _clamp_dim(text_out.get("D2"))
    d3 = _clamp_dim(visual_out.get("D3"))
    d4 = _clamp_dim(text_out.get("D4"))
    d5 = _clamp_dim(text_out.get("D5"))
    dims_sum = d2 + d3 + d4 + d5
    reward = 0.5 * verifier_binary + dims_sum / 40.0

    return {
        "reward": round(reward, 6),
        "verifier_score": verifier_binary,
        "D2_redundancy": d2,
        "D3_visual_grounding": d3,
        "D4_consistency": d4,
        "D5_recovery": d5,
        "taste_sum": dims_sum,
        "formula": "0.5*verifier_score + (D2+D3+D4+D5)/40",
        "text_rationale": text_out.get("rationale", ""),
        "visual_rationale": visual_out.get("rationale", ""),
        "visual_per_pair": visual_out.get("per_pair", []),
        "text_model": text_model,
        "visual_model": None if skip_visual else visual_model,
    }
