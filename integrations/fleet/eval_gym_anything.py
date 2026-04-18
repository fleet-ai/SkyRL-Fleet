#!/usr/bin/env python3
"""Evaluate Gemini (or any litellm model) on gym-anything CUA tasks.

Runs multi-turn rollouts against a gym-anything remote server, collects
verifier rewards, and dumps per-task results to JSONL.

Usage:
    python -m integrations.fleet.eval_gym_anything \
        --model gemini/gemini-2.5-flash \
        --tasks tasks_gym_anything.json \
        --server http://<ip>:5000 \
        --max-turns 25 \
        --concurrency 20 \
        --output results.jsonl
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
litellm.suppress_debug_info = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reuse gym-anything's Gemini system prompt structure
SYSTEM_PROMPT = """You are controlling a desktop application via mouse and keyboard. Complete the task by interacting with the GUI.

Each turn you receive a screenshot of the current screen. Respond with exactly ONE action.

Available actions (use <tool_call> format):
<tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "right_click", "coordinate": [x, y]}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "double_click", "coordinate": [x, y]}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "type", "text": "hello"}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "s"]}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "scroll", "coordinate": [x, y], "pixels": 3}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "drag", "coordinate": [x1, y1], "coordinate2": [x2, y2]}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "screenshot"}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "wait", "time": 2}}</tool_call>
<tool_call>{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}</tool_call>

Coordinates use a [0, 999] grid. (0,0) is top-left, (999,999) is bottom-right.
Click the CENTER of UI elements. After each action, you'll receive a new screenshot.
When the task is complete, use action=terminate with status=success.
Do NOT keep clicking after the task is done."""


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool_call from model response."""
    import re
    for tag in ["tool_call", "function_call"]:
        match = re.search(rf"<{tag}>(.*?)(?:</{tag}>|\Z)", text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                for extra in range(1, 4):
                    try:
                        parsed = json.loads(raw + "}" * extra)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            name = parsed.get("name") or parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("params", {})
            if name:
                return {"name": name, "arguments": args}
    return None


def scale_coord(x: int, y: int, w: int = 1920, h: int = 1080) -> Tuple[int, int]:
    return int(x / 1000 * w), int(y / 1000 * h)


def tool_call_to_actions(tc: Dict[str, Any]) -> Tuple[List[Dict], bool]:
    """Convert parsed tool call to gym-anything action dicts. Returns (actions, is_terminal)."""
    args = tc.get("arguments", {})
    action = args.get("action", "")

    if action == "terminate":
        return [], True
    if action == "screenshot":
        return [{"action": "screenshot"}], False
    if action == "wait":
        return [{"action": "wait", "time": args.get("time", 1.0)}], False
    if action == "key":
        keys = args.get("keys", [])
        return [{"keyboard": {"keys": keys if isinstance(keys, list) else [keys]}}], False
    if action == "type":
        actions = []
        if args.get("clear"):
            actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
        actions.append({"keyboard": {"text": args.get("text", "")}})
        return actions, False
    if action in ("click", "left_click"):
        x, y = scale_coord(*args.get("coordinate", [500, 500]))
        return [{"mouse": {"left_click": [x, y]}}], False
    if action == "right_click":
        x, y = scale_coord(*args.get("coordinate", [500, 500]))
        return [{"mouse": {"right_click": [x, y]}}], False
    if action == "double_click":
        x, y = scale_coord(*args.get("coordinate", [500, 500]))
        return [{"mouse": {"double_click": [x, y]}}], False
    if action in ("drag", "left_click_drag"):
        c1 = args.get("coordinate", [500, 500])
        c2 = args.get("coordinate2", c1)
        x1, y1 = scale_coord(c1[0], c1[1])
        x2, y2 = scale_coord(c2[0], c2[1])
        return [{"mouse": {"left_click_drag": [[x1, y1], [x2, y2]]}}], False
    if action == "scroll":
        actions = []
        if "coordinate" in args:
            x, y = scale_coord(*args["coordinate"])
            actions.append({"mouse": {"move": [x, y]}})
        actions.append({"mouse": {"scroll": int(args.get("pixels", 0))}})
        return actions, False
    if action == "mouse_move":
        x, y = scale_coord(*args.get("coordinate", [500, 500]))
        return [{"mouse": {"move": [x, y]}}], False

    return [{"action": "screenshot"}], False


def obs_to_b64(obs: Dict[str, Any]) -> Optional[str]:
    screen = obs.get("screen", {})
    if "png_b64" in screen:
        return screen["png_b64"]
    path = screen.get("path")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    return None


async def call_model(messages: List[Dict], model: str, temperature: float = 0.5) -> str:
    """Call model via litellm (async)."""
    for attempt in range(5):
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                timeout=120,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            logger.warning(f"Empty response from {model}, retrying")
        except Exception as e:
            logger.warning(f"Model call failed (attempt {attempt+1}/5): {e}")
            await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Failed to get response from {model} after 5 attempts")


async def run_task(
    task: Dict[str, Any],
    server_url: str,
    model: str,
    max_turns: int,
    temperature: float,
    sem: asyncio.Semaphore,
    save_screenshots: bool = False,
    screenshot_base_dir: str = "/tmp/gym_anything_screenshots",
) -> Dict[str, Any]:
    """Run a single task: create env, multi-turn rollout, return result."""
    import requests

    task_key = task["task_key"]
    env_dir = task["env_dir"]
    task_id = task["task_id"]
    description = task.get("description", "Complete the task.")

    async with sem:
        start = time.time()
        env_id = None
        try:
            # Create env
            resp = await asyncio.to_thread(
                requests.post,
                f"{server_url}/envs/create",
                json={"env_dir": env_dir, "task_id": task_id},
                timeout=60,
            )
            resp.raise_for_status()
            env_id = resp.json()["env_id"]

            # Reset
            resp = await asyncio.to_thread(
                requests.post,
                f"{server_url}/envs/{env_id}/reset",
                json={"use_cache": True, "cache_level": "post_start"},
                timeout=600,
            )
            resp.raise_for_status()
            reset_result = resp.json()
            if reset_result.get("error"):
                return {"task_key": task_key, "env_name": task.get("env_name", ""), "reward": 0.0, "turns": 0, "error": reset_result["error"], "elapsed": time.time() - start}

            # Wait for app to load past splash screens before first screenshot
            await asyncio.sleep(10)

            # Capture fresh screenshot after wait (the reset screenshot may be stale)
            try:
                resp = await asyncio.to_thread(
                    requests.post,
                    f"{server_url}/envs/{env_id}/step",
                    json={"actions": [{"action": "screenshot"}]},
                    timeout=300,
                )
                resp.raise_for_status()
                obs = (resp.json().get("observation") or {}).get("screen") or {}
            except Exception:
                obs = (reset_result.get("observation") or {}).get("screen") or {}

            # Build initial messages
            png_b64 = obs.get("png_b64", "") if isinstance(obs, dict) else ""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Task: {description}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}} if png_b64 else {"type": "text", "text": "[screenshot unavailable]"},
                ]},
            ]

            reward = 0.0
            turns = 0

            for turn in range(max_turns):
                turns = turn + 1

                # Call model
                try:
                    response_text = await call_model(messages, model, temperature)
                except Exception as e:
                    logger.warning(f"{task_key} turn {turn}: model call failed: {e}")
                    break

                messages.append({"role": "assistant", "content": response_text})

                # Parse action
                tc = parse_tool_call(response_text)
                if not tc:
                    messages.append({"role": "user", "content": "No tool call found. Use <tool_call> format."})
                    continue

                actions, is_terminal = tool_call_to_actions(tc)

                if is_terminal:
                    # Mark done, get reward
                    resp = await asyncio.to_thread(
                        requests.post,
                        f"{server_url}/envs/{env_id}/step",
                        json={"actions": [{"action": "screenshot"}], "mark_done": True},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    step_result = resp.json()
                    verifier = (step_result.get("info") or {}).get("verifier") or {}
                    reward = verifier.get("score", 0) / 100.0
                    break

                if not actions:
                    continue

                # Execute action
                mark_done = (turn == max_turns - 1)
                resp = await asyncio.to_thread(
                    requests.post,
                    f"{server_url}/envs/{env_id}/step",
                    json={"actions": actions, "mark_done": mark_done},
                    timeout=300,
                )
                resp.raise_for_status()
                step_result = resp.json()

                if mark_done:
                    verifier = (step_result.get("info") or {}).get("verifier") or {}
                    reward = verifier.get("score", 0) / 100.0
                    break

                # Get screenshot for next turn
                step_obs = step_result.get("observation") or {}
                ss = (step_obs.get("screen") or {}).get("png_b64", "")
                if ss:
                    ss_url = f"data:image/png;base64,{ss}"
                    messages.append({"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": ss_url}},
                    ]})
                    # Save screenshot immediately if enabled
                    if save_screenshots:
                        try:
                            import base64 as b64mod
                            ss_dir = Path(screenshot_base_dir) / task_key.replace("/", "__")
                            ss_dir.mkdir(parents=True, exist_ok=True)
                            ss_path = ss_dir / f"turn_{turn:03d}.png"
                            ss_path.write_bytes(b64mod.b64decode(ss))
                        except Exception as e:
                            logger.debug(f"Screenshot save failed: {e}")
                else:
                    messages.append({"role": "user", "content": "[screenshot unavailable]"})

            elapsed = time.time() - start
            logger.info(f"{task_key}: reward={reward:.2f}, turns={turns}, time={elapsed:.0f}s")

            # Build trajectory
            trajectory = []
            screenshot_dir = None
            if save_screenshots:
                screenshot_dir = Path(screenshot_base_dir) / task_key.replace("/", "__")
                screenshot_dir.mkdir(parents=True, exist_ok=True)

            ss_idx = 0
            for msg in messages:
                m = dict(msg)
                if isinstance(m.get("content"), list):
                    new_content = []
                    for c in m["content"]:
                        if c.get("type") == "image_url":
                            url = c.get("image_url", {}).get("url", "")
                            if save_screenshots and screenshot_dir and url.startswith("data:image"):
                                # Save screenshot to disk
                                import base64 as b64mod
                                b64_data = url.split("base64,")[1] if "base64," in url else ""
                                if b64_data:
                                    ss_path = screenshot_dir / f"turn_{ss_idx:03d}.png"
                                    ss_path.write_bytes(b64mod.b64decode(b64_data))
                                    new_content.append({"type": "image_url", "image_url": {"url": str(ss_path)}})
                                    ss_idx += 1
                                else:
                                    new_content.append({"type": "image_url", "image_url": {"url": "[screenshot]"}})
                            else:
                                new_content.append({"type": "image_url", "image_url": {"url": "[screenshot]"}})
                        else:
                            new_content.append(c)
                    m["content"] = new_content
                trajectory.append(m)

            return {"task_key": task_key, "env_name": task.get("env_name", ""), "reward": reward, "turns": turns, "elapsed": elapsed, "trajectory": trajectory}

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{task_key}: error: {e}")
            return {"task_key": task_key, "env_name": task.get("env_name", ""), "reward": 0.0, "turns": 0, "error": str(e), "elapsed": elapsed}

        finally:
            if env_id:
                try:
                    await asyncio.to_thread(
                        requests.post, f"{server_url}/envs/{env_id}/close", timeout=10,
                    )
                except Exception:
                    pass


async def run_eval(
    tasks: List[Dict],
    server_url: str,
    model: str,
    max_turns: int,
    concurrency: int,
    temperature: float,
    output_path: str,
    save_screenshots: bool = False,
    screenshot_base_dir: str = "/tmp/gym_anything_screenshots",
):
    sem = asyncio.Semaphore(concurrency)
    results = []

    # Run all tasks concurrently (bounded by semaphore)
    coros = [run_task(t, server_url, model, max_turns, temperature, sem, save_screenshots, screenshot_base_dir) for t in tasks]
    for i, coro in enumerate(asyncio.as_completed(coros), 1):
        result = await coro
        results.append(result)

        # Write incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        if i % 10 == 0 or i == len(tasks):
            n_pass = sum(1 for r in results if r.get("reward", 0) > 0)
            logger.info(f"Progress: {i}/{len(tasks)}, pass={n_pass}/{i} ({n_pass/i*100:.1f}%)")

    # Summary
    n_pass = sum(1 for r in results if r.get("reward", 0) > 0)
    avg_reward = sum(r.get("reward", 0) for r in results) / len(results) if results else 0
    avg_turns = sum(r.get("turns", 0) for r in results) / len(results) if results else 0
    n_errors = sum(1 for r in results if "error" in r)

    logger.info(f"\n=== FINAL ===")
    logger.info(f"Tasks: {len(results)}, Pass: {n_pass} ({n_pass/len(results)*100:.1f}%), Avg reward: {avg_reward:.4f}")
    logger.info(f"Avg turns: {avg_turns:.1f}, Errors: {n_errors}")

    # Per-env breakdown
    env_results = {}
    for r in results:
        env = r.get("env_name", "unknown")
        if env not in env_results:
            env_results[env] = []
        env_results[env].append(r.get("reward", 0))

    logger.info(f"\nPer-env pass@1:")
    for env in sorted(env_results):
        rewards = env_results[env]
        pass_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        if pass_rate > 0:
            logger.info(f"  {env}: {pass_rate*100:.1f}% ({sum(1 for r in rewards if r > 0)}/{len(rewards)})")


def main():
    parser = argparse.ArgumentParser(description="Eval Gemini on gym-anything CUA tasks")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash", help="litellm model string")
    parser.add_argument("--tasks", required=True, help="Path to task index JSON")
    parser.add_argument("--server", required=True, help="gym-anything server URL")
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--output", default="eval_results.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N tasks (for testing)")
    parser.add_argument("--env-filter", default=None, help="Comma-separated env names to include")
    parser.add_argument("--save-screenshots", action="store_true", help="Save screenshots to disk per task")
    parser.add_argument("--screenshot-dir", default="/tmp/gym_anything_screenshots", help="Directory for saved screenshots")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip server smoke test")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)

    if args.env_filter:
        allowed = set(args.env_filter.split(","))
        tasks = [t for t in tasks if t.get("env_name") in allowed]

    if args.limit:
        tasks = tasks[:args.limit]

    # Smoke test: verify server produces valid screenshots before committing to full eval
    if not args.skip_smoke_test:
        logger.info("Running smoke test against server...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/gym-anything-smoke-test.py", args.server, "--timeout", "600"],
                timeout=1800,
            )
            if result.returncode != 0:
                logger.error("Smoke test FAILED. Server not ready. Use --skip-smoke-test to override.")
                sys.exit(1)
            logger.info("Smoke test passed.")
        except FileNotFoundError:
            logger.warning("Smoke test script not found, skipping. Run from SkyRL repo root.")
        except subprocess.TimeoutExpired:
            logger.error("Smoke test timed out after 30 min.")
            sys.exit(1)

    logger.info(f"Evaluating {len(tasks)} tasks with {args.model}, max_turns={args.max_turns}, concurrency={args.concurrency}")

    # Clear output file
    open(args.output, "w").close()

    asyncio.run(run_eval(tasks, args.server, args.model, args.max_turns, args.concurrency, args.temperature, args.output, args.save_screenshots, args.screenshot_dir))


if __name__ == "__main__":
    main()
