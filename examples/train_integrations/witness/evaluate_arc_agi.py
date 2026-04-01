"""
ARC-AGI-3 Evaluation Script — learned_adaptive_ascii mode.

Evaluates a trained Qwen3.5-9B checkpoint on the 25 official ARC-AGI-3
public games using the model's VLM capability to perceive raw pixel frames.

The model receives each 64×64 frame as an RGB image and must:
1. Analyze the visual scene (in its thinking)
2. Internally construct an ASCII representation
3. Output an action: <action>NUMBER</action>

Usage:
  python evaluate_arc_agi.py \\
    --checkpoint ~/checkpoint/ \\
    --max_turns 200 \\
    --output_dir ~/eval_results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── ARC color palette (16 colors, same as witness_grid.py) ─────────
ARC_PALETTE = [
    (255, 255, 255),  # 0: white
    (128, 128, 128),  # 1: light gray
    (192, 192, 192),  # 2: gray
    (64, 64, 64),     # 3: dark gray
    (32, 32, 32),     # 4: near black
    (0, 0, 0),        # 5: black
    (200, 0, 200),    # 6: magenta
    (200, 128, 200),  # 7: light magenta
    (255, 0, 0),      # 8: red
    (0, 0, 255),      # 9: blue
    (128, 128, 255),  # 10: light blue
    (255, 255, 0),    # 11: yellow
    (255, 165, 0),    # 12: orange
    (128, 64, 64),    # 13: maroon
    (0, 200, 0),      # 14: green
    (128, 0, 128),    # 15: purple
]

ACTION_NAMES = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "CONFIRM", 6: "ACTION6", 7: "ACTION7",
}


def frame_to_image(frame: np.ndarray, scale: int = 8):
    """Convert 64×64 int grid to upscaled RGB PIL Image."""
    from PIL import Image
    h, w = frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            idx = int(frame[r, c])
            rgb[r, c] = ARC_PALETTE[min(idx, len(ARC_PALETTE) - 1)]
    img = Image.fromarray(rgb)
    return img.resize((h * scale, w * scale), Image.NEAREST)


def extract_frame(obs) -> np.ndarray:
    """Extract 64×64 numpy array from SDK observation."""
    # SDK may return frame in different formats
    if hasattr(obs, 'frame'):
        frame = obs.frame
        if isinstance(frame, list):
            if len(frame) > 0:
                arr = frame[0] if isinstance(frame[0], np.ndarray) else np.array(frame[0])
                return arr.astype(np.int32)
            return np.zeros((64, 64), dtype=np.int32)
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3:
                return frame[0].astype(np.int32)
            return frame.astype(np.int32)
    # Fallback: try to get raw frame data
    if hasattr(obs, 'observation') and hasattr(obs.observation, 'frame'):
        return extract_frame(obs.observation)
    return np.zeros((64, 64), dtype=np.int32)


def parse_action(text: str) -> int:
    """Extract action ID from model output."""
    matches = re.findall(r"<action>\s*(\d+)\s*</action>", text)
    if matches:
        return int(matches[-1])
    name_matches = re.findall(r"<action>\s*(\w+)\s*</action>", text)
    if name_matches:
        name = name_matches[-1].upper()
        name_to_id = {v: k for k, v in ACTION_NAMES.items()}
        if name in name_to_id:
            return name_to_id[name]
    bare = re.findall(r"\b([1-7])\b", text)
    if bare:
        return int(bare[-1])
    return 1  # default UP


def build_actions_desc(available_actions: List[int]) -> str:
    """Build dynamic action description from available actions."""
    parts = []
    for a in sorted(available_actions):
        name = ACTION_NAMES.get(a, f"ACTION{a}")
        parts.append(f"{a}={name}")
    return "  ".join(parts)


SYSTEM_PROMPT_TEMPLATE = """You are playing an interactive puzzle game. You see a 64×64 pixel grid as an image.

Your task:
1. OBSERVE the frame carefully — identify objects, cursor, grid structure, constraints
2. DESCRIBE what you see as a compact ASCII representation in your thinking
3. DECIDE your action based on your analysis

Available actions: {actions_desc}

Your goal: figure out the puzzle rules by exploring, then solve all levels.
Respond with ONLY your chosen action: <action>NUMBER</action>
For example: <action>4</action> to move RIGHT."""


MAX_HISTORY_TURNS = 5


def trim_history(chat: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Keep system prompt + most recent max_turns rounds of conversation."""
    if len(chat) <= 1 + max_turns * 2:
        return chat
    system = chat[0]
    recent = chat[-(max_turns * 2):]
    return [system] + recent


def _make_game_action(action_id: int):
    """Convert int action_id to arcengine GameAction enum."""
    from arcengine import GameAction
    mapping = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
    }
    # For ACTION6+, try direct enum access
    if action_id in mapping:
        return mapping[action_id]
    try:
        return GameAction(action_id)
    except (ValueError, KeyError):
        return GameAction.ACTION1  # fallback


def evaluate_game(
    arcade,
    game_id: str,
    llm,
    tokenizer,
    sampling_params,
    args,
) -> Dict[str, Any]:
    """Evaluate model on a single ARC-AGI-3 game."""

    start_time = time.time()

    env = arcade.make(game_id)
    obs = env.reset()
    frame = extract_frame(obs)

    # Detect available actions and total levels
    available_actions = getattr(obs, 'available_actions', [1, 2, 3, 4, 5])
    if not available_actions:
        available_actions = [1, 2, 3, 4, 5]
    num_actions = max(available_actions)
    total_levels = getattr(obs, 'total_levels', None)
    if total_levels is None:
        total_levels = getattr(env, 'total_levels', 10)

    actions_desc = build_actions_desc(available_actions)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(actions_desc=actions_desc)

    levels_completed = 0
    total_actions = 0
    prev_action_name = "NONE"
    chat = []

    print(f"  {game_id}: starting ({total_levels} levels, {num_actions} actions)...", end="", flush=True)

    for turn in range(args.max_turns):
        # Frame → image
        image = frame_to_image(frame)

        # Build user message
        user_text = (
            f"Game: {game_id} | Level: {levels_completed}/{total_levels} | "
            f"Step: {total_actions}/{args.max_turns}\n"
            f"Previous action: {prev_action_name}\n"
            f"Analyze the frame, then choose your action."
        )

        # For chat template: use image_url format (Qwen VL convention)
        import base64
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]

        if not chat:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            chat.append({"role": "user", "content": user_content})
            chat = trim_history(chat)

        # Generate via vLLM
        try:
            prompt_ids = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=True
            )

            # Pass image as multi_modal_data for vLLM
            outputs = llm.generate(
                [{"prompt_token_ids": prompt_ids, "multi_modal_data": {"image": [image]}}],
                sampling_params,
            )
            response = outputs[0].outputs[0].text
        except Exception as e:
            print(f" ERROR at step {turn}: {e}")
            response = "<action>1</action>"  # fallback

        # Parse action
        action_id = parse_action(response)
        action_id = min(max(action_id, 1), num_actions)
        prev_action_name = ACTION_NAMES.get(action_id, f"ACTION{action_id}")

        # Execute
        game_action = _make_game_action(action_id)
        obs = env.step(game_action)
        frame = extract_frame(obs)
        total_actions += 1

        # Check level completion
        new_completed = getattr(obs, 'levels_completed', 0)
        if new_completed > levels_completed:
            levels_completed = new_completed
            print(f" [L{levels_completed}]", end="", flush=True)

        # All cleared?
        if levels_completed >= total_levels:
            break

        # Append assistant response to history
        chat.append({"role": "assistant", "content": response})

    elapsed = time.time() - start_time
    score = levels_completed / max(total_levels, 1)
    print(f" → {levels_completed}/{total_levels} ({elapsed:.0f}s)")

    return {
        "game_id": game_id,
        "levels_completed": levels_completed,
        "total_levels": total_levels,
        "total_actions": total_actions,
        "num_available_actions": num_actions,
        "score": score,
        "elapsed_s": round(elapsed, 1),
    }


def print_report(results: List[Dict]):
    """Print formatted evaluation report."""
    print()
    print("=" * 70)
    print("ARC-AGI-3 Evaluation Report (learned_adaptive_ascii)")
    print("=" * 70)

    total_solved = sum(r["levels_completed"] for r in results)
    total_levels = sum(r["total_levels"] for r in results)

    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        bar = "█" * int(r["score"] * 20)
        print(f"  {r['game_id']:>8s}  {bar:<20s}  "
              f"{r['levels_completed']:>2d}/{r['total_levels']:<2d}  "
              f"score={r['score']:.3f}  "
              f"actions={r['total_actions']}  "
              f"({r['elapsed_s']:.0f}s)")

    print("-" * 70)
    overall = sum(r["score"] for r in results) / len(results) if results else 0
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"  Games: {len(results)}  |  Levels: {total_solved}/{total_levels}  |  "
          f"Overall score: {overall:.4f}  |  Time: {total_time:.0f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--max_turns", type=int, default=200,
                        help="Max actions per game (default 200)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory for results JSON")
    parser.add_argument("--games", nargs="+", default=None,
                        help="Specific game IDs to evaluate (default: all)")
    args = parser.parse_args()

    print("ARC-AGI-3 Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  TP: {args.tp}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoTokenizer
    import vllm
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = vllm.LLM(
        model=args.checkpoint,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        stop=["</action>"],
    )
    print("Model loaded.\n")

    # Discover games (layered fallback)
    print("Discovering ARC-AGI-3 games...")
    from arc_agi import Arcade, OperationMode

    if args.games:
        # User specified games explicitly
        arcade = Arcade()
        game_ids = args.games
        print(f"  Using user-specified games: {game_ids}")
    else:
        # Try 1: Default mode (auto-downloads assets if needed)
        game_ids = []
        try:
            arcade = Arcade()
            game_ids = [e.game_id for e in arcade.get_environments()]
            print(f"  Default mode: found {len(game_ids)} games")
        except Exception as e:
            print(f"  Default mode failed: {e}")

        # Try 2: Offline with local arc-witness-envs
        if not game_ids:
            try:
                env_dir = os.path.expanduser("~/arc-witness-envs/environment_files")
                if os.path.isdir(env_dir):
                    arcade = Arcade(
                        operation_mode=OperationMode.OFFLINE,
                        environments_dir=env_dir,
                    )
                    game_ids = [e.game_id for e in arcade.get_environments()]
                    print(f"  Local arc-witness-envs: found {len(game_ids)} games")
            except Exception as e:
                print(f"  Local fallback failed: {e}")

        # Try 3: Hardcoded known game IDs as last resort
        if not game_ids:
            game_ids = [
                "tw01", "tw02", "tw03", "tw04", "tw05", "tw06", "tw07",
                "tw08", "tw09", "tw10", "tw11", "tw12", "tw13",
            ]
            try:
                env_dir = os.path.expanduser("~/arc-witness-envs/environment_files")
                arcade = Arcade(
                    operation_mode=OperationMode.OFFLINE,
                    environments_dir=env_dir,
                )
            except Exception:
                arcade = Arcade()
            print(f"  Using hardcoded fallback: {len(game_ids)} games")

    print(f"  Final game list ({len(game_ids)}): {game_ids[:5]}{'...' if len(game_ids) > 5 else ''}\n")

    # Evaluate each game
    print("Evaluating...")
    results = []
    for game_id in game_ids:
        try:
            result = evaluate_game(arcade, game_id, llm, tokenizer, sampling_params, args)
            results.append(result)
        except Exception as e:
            print(f"  {game_id}: FAILED — {e}")
            results.append({
                "game_id": game_id,
                "levels_completed": 0,
                "total_levels": 0,
                "total_actions": 0,
                "score": 0.0,
                "elapsed_s": 0,
                "error": str(e),
            })

    # Report
    print_report(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_results.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "max_turns": args.max_turns,
        "mode": "learned_adaptive_ascii",
        "results": results,
        "overall_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "total_levels_completed": sum(r["levels_completed"] for r in results),
        "total_levels": sum(r["total_levels"] for r in results),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
