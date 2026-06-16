"""
Fleet Task Training with Tinker Backend.

This entrypoint uses Tinker (hosted) for training and inference,
combined with Fleet environments via OpenEnv for rollout collection.

Usage:
    python -m integrations.fleet.entrypoints.main_fleet_tinker \
        --model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
        --tasks-file /path/to/tasks.json \
        --dataset-file /path/to/train.parquet \
        --eval-dataset-file /path/to/validation.parquet

Environment Variables:
    TINKER_API_KEY: Tinker API key for authentication (required)
    TINKER_API_URL: Tinker service URL (optional, SDK uses default if not set)
    FLEET_API_KEY: Fleet API key for environment access
    WANDB_API_KEY: Weights & Biases API key for logging

Architecture:
    1. Load tasks from JSON file (same format as SkyRL Fleet integration)
    2. For each training step:
       a. Save current model weights for sampling
       b. Create SamplingClient from Tinker
       c. Collect rollouts using FleetTaskEnv (OpenEnv) + Tinker inference
       d. Compute GRPO advantages
       e. Train using Tinker's forward_backward + optim_step
    3. Checkpoints saved via Tinker API

Metrics (matching SkyRL):
    - reward/avg_pass_at_{n}: Pass@k across all prompts
    - reward/variance_per_prompt: Mean within-prompt reward variance (GRPO learning signal)
    - reward/{env_key}/pass_at_{n}: Per-environment pass@k
    - reward/{env_key}/variance_per_prompt: Per-environment variance (learning signal)
    - eval/all/pass_at_{n}: Evaluation pass@k
    - eval/{env_key}/pass_at_{n}: Per-environment eval pass@k
"""

import asyncio
import base64
import io
import logging
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
import tinker
import torch
import wandb
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Use SkyRL's FleetTaskEnv wrapper (now supports async via init_async/step_async)
from omegaconf import OmegaConf
from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

# Import SkyRL's overlong filtering for parity
from skyrl.train.generators.utils import (
    apply_overlong_filtering,
    decode_base64_image,
    is_multimodal_conversation,
    is_multimodal_message,
)

# Import shared metrics module for consistent metric calculation with SkyRL trainer
from integrations.fleet.reward_metrics import (
    compute_pass_at_n as _compute_pass_at_n,
    compute_reward_metrics,
    compute_per_group_metrics,
    sanitize_metric_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Per-await timeouts. Tinker's 120-min progress_timeout is shared across the
# SamplingClient and starvation-prone under concurrent rollouts.
TINKER_SAMPLE_TIMEOUT_S = 600
ENV_INIT_TIMEOUT_S = 90
ENV_STEP_TIMEOUT_S = 120
ENV_CLOSE_TIMEOUT_S = 30

# Thread pool for env operations - isolates MCP connections per thread (like SkyRL)
_env_executor: ThreadPoolExecutor = None


def _get_env_executor(max_workers: int = 16) -> ThreadPoolExecutor:
    global _env_executor
    if _env_executor is None:
        _env_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fleet-env-")
    return _env_executor


async def _run_in_executor(func, *args):
    """Run sync function in thread pool - each thread gets isolated event loop/connections."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_env_executor(), func, *args)


async def _env_init(env, *args, **kwargs):
    """Prefer FleetTaskEnv.init_async when present so the verifier's Fleet
    client stays on the running event loop. Falling back to env.init via the
    thread pool causes 'Event loop is closed' once a worker thread's loop is
    torn down between rollouts. Pattern lifted from
    skyrl/train/generators/skyrl_gym_generator.py."""
    if hasattr(env, "init_async"):
        return await env.init_async(*args, **kwargs)
    return await _run_in_executor(env.init, *args)


async def _env_step(env, action):
    """Async-preferring env.step wrapper. See _env_init for rationale."""
    if hasattr(env, "step_async"):
        return await env.step_async(action)
    return await _run_in_executor(env.step, action)


async def _env_close(env):
    """Async-preferring env.close wrapper. See _env_init for rationale."""
    if hasattr(env, "close_async"):
        return await env.close_async()
    return await _run_in_executor(env.close)


class RolloutOutput(BaseModel):
    """Output from a single rollout collection."""

    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float]
    loss_mask: List[int]
    reward: float
    task_key: str
    env_key: str
    turns: int
    tool_calls: int
    tool_errors: int = 0  # Count of tool call errors in this rollout
    stop_reason: str
    duration: float
    # Timing breakdown for WandB
    total_gen_time: float = 0.0  # Total Tinker generation time
    total_step_time: float = 0.0  # Total MCP/Fleet step time
    total_tokens: int = 0  # Total tokens generated
    error: Optional[str] = None

    class Config:
        frozen = True


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have mean 0 and std 1."""
    if not advantages or len(advantages) == 1:
        return advantages
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return [0.0] * len(advantages)
    return [(a - mean) / (std + 1e-8) for a in advantages]


def compute_advantages_grpo(
    rewards: List[float],
    group_size: int = None,
    normalize: bool = True,
) -> List[float]:
    """
    GRPO (Group Relative Policy Optimization) advantage estimation.

    For each group of trajectories from the same prompt, compute advantages
    as deviations from the group mean.
    """
    rewards = np.array(rewards)

    if group_size is None:
        group_size = len(rewards)

    n_groups = len(rewards) // group_size
    advantages = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        advantages.extend(group_advantages.tolist())

    remaining = len(rewards) % group_size
    if remaining > 0:
        remaining_rewards = rewards[-remaining:]
        remaining_mean = remaining_rewards.mean()
        advantages.extend((remaining_rewards - remaining_mean).tolist())

    if normalize:
        advantages = normalize_advantages(advantages)

    return advantages


def compute_pass_at_n(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> float:
    """
    Compute pass@n metric using the shared metrics module.

    For each unique prompt (task_key), if ANY of the n trajectories has reward > 0,
    that counts as a "pass".

    This function is a thin wrapper around the shared compute_pass_at_n for backward
    compatibility with the rollout dict format.
    """
    rewards = [r.get("reward", 0.0) for r in rollouts]
    uids = [r.get("task_key", "unknown") for r in rollouts]
    return _compute_pass_at_n(rewards, uids)


def compute_per_env_metrics(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> Dict[str, float]:
    """
    Compute per-environment metrics using the shared metrics module.

    This function is a thin wrapper around the shared compute_per_group_metrics for
    backward compatibility with the rollout dict format.
    """
    rewards = [r.get("reward", 0.0) for r in rollouts]
    uids = [r.get("task_key", "unknown") for r in rollouts]
    env_keys = [r.get("env_key", "unknown") for r in rollouts]

    return compute_per_group_metrics(
        rewards=rewards,
        uids=uids,
        groups=env_keys,
        n_samples_per_prompt=n_samples_per_prompt,
        prefix="reward",
    )


def compute_rollout_metrics(
    rollouts: List[Dict[str, Any]],
    valid_rollouts: List[Dict[str, Any]],
    rewards: List[float],
    advantages: List[float],
    n_samples_per_prompt: int,
) -> Dict[str, Any]:
    """
    Compute all rollout metrics using the shared metrics module.

    Args:
        rollouts: All rollouts (including failed ones)
        valid_rollouts: Only valid rollouts
        rewards: Rewards for valid rollouts
        advantages: GRPO advantages for valid rollouts
        n_samples_per_prompt: Number of samples per prompt

    Returns:
        Dict of metrics for wandb logging
    """
    metrics = {}

    # Extract data for shared module
    uids = [r.get("task_key", "unknown") for r in valid_rollouts]
    env_keys = [r.get("env_key", "unknown") for r in valid_rollouts]

    # Core reward metrics using shared module
    core_metrics = compute_reward_metrics(rewards, uids, n_samples_per_prompt)
    metrics[f"reward/avg_pass_at_{n_samples_per_prompt}"] = core_metrics[f"pass_at_{n_samples_per_prompt}"]
    metrics["reward/avg_raw_reward"] = np.mean(rewards)
    metrics["reward/variance_per_prompt"] = core_metrics["variance_per_prompt"]
    metrics["reward/mean_positive_reward"] = core_metrics["mean_positive_reward"]

    # Advantage metrics (Tinker-specific)
    metrics["advantage/mean"] = np.mean(advantages)
    metrics["advantage/std"] = np.std(advantages)
    metrics["rollouts/valid"] = len(valid_rollouts)
    metrics["rollouts/total"] = len(rollouts)

    # Per-environment reward metrics using shared module
    per_env_metrics = compute_per_group_metrics(
        rewards=rewards,
        uids=uids,
        groups=env_keys,
        n_samples_per_prompt=n_samples_per_prompt,
        prefix="reward",
    )
    metrics.update(per_env_metrics)

    # Per-environment rollout stats (turns, tool_calls, tool_errors, duration) - Tinker-specific
    rollout_stats = defaultdict(list)
    for r in valid_rollouts:
        env_key = sanitize_metric_key(r.get("env_key", "unknown"))
        rollout_stats[f"rollout/{env_key}/turns"].append(r.get("turns", 0))
        rollout_stats[f"rollout/{env_key}/tool_calls"].append(r.get("tool_calls", 0))
        rollout_stats[f"rollout/{env_key}/tool_errors"].append(r.get("tool_errors", 0))
        rollout_stats[f"rollout/{env_key}/duration"].append(r.get("duration", 0.0))

    for key, values in rollout_stats.items():
        metrics[key] = np.mean(values)

    # Compute tool error rate per environment
    env_keys_seen = set()
    for r in valid_rollouts:
        env_keys_seen.add(sanitize_metric_key(r.get("env_key", "unknown")))
    for env_key in env_keys_seen:
        total_calls = sum(rollout_stats[f"rollout/{env_key}/tool_calls"])
        total_errors = sum(rollout_stats[f"rollout/{env_key}/tool_errors"])
        if total_calls > 0:
            metrics[f"rollout/{env_key}/tool_error_rate"] = total_errors / total_calls
        else:
            metrics[f"rollout/{env_key}/tool_error_rate"] = 0.0

    # Overall rollout duration stats
    durations = [r.get("duration", 0.0) for r in valid_rollouts]
    metrics["rollout/avg_duration"] = np.mean(durations)
    metrics["rollout/max_duration"] = np.max(durations)
    metrics["rollout/min_duration"] = np.min(durations)

    return metrics


def prepare_training_data(
    rollouts: List[Dict[str, Any]],
    advantages: List[float],
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
) -> tuple:
    """
    Prepare training data from rollouts (matching SkyRL's generate_batched pattern).

    Applies:
    1. DAPO overlong filtering (zero loss mask if response doesn't end with EOS)
    2. Sequence truncation for max_sequence_length
    3. Builds Tinker Datum objects for training

    Args:
        rollouts: List of rollout dicts with prompt_ids, response_ids, logprobs, loss_mask
        advantages: GRPO advantages for each rollout
        tokenizer: Tokenizer for EOS token ID
        max_sequence_length: Maximum sequence length for training

    Returns:
        Tuple of (training_datums, truncated_count)
    """
    # Apply DAPO overlong filtering (zero out loss mask for truncated responses)
    all_loss_masks = [r.loss_mask for r in rollouts]
    stop_reasons = [r.stop_reason for r in rollouts]
    filtered_loss_masks = apply_overlong_filtering(all_loss_masks, stop_reasons)

    training_datums = []
    truncated_count = 0

    for idx, rollout in enumerate(rollouts):
        prompt_ids = rollout.prompt_ids
        response_ids = rollout.response_ids
        logprobs = rollout.logprobs
        loss_mask_data = filtered_loss_masks[idx]

        full_sequence = prompt_ids + response_ids
        prompt_len = len(prompt_ids)

        # Truncate sequences exceeding model's max length for Tinker API
        if len(full_sequence) > max_sequence_length:
            truncated_count += 1
            full_sequence = full_sequence[:max_sequence_length]
            response_len = len(full_sequence) - prompt_len
            response_ids = response_ids[:response_len]
            logprobs = logprobs[:response_len] if logprobs else []
            loss_mask_data = loss_mask_data[:response_len]

        # Ensure logprobs and response_ids are in sync before building training data
        if len(logprobs) != len(response_ids):
            logger.warning(
                f"Datum {idx}: logprobs ({len(logprobs)}) != response_ids ({len(response_ids)}), fixing"
            )
            if len(logprobs) > len(response_ids):
                logprobs = logprobs[: len(response_ids)]
            else:
                logprobs = logprobs + [0.0] * (len(response_ids) - len(logprobs))

        # Target tokens (shifted by 1)
        target_tokens = full_sequence[1:]
        seq_len = len(target_tokens)

        # Logprobs (0 for prompt, actual for response)
        full_logprobs = [0.0] * prompt_len + logprobs
        full_logprobs = full_logprobs[1:]

        # Loss mask (0 for prompt, actual for response)
        full_mask = [0] * prompt_len + loss_mask_data
        full_mask = full_mask[1:]

        # Safety: ensure all arrays match target_tokens length
        full_logprobs = full_logprobs[:seq_len] + [0.0] * max(0, seq_len - len(full_logprobs))
        full_mask = full_mask[:seq_len] + [0] * max(0, seq_len - len(full_mask))

        # Advantages (apply only where loss mask is 1)
        advantage_value = advantages[idx]
        full_advantages = torch.zeros(len(full_sequence))
        for i in range(prompt_len, len(full_sequence)):
            if i - 1 < len(full_mask) and full_mask[i - 1] > 0:
                full_advantages[i] = advantage_value
        full_advantages = full_advantages[1:]

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(full_logprobs)),
                "advantages": TensorData.from_torch(full_advantages),
            },
        )
        training_datums.append(datum)

    return training_datums, truncated_count


def tokenize_chat(
    tokenizer: AutoTokenizer,
    chat_history: List[Dict],
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Tokenize chat history and ensure we get a plain list of token IDs.

    apply_chat_template can return different types depending on the tokenizer:
    - List[int] for some tokenizers
    - BatchEncoding dict with 'input_ids' key for others

    Tinker's ModelInput.from_ints() requires a plain list of integers.

    `tools`, when provided, is rendered via the HF chat-template `tools=`
    kwarg. For Kimi-K2 this becomes a single `<|im_system|>tool_declare<|im_middle|>...<|im_end|>`
    block at the top of the prompt (outside the message loop, so it does not
    repeat per turn). Tokenizers without tool support simply ignore the kwarg.
    """
    kwargs: Dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
    }
    if tools:
        kwargs["tools"] = tools
    result = tokenizer.apply_chat_template(chat_history, **kwargs)
    # Handle BatchEncoding (dict-like) vs plain list
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    elif isinstance(result, dict) and "input_ids" in result:
        return list(result["input_ids"])
    else:
        return list(result)


# -------------------------------------------------------------------------- #
# Vision-language support for Tinker rollouts
# -------------------------------------------------------------------------- #
# Tinker's `ModelInput` accepts a list of mixed `EncodedTextChunk` and
# `ImageChunk` (and `ImageAssetPointerChunk`). For VL models like
# Kimi-K2.6 we need to thread screenshot bytes through to the model instead
# of letting them tokenize as base64 garbage. Pattern ported from SkyRL's
# skyrl_gym_generator (`apply_chat_template_with_images`,
# `extract_images_from_conversation`, `_sanitize_messages_for_template`),
# adapted to Tinker's chunk API.
#
# Two streams:
#   1. Sampling: build a chunks list with EncodedTextChunk + ImageChunk so
#      the vision model actually sees the screenshots.
#   2. Training: tokenize the conversation text-only with `[image]`
#      placeholders. The training stream doesn't carry image bytes (would
#      need Datum-level chunk support); observations are loss-masked anyway
#      and the policy gradient flows through assistant text only.

# Sentinel string we splice into the chat-template-rendered text wherever an
# image appeared. Unicode control chars so it can't collide with anything a
# legitimate tokenizer would produce.
_IMAGE_PLACEHOLDER = "IMAGE"

# Local estimate for image token cost in the rollout's `max_input_length`
# bookkeeping. The Tinker server computes the AUTHORITATIVE per-image count
# from the raw bytes on its side, so we do NOT pass `expected_tokens=` on
# ImageChunk — if our estimate disagrees with the server's tokenization
# (e.g. Kimi-K2.6 returned 1372 tokens for a 1366x768 screenshot vs our 1024
# guess), Tinker rejects the request with `BadRequestError: Expected N
# tokens, got M from image`. This estimate is advisory only.
_DEFAULT_IMAGE_TOKENS = 1500

def _decode_data_url(url: str) -> Optional[Tuple[bytes, str]]:
    """Decode a `data:image/...;base64,...` URL into `(jpeg_bytes, "jpeg")`.

    Delegates to SkyRL's `decode_base64_image` (returns a PIL Image), then
    transcodes via PIL to JPEG bytes that Tinker `ImageChunk` accepts.
    Returns None for HTTP URLs (no fetcher) or decoding errors.
    """
    if not url or not url.startswith("data:image"):
        return None
    try:
        pil_img = decode_base64_image(url)
    except Exception as e:
        logger.warning(f"image decode failed: {e}")
        return None
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "jpeg"


def _sanitize_content(
    content: Any,
) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Walk a single message's `content`. Return `(text_with_placeholders, images)`.

    `images` is a list of `(image_bytes, format)` tuples in left-to-right
    order. Text segments are joined with `\\n`. Each image becomes
    `_IMAGE_PLACEHOLDER` in the returned string so the caller can splice
    `ImageChunk`s into the chunk stream at the right positions.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []
    parts: List[str] = []
    images: List[Tuple[bytes, str]] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                decoded = _decode_data_url(url)
                if decoded is not None:
                    images.append(decoded)
                    parts.append(_IMAGE_PLACEHOLDER)
                else:
                    # Couldn't decode (HTTP url, weird format, etc.). Emit a
                    # textual marker so the model knows an image was here.
                    parts.append("[image]")
            elif t == "image" and "image" in item:
                # Some envs pass {"type": "image", "image": <bytes-or-url>}.
                v = item["image"]
                if isinstance(v, bytes):
                    images.append((v, "jpeg"))  # assume jpeg; server will validate
                    parts.append(_IMAGE_PLACEHOLDER)
                elif isinstance(v, str):
                    decoded = _decode_data_url(v)
                    if decoded is not None:
                        images.append(decoded)
                        parts.append(_IMAGE_PLACEHOLDER)
                    else:
                        parts.append("[image]")
                else:
                    parts.append("[image]")
            else:
                # Unknown dict shape; stringify whatever text is in there.
                txt = item.get("text") or item.get("content") or ""
                parts.append(str(txt) if txt else "[unknown_content]")
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p), images


def build_model_input_chunks(
    tokenizer: AutoTokenizer,
    chat_history: List[Dict],
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Any], int]:
    """Build Tinker `ModelInput` chunks from a (possibly multimodal) chat history.

    Returns `(chunks, estimated_total_tokens)`. The token count includes
    text tokens plus `_DEFAULT_IMAGE_TOKENS` per image (advisory: the Tinker
    server computes the real count).

    Text-only conversations return a single `EncodedTextChunk`.
    Multimodal conversations interleave `EncodedTextChunk` with `ImageChunk`
    at the exact positions where images appeared in the original chat.

    `tools`, when provided, is rendered via the HF chat-template `tools=`
    kwarg (see `tokenize_chat` for the rationale). It appears once at the
    top of the rendered prompt regardless of how many messages or turns the
    chat has.

    If the chat template mangles the placeholder (some templates URL-encode
    or strip control chars), we fall back to text-only with `[image]`
    markers so the rollout doesn't crash.
    """
    # Fast path: text-only conversation, no images to splice. Skip the
    # per-message sanitization walk and just call the standard tokenize_chat.
    if not is_multimodal_conversation(chat_history):
        tokens = tokenize_chat(tokenizer, chat_history, add_generation_prompt, tools=tools)
        return [types.EncodedTextChunk(tokens=tokens)], len(tokens)

    # Multimodal path: sanitize each message; collect ordered images.
    text_msgs: List[Dict] = []
    all_images: List[Tuple[bytes, str]] = []
    for msg in chat_history:
        sanitized_text, imgs = _sanitize_content(msg.get("content"))
        text_msgs.append({**msg, "content": sanitized_text})
        all_images.extend(imgs)

    # Apply chat template to the placeholder'd text. Pass tools= so VL models
    # also get their native tool_declare block rendered once at the top.
    template_kwargs: Dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if tools:
        template_kwargs["tools"] = tools
    flat = tokenizer.apply_chat_template(text_msgs, **template_kwargs)
    if not isinstance(flat, str):
        flat = str(flat)

    if not all_images:
        # Defensive fallback (shouldn't reach here after the is_multimodal
        # check above, but in case sanitize emitted only [image] markers).
        tokens = tokenizer.encode(flat, add_special_tokens=False)
        return [types.EncodedTextChunk(tokens=tokens)], len(tokens)

    # Pass 3: split the rendered text on the placeholder and emit chunks.
    segments = flat.split(_IMAGE_PLACEHOLDER)
    if len(segments) - 1 != len(all_images):
        # Template ate the placeholder. Fall back: image -> [image] text marker.
        flat = flat.replace(_IMAGE_PLACEHOLDER, "[image]")
        tokens = tokenizer.encode(flat, add_special_tokens=False)
        logger.warning(
            "build_model_input_chunks: placeholder mismatch "
            f"(segments={len(segments)}, images={len(all_images)}); "
            "falling back to text-only [image] markers."
        )
        return [types.EncodedTextChunk(tokens=tokens)], len(tokens)

    chunks: List[Any] = []
    total = 0
    for i, seg in enumerate(segments):
        if seg:
            seg_tokens = tokenizer.encode(seg, add_special_tokens=False)
            if seg_tokens:
                chunks.append(types.EncodedTextChunk(tokens=seg_tokens))
                total += len(seg_tokens)
        if i < len(all_images):
            img_bytes, fmt = all_images[i]
            # NOTE: do NOT pass expected_tokens — Tinker validates the value
            # against the server-side tokenization and rejects on mismatch
            # (BadRequestError: "Expected N tokens, got M from image"). The
            # local _DEFAULT_IMAGE_TOKENS estimate stays in the running total
            # for `max_input_length` bookkeeping only.
            chunks.append(
                types.ImageChunk(
                    data=img_bytes,
                    format=fmt,
                )
            )
            total += _DEFAULT_IMAGE_TOKENS
    return chunks, total


def sanitize_text_only(content: Any) -> str:
    """Reduce a possibly-multimodal `content` to a single string with `[image]`
    placeholders. Used for the training stream where we don't carry image
    bytes through `forward_backward`.
    """
    text, _imgs = _sanitize_content(content)
    return text.replace(_IMAGE_PLACEHOLDER, "[image]")


async def collect_fleet_rollout(
    task_config: Dict[str, Any],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    max_generate_length: int = 2048,
    max_input_length: int = 30720,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_sequences: List[str] = None,
) -> Dict[str, Any]:
    """
    Collect a single trajectory using Fleet environment and Tinker inference.

    Uses SkyRL's FleetTaskEnv wrapper with async methods for environment interaction.

    Args:
        max_generate_length: Max tokens per generation step.
        max_input_length: Max context length before ending rollout (matching SkyRL).
    """
    rollout_start = time.time()

    task_key = task_config.get("task_key") or task_config.get("key")

    # Create SkyRL FleetTaskEnv wrapper.
    # TTL of 2 hours - some rollouts with many turns can take 30+ minutes.
    # `use_tools_channel=True` asks the env to omit the in-system-message tool
    # JSON dump; we pass tools via `apply_chat_template(tools=...)` so models
    # like Kimi-K2 / Qwen3+ see them in their native tool_declare channel.
    env_config = OmegaConf.create({"tasks_file": tasks_file, "ttl_seconds": 7200})
    extras = {"task_key": task_key, "max_turns": max_turns, "use_tools_channel": True}

    env = FleetTaskEnv(env_config=env_config, extras=extras)

    try:
        # Initialize environment in thread pool - isolates MCP connections.
        # On timeout, let the TimeoutError raise to collect_single_rollout's
        # generic Exception handler.
        chat_history, metadata = await asyncio.wait_for(
            _env_init(env, []), ENV_INIT_TIMEOUT_S
        )
        env_key = metadata.get("env_key", "unknown")
        env_tools = metadata.get("tools") or env.tools

        # Tokenize initial prompt
        prompt_ids = tokenize_chat(
            tokenizer, chat_history, add_generation_prompt=True, tools=env_tools
        )

        all_response_ids = []
        all_logprobs = []
        loss_mask = []
        done = False
        total_reward = 0.0
        stop_reason = "stop"
        # Timing accumulators for WandB
        total_gen_time = 0.0
        total_step_time = 0.0
        total_tokens = 0

        async def _force_verifier(reason: str) -> None:
            """Final-state escape hatch: trigger the verifier even when the
            rollout has to exit before issuing a real tool call (length cap,
            sample timeout, empty response). Without this, every length/timeout
            exit silently bypasses the verifier and the rollout returns
            reward=0 by default even when the env state would have scored.

            We call env.step_async("<done>") which lands in the env's
            agent-done-without-tool-call branch and sends {"done": True} to
            OpenEnv → _compute_reward() runs → reward reflects current state.
            Wrapped in best-effort; verifier failures fall back to 0.0.
            """
            nonlocal total_reward, done
            try:
                final_out = await asyncio.wait_for(
                    _env_step(env, "<done>"), ENV_STEP_TIMEOUT_S
                )
                total_reward = final_out.get("reward", 0.0) or 0.0
                done = True
                logger.info(
                    f"[{task_key}] Forced verifier on {reason} exit: reward={total_reward}"
                )
            except Exception as e:
                logger.warning(
                    f"[{task_key}] Forced verifier on {reason} exit failed: {e}"
                )

        while not done and env.turns < max_turns:
            turn_num = env.turns + 1  # 1-indexed for logging

            # Prepare input for Tinker (use env's chat_history). For multimodal
            # conversations (e.g. computer_use envs returning screenshots),
            # build_model_input_chunks emits interleaved EncodedTextChunk +
            # ImageChunk so the vision model actually sees images instead of
            # tokenizing base64 strings. Text-only path hits a fast return
            # equivalent to the prior tokenize_chat call. `tools=` renders the
            # native tool_declare block once at the top of the prompt — does
            # not repeat per turn.
            prompt_chunks, prompt_len = build_model_input_chunks(
                tokenizer, env.chat_history, add_generation_prompt=True, tools=env_tools
            )

            # Check context length limit (matching SkyRL's skyrl_gym_generator.py:274)
            if prompt_len > max_input_length:
                logger.info(
                    f"[{task_key}] Turn {turn_num}: context length ({prompt_len}) exceeds max ({max_input_length}), ending"
                )
                stop_reason = "length"
                await _force_verifier("length")
                break

            # Generate with Tinker
            gen_start = time.time()
            sampling_params_kwargs = {
                "max_tokens": max_generate_length,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop_sequences:
                sampling_params_kwargs["stop"] = stop_sequences
            sampling_params = types.SamplingParams(**sampling_params_kwargs)

            try:
                result = await asyncio.wait_for(
                    sampling_client.sample_async(
                        prompt=types.ModelInput(chunks=prompt_chunks),
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    TINKER_SAMPLE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{task_key}] Turn {turn_num}: tinker sample timeout ({TINKER_SAMPLE_TIMEOUT_S}s)")
                stop_reason = "tinker_timeout"
                await _force_verifier("tinker_timeout")
                break
            gen_time = time.time() - gen_start
            total_gen_time += gen_time

            if not result.sequences or len(result.sequences) == 0:
                logger.warning(f"[{task_key}] Turn {turn_num}: no sequences returned from Tinker")
                stop_reason = "no_sequences"
                await _force_verifier("no_sequences")
                break

            sequence = result.sequences[0]
            output_ids = sequence.tokens
            output_logprobs = sequence.logprobs if sequence.logprobs else []

            # Guard: logprobs must match token count (Tinker may return different lengths)
            if output_logprobs and len(output_logprobs) != len(output_ids):
                logger.warning(
                    f"[{task_key}] Turn {turn_num}: logprobs length ({len(output_logprobs)}) != tokens length ({len(output_ids)}), truncating/padding"
                )
                if len(output_logprobs) > len(output_ids):
                    output_logprobs = output_logprobs[: len(output_ids)]
                else:
                    output_logprobs = output_logprobs + [0.0] * (len(output_ids) - len(output_logprobs))

            # Decode output
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            # Collect trajectory data (assistant response tokens - trainable)
            all_response_ids.extend(output_ids)
            if output_logprobs:
                all_logprobs.extend(output_logprobs)
            else:
                all_logprobs.extend([0.0] * len(output_ids))
            loss_mask.extend([1] * len(output_ids))

            # Step environment in thread pool - isolates MCP connections
            step_start = time.time()
            try:
                step_output = await asyncio.wait_for(_env_step(env, output_text), ENV_STEP_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(f"[{task_key}] Turn {turn_num}: env step timeout ({ENV_STEP_TIMEOUT_S}s)")
                stop_reason = "env_step_timeout"
                await _force_verifier("env_step_timeout")
                break
            step_time = time.time() - step_start
            total_step_time += step_time
            total_tokens += len(output_ids)

            # Observation content for the training stream (loss-masked, no
            # image bytes; the model already SAW images via ImageChunks in the
            # sampling prompt). sanitize_text_only handles all observed
            # shapes from Fleet envs: dict-with-string-content, dict-with-
            # list-content (MCP multipart), raw list, raw string. Images are
            # replaced with the literal "[image]" marker so the training
            # token stream stays bounded and the model still sees that an
            # image appeared at that turn.
            if step_output["observations"]:
                obs = step_output["observations"][0]
                if isinstance(obs, dict):
                    raw = obs.get("content", "")
                else:
                    raw = obs
                obs_content = sanitize_text_only(raw)
                obs_ids = tokenizer.encode(obs_content, add_special_tokens=False)
                all_response_ids.extend(obs_ids)
                all_logprobs.extend([0.0] * len(obs_ids))
                loss_mask.extend([0] * len(obs_ids))

            total_reward = step_output["reward"]
            done = step_output["done"]

            # Single per-turn progress line. Without this, modalities whose
            # tool calls succeed silently (computer_use, browser_use) emit only
            # tokenizer warnings between "Progress: N/8" events and live
            # rollouts look indistinguishable from hung ones.
            logger.info(
                f"[{task_key}] Turn {turn_num}/{max_turns}: gen={gen_time:.1f}s "
                f"step={step_time:.1f}s tokens={len(output_ids)} prompt_len={prompt_len} "
                f"reward={total_reward} done={done} tool_calls={env.tool_calls} "
                f"tool_errors={env.tool_errors}"
            )

        return RolloutOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            logprobs=all_logprobs,
            loss_mask=loss_mask,
            reward=total_reward,
            task_key=task_key,
            env_key=env_key,
            turns=env.turns,
            tool_calls=env.tool_calls,
            tool_errors=env.tool_errors,
            stop_reason=stop_reason,
            duration=time.time() - rollout_start,
            total_gen_time=total_gen_time,
            total_step_time=total_step_time,
            total_tokens=total_tokens,
        )

    finally:
        try:
            await asyncio.wait_for(_env_close(env), ENV_CLOSE_TIMEOUT_S)
        except Exception as e:
            logger.warning(f"env close failed/timeout for {task_key}: {e}")


async def collect_batch_rollouts(
    batch: List[Dict[str, Any]],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    max_generate_length: int = 2048,
    max_input_length: int = 30720,
    n_samples_per_prompt: int = 1,
    max_concurrent: int = 8,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_sequences: List[str] = None,
) -> List[Dict[str, Any]]:
    """Collect rollouts for a batch of tasks with limited concurrency.

    Args:
        max_concurrent: Maximum number of concurrent Fleet environment connections.
            Now safe to increase since ThreadPoolExecutor isolates connections.
    """
    # Semaphore to limit concurrent Fleet environment connections
    semaphore = asyncio.Semaphore(max_concurrent)

    async def collect_single_rollout(task_config: Dict[str, Any], index: int) -> tuple:
        """Wrapper to collect a single rollout with error handling and concurrency limit."""
        async with semaphore:
            rollout_start = time.time()
            try:
                rollout = await collect_fleet_rollout(
                    task_config=task_config,
                    tasks_file=tasks_file,
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    max_turns=max_turns,
                    max_generate_length=max_generate_length,
                    max_input_length=max_input_length,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                )
                # Per-rollout completion marker so operators can see in-flight
                # success rate before the step-end pass@k flush.
                logger.info(
                    f"rollout done: task={rollout.task_key} env={rollout.env_key} "
                    f"reward={rollout.reward} turns={rollout.turns}/{max_turns} "
                    f"tool_calls={rollout.tool_calls} tool_errors={rollout.tool_errors} "
                    f"stop={rollout.stop_reason} dur={rollout.duration:.1f}s"
                )
                return index, rollout
            except Exception as e:
                logger.error(f"Failed to collect rollout for {task_config.get('task_key')}: {e}")
                return index, RolloutOutput(
                    prompt_ids=[],
                    response_ids=[],
                    logprobs=[],
                    loss_mask=[],
                    reward=0.0,
                    task_key=task_config.get("task_key", "unknown"),
                    env_key=task_config.get("env_key", "unknown"),
                    turns=0,
                    tool_calls=0,
                    tool_errors=0,
                    stop_reason="error",
                    error=str(e),
                    duration=time.time() - rollout_start,
                )

    # Create all rollout tasks (batch_size * n_samples_per_prompt)
    tasks = []
    index = 0
    for task_config in batch:
        for _ in range(n_samples_per_prompt):
            tasks.append(collect_single_rollout(task_config, index))
            index += 1

    total = len(tasks)
    logger.info(f"  Collecting {total} rollouts (max {max_concurrent} concurrent)...")
    rollouts = [None] * total
    completed = 0
    last_logged = 0
    log_interval = max(1, total // 4)  # Log at ~25%, 50%, 75%, 100%

    # Run rollouts with limited concurrency via semaphore
    for coro in asyncio.as_completed(tasks):
        idx, rollout = await coro
        rollouts[idx] = rollout
        completed += 1

        # Log progress at intervals
        if completed - last_logged >= log_interval or completed == total:
            logger.info(f"  Progress: {completed}/{total} rollouts completed")
            last_logged = completed

    return rollouts


def collate_fn(batch):
    """Return batch as-is without tensor collation."""
    return batch


async def main(
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    tasks_file: str = None,
    dataset_file: str = None,
    eval_dataset_file: str = None,
    batch_size: int = 8,
    eval_batch_size: int = 32,
    learning_rate: float = 4e-5,
    lora_rank: int = 16,
    max_steps: int = 200,
    max_turns: int = 50,
    max_generate_length: int = 2048,
    max_input_length: int = 30720,
    max_sequence_length: int = 32768,
    n_samples_per_prompt: int = 4,
    eval_every: int = 20,
    seed: int = 42,
    wandb_project: str = "fleet-tinker-grpo",
    wandb_name: str = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_sequences: List[str] = None,
    loss_fn: str = "ppo",
    eval_before_train: bool = False,
    results_out: str = None,
    save_state_every: int = 10,
    max_concurrent: int = 8,
):
    """
    Main training loop using Tinker for training/inference and Fleet for environments.
    """
    set_seed(seed)
    eval_entries: list[dict] = []  # populated by _run_eval below; sourced for results_out
    state_checkpoints: list[dict] = []  # path + step of every save_state call

    async def _run_eval(eval_sampling_client, step_index: int) -> float | None:
        if not eval_dataset:
            return None
        logger.info(f"Running held-out eval at step={step_index}...")
        eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        all_eval_rollouts = []
        for eval_batch in eval_dataloader:
            eval_rollouts = await collect_batch_rollouts(
                batch=eval_batch,
                tasks_file=tasks_file,
                sampling_client=eval_sampling_client,
                tokenizer=tokenizer,
                max_turns=max_turns,
                max_generate_length=max_generate_length,
                max_input_length=max_input_length,
                n_samples_per_prompt=1,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                max_concurrent=max_concurrent,
            )
            all_eval_rollouts.extend([r for r in eval_rollouts if not r.error])
        if not all_eval_rollouts:
            logger.warning(f"step={step_index}: eval produced no valid rollouts")
            return None
        eval_rewards = [r.reward for r in all_eval_rollouts]
        eval_rollouts_dicts = [r.model_dump() for r in all_eval_rollouts]
        eval_pass_at_1 = compute_pass_at_n(eval_rollouts_dicts, 1)
        eval_per_env = compute_per_env_metrics(eval_rollouts_dicts, 1)
        eval_metrics = {
            "eval/all/pass_at_1": eval_pass_at_1,
            "eval/all/mean_positive_reward": (
                np.mean([r for r in eval_rewards if r > 0]) if any(r > 0 for r in eval_rewards) else 0.0
            ),
            "eval/num_samples": len(all_eval_rollouts),
        }
        for key, value in eval_per_env.items():
            eval_metrics[key.replace("reward/", "eval/")] = value
        # step_index < 0 indicates pre-train eval; log at step 0 in WandB so the
        # curve has a baseline point without confusing wandb's monotonic step.
        wandb.log(eval_metrics, step=max(step_index, 0), commit=True)
        logger.info(f"step={step_index}: eval pass@1={eval_pass_at_1:.3f}, n={len(all_eval_rollouts)}")
        eval_entries.append({
            "step": step_index,
            "pass_at_1": float(eval_pass_at_1),
            "num_samples": len(all_eval_rollouts),
        })
        return eval_pass_at_1

    # Setup WandB run name
    if wandb_name is None:
        wandb_name = f"{model_name.split('/')[-1]}_{datetime.now().strftime('%m%d_%H%M')}"

    # Initialize WandB
    if stop_sequences is None:
        stop_sequences = []

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_turns": max_turns,
            "max_generate_length": max_generate_length,
            "max_input_length": max_input_length,
            "max_sequence_length": max_sequence_length,
            "n_samples_per_prompt": n_samples_per_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "loss_fn": loss_fn,
        },
    )

    # Load datasets
    train_dataset = load_dataset("parquet", data_files=dataset_file)["train"]
    eval_dataset = load_dataset("parquet", data_files=eval_dataset_file)["train"] if eval_dataset_file else None

    logger.info(f"Loaded {len(train_dataset)} training samples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} eval samples")

    # Fleet trace job: groups every rollout's full chat_history + reward under
    # one job_id in the Fleet UI so the user can inspect step-0 baseline traces
    # and post-train deltas side by side. Mirrors SkyRL's trainer.py pattern
    # (FleetEnv.set_trace_config before eval, clear in finally). Setting the
    # class-level config on FleetTaskEnv makes its existing step_async upload
    # path fire on every episode_done — no per-rollout plumbing needed.
    fleet_api_key = os.environ.get("FLEET_API_KEY")
    if fleet_api_key:
        try:
            from envs.fleet_env.trace import create_trace_job

            trace_job_name = f"tinker_{wandb_name}_{datetime.now().strftime('%m%d_%H%M')}"
            trace_job_id = await create_trace_job(fleet_api_key, trace_job_name)
            FleetTaskEnv.set_trace_config(job_id=trace_job_id, model=model_name)
            logger.info(
                f"Created Fleet trace job: {trace_job_id} (name={trace_job_name}). "
                "Every rollout's chat_history + reward will be uploaded to Fleet UI on episode_done."
            )
        except Exception as e:
            logger.warning(
                f"Failed to create Fleet trace job — traces will not be uploaded: {e}"
            )
    else:
        logger.info("FLEET_API_KEY unset; skipping Fleet trace job creation.")

    # Setup Tinker
    tinker_url = os.environ.get("TINKER_API_URL")
    tinker_api_key = os.environ.get("TINKER_API_KEY")

    service_client_kwargs = {}
    if tinker_url:
        service_client_kwargs["base_url"] = tinker_url
    if tinker_api_key:
        service_client_kwargs["api_key"] = tinker_api_key

    service_client = tinker.ServiceClient(**service_client_kwargs)
    training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=lora_rank)

    adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    # trust_remote_code is required for models like moonshotai/Kimi-K2.6 whose
    # tokenizer config references a custom processor class on the HF Hub.
    # Tinker exposes long-context PEFT variants with a `:peft:<context>` suffix
    # (e.g., "moonshotai/Kimi-K2.6:peft:131072"); strip it for the HF Hub lookup
    # since only the base repo name is a valid HF identifier.
    hf_model_name = model_name.split(":peft:")[0]
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    # Create dataloader
    def create_dataloader(epoch: int):
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + epoch),
        )

    steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
    current_epoch = 0
    train_dataloader = create_dataloader(current_epoch)
    train_iterator = iter(train_dataloader)

    # Pre-train eval: snapshot the base (untrained) LoRA weights and evaluate
    # on the held-out split so we have a baseline for the post-train delta.
    pre_sampling_path: str | None = None
    final_sampling_path: str | None = None
    if eval_before_train and eval_dataset:
        pre_sampling_path = training_client.save_weights_for_sampler(name="step_pretrain").result().path
        pre_sampling_client = service_client.create_sampling_client(model_path=pre_sampling_path)
        await _run_eval(pre_sampling_client, step_index=-1)

    # Training loop
    pbar = tqdm(range(max_steps), desc="Training", unit="step")
    for step in pbar:
        step_start = time.time()
        metrics = {"step": step, "epoch": step // steps_per_epoch}

        # On-policy sampler for this step's rollouts. Use the ephemeral variant
        # so we don't persist an 8.7 GB named checkpoint at every step — only
        # step_pretrain and step_final are durable (eval needs to replay them).
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )

        # Get batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            current_epoch += 1
            train_dataloader = create_dataloader(current_epoch)
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Collect rollouts
        logger.info(f"Step {step}: Collecting rollouts for {len(batch)} tasks...")
        rollout_start = time.time()

        rollouts = await collect_batch_rollouts(
            batch=batch,
            tasks_file=tasks_file,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            max_turns=max_turns,
            max_generate_length=max_generate_length,
            max_input_length=max_input_length,
            n_samples_per_prompt=n_samples_per_prompt,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            max_concurrent=max_concurrent,
        )

        metrics["time/rollout"] = time.time() - rollout_start

        # Filter valid rollouts and log invalid ones
        # Note: rollouts are RolloutOutput Pydantic objects - use attribute access
        valid_rollouts = []
        invalid_rollouts = []
        for r in rollouts:
            if r.response_ids and not r.error:
                valid_rollouts.append(r)
            else:
                invalid_rollouts.append(r)

        if invalid_rollouts:
            for r in invalid_rollouts:
                task_key = r.task_key
                error = r.error or "no response_ids"
                stop_reason = r.stop_reason
                logger.warning(f"Step {step}: Invalid rollout for {task_key}: {error} (stop_reason={stop_reason})")
            metrics["rollouts/invalid"] = len(invalid_rollouts)

        if not valid_rollouts:
            logger.warning(f"Step {step}: No valid rollouts, skipping")
            continue

        # Compute GRPO advantages
        rewards = [r.reward for r in valid_rollouts]
        advantages = compute_advantages_grpo(rewards, group_size=n_samples_per_prompt, normalize=True)

        # Compute all rollout metrics (convert to dicts for metrics functions)
        rollout_metrics = compute_rollout_metrics(
            rollouts=[r.model_dump() for r in rollouts],
            valid_rollouts=[r.model_dump() for r in valid_rollouts],
            rewards=rewards,
            advantages=advantages,
            n_samples_per_prompt=n_samples_per_prompt,
        )
        metrics.update(rollout_metrics)

        # Compute timing metrics from valid rollouts
        gen_times = [r.total_gen_time for r in valid_rollouts]
        step_times = [r.total_step_time for r in valid_rollouts]
        tokens = [r.total_tokens for r in valid_rollouts]
        durations = [r.duration for r in valid_rollouts]

        metrics["time/gen_total"] = sum(gen_times)
        metrics["time/gen_mean"] = np.mean(gen_times)
        metrics["time/step_total"] = sum(step_times)
        metrics["time/step_mean"] = np.mean(step_times)
        metrics["time/gen_pct"] = 100 * sum(gen_times) / sum(durations) if sum(durations) > 0 else 0
        metrics["time/step_pct"] = 100 * sum(step_times) / sum(durations) if sum(durations) > 0 else 0
        metrics["throughput/tokens_total"] = sum(tokens)
        metrics["throughput/tokens_per_sec_gen"] = sum(tokens) / sum(gen_times) if sum(gen_times) > 0 else 0
        metrics["throughput/tokens_per_sec_effective"] = sum(tokens) / sum(durations) if sum(durations) > 0 else 0

        # Prepare training data (DAPO filtering + truncation + datum creation)
        training_datums, truncated_count = prepare_training_data(
            rollouts=valid_rollouts,
            advantages=advantages,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )

        metrics["rollouts/truncated_overlong"] = truncated_count
        if truncated_count > 0:
            logger.info(f"Step {step}: Truncated {truncated_count} overlong sequences")

        if not training_datums:
            logger.warning(f"Step {step}: No valid training sequences after filtering, skipping")
            continue

        # Training step
        logger.info(f"Step {step}: Training on {len(training_datums)} sequences...")
        train_start = time.time()

        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn=loss_fn)
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_future.result()
        optim_step_future.result()

        metrics["time/train"] = time.time() - train_start
        metrics["time/total"] = time.time() - step_start

        # Log metrics (commit=True forces immediate sync)
        wandb.log(metrics, step=step, commit=True)
        pbar.set_postfix(
            {
                f"pass@{n_samples_per_prompt}": f"{metrics[f'reward/avg_pass_at_{n_samples_per_prompt}']:.3f}",
                "reward": f"{metrics['reward/avg_raw_reward']:.3f}",
                "time": f"{metrics['time/total']:.1f}s",
            }
        )

        # Periodic state checkpoint for resumability. Persists LoRA adapter +
        # Adam moments + step counter so a transient crash (Tinker 402, infra
        # 5xx, runner timeout) doesn't lose the hours of sampling that
        # preceded the last gradient step. Fires AFTER optim_step has landed,
        # so the saved state reflects the policy resulting from this step's
        # gradient update. (step+1) % N == 0 means: after every N steps of
        # progress. Save again at the very last step regardless of cadence
        # so a completed run always has at least one resume point.
        is_last_step = step == max_steps - 1
        if save_state_every > 0 and ((step + 1) % save_state_every == 0 or is_last_step):
            try:
                state_path = training_client.save_state(
                    name=f"state_{step + 1:06d}"
                ).result().path
                state_checkpoints.append({"step": step + 1, "path": state_path})
                logger.info(
                    f"Saved training state after step {step}: {state_path}"
                )
            except Exception as e:
                logger.warning(f"save_state after step {step} failed: {e}")

        # Periodic + final-step eval, parity with skyrl/train/trainer.py:374.
        # Periodic: every eval_every steps after step 0 (step=0 is
        # indistinguishable from the untrained baseline; use --eval-before-train
        # for that). Final-step: always runs, with a durable "step_final"
        # checkpoint so the auto-train launcher can record post_pass_rate and
        # resume from it.
        is_final_step = step == max_steps - 1
        is_periodic = eval_every > 0 and step > 0 and step % eval_every == 0
        if eval_dataset and (is_periodic or is_final_step):
            if is_final_step:
                final_sampling_path = training_client.save_weights_for_sampler(name="step_final").result().path
                eval_client = service_client.create_sampling_client(model_path=final_sampling_path)
                await _run_eval(eval_client, step_index=max_steps)
            else:
                await _run_eval(sampling_client, step_index=step)

    # Surface pre/post pass rates to disk for the launcher to read. The first
    # entry (pre-train if requested, otherwise step 0) is the baseline; the
    # last entry is the post-train number. Delta is the headline metric.
    if results_out:
        import json as _json
        pre = eval_entries[0]["pass_at_1"] if eval_entries else None
        post = eval_entries[-1]["pass_at_1"] if eval_entries else None
        delta = (post - pre) if (pre is not None and post is not None) else None
        try:
            wandb_url = wandb.run.get_url() if wandb.run else None
        except Exception:
            wandb_url = None
        payload = {
            "model_name": model_name,
            "num_steps": max_steps,
            "n_train": len(train_dataset),
            "n_holdout": len(eval_dataset) if eval_dataset else 0,
            "pre_pass_rate": pre,
            "post_pass_rate": post,
            "delta": delta,
            "entries": eval_entries,
            "wandb_url": wandb_url,
            "wandb_run_name": wandb_name,
            # Tinker checkpoint paths — feed any of these to
            # `service_client.create_sampling_client(model_path=...)` for
            # inference, or to `tinker checkpoint push-hf <path>` to export the
            # LoRA adapter to HuggingFace Hub as a PEFT adapter.
            "checkpoints": {
                "step_pretrain": pre_sampling_path,
                "step_final": final_sampling_path,
                "states": state_checkpoints,
            },
        }
        os.makedirs(os.path.dirname(results_out) or ".", exist_ok=True)
        with open(results_out, "w") as fh:
            _json.dump(payload, fh, indent=2)
        logger.info(f"Wrote eval results to {results_out}")

    wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Task Training with Tinker")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--tasks-file", type=str, required=True, help="Path to tasks JSON file")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--eval-dataset-file", type=str, default=None, help="Path to eval parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-generate-length", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--max-input-length", type=int, default=30720, help="Max context length before ending rollout")
    parser.add_argument("--max-sequence-length", type=int, default=32768, help="Max sequence length for training")
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Max concurrent rollouts (Fleet env instances + Tinker sampling requests). Default 32.",
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="fleet-tinker-grpo")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--track-extra-gradient-metrics",
        type=bool,
        default=False,
        help="Track additional gradient metrics (for parity with SkyRL config)",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument(
        "--stop-sequences",
        type=str,
        default="[]",
        help="JSON list of stop sequences (e.g. '[\"</tool_call>\"]')",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="ppo",
        help="Loss function for Tinker forward_backward (e.g. ppo, grpo)",
    )
    parser.add_argument(
        "--eval-before-train",
        action="store_true",
        help="Run held-out eval once on the base (untrained) LoRA before step 0. Used by auto-train Tinker pipeline to record a pre_pass_rate baseline.",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default=None,
        help="Path to write a JSON summary of pre/post held-out pass rates. Consumed by the auto-train Tinker launcher.",
    )
    parser.add_argument(
        "--save-state-every",
        type=int,
        default=10,
        help="Save a full training state checkpoint every N steps (default 10). 0 disables.",
    )

    args = parser.parse_args()

    import json as _json

    stop_sequences = _json.loads(args.stop_sequences)

    asyncio.run(
        main(
            model_name=args.model_name,
            tasks_file=args.tasks_file,
            dataset_file=args.dataset_file,
            eval_dataset_file=args.eval_dataset_file,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_steps=args.max_steps,
            max_turns=args.max_turns,
            max_generate_length=args.max_generate_length,
            max_input_length=args.max_input_length,
            max_sequence_length=args.max_sequence_length,
            n_samples_per_prompt=args.n_samples_per_prompt,
            eval_every=args.eval_every,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_sequences=stop_sequences,
            loss_fn=args.loss_fn,
            eval_before_train=args.eval_before_train,
            results_out=args.results_out,
            save_state_every=args.save_state_every,
            max_concurrent=args.max_concurrent,
        )
    )
