"""GRPO / datum helpers for the witness tinker loop — VENDORED from Deniz's
`integrations/fleet/entrypoints/main_fleet_tinker.py` (kept verbatim where it carries
his logprob/shape/DAPO fixes) + `skyrl/train/generators/utils.apply_overlong_filtering`.

Why vendored, not imported: importing `main_fleet_tinker` drags its heavy top-level deps
(`omegaconf`, `skyrl_gym.envs.fleet_task.FleetTaskEnv`, `skyrl.train` → `loguru`/ray) that
the lean agent venv (agent + tinker only) doesn't have and shouldn't need — those are only
used by the OpenEnv collector we don't call. These helpers are pure (numpy/torch/tinker).

KEEP IN SYNC: if Deniz changes `prepare_training_data` / `compute_advantages_grpo` /
`tokenize_chat` / `apply_overlong_filtering`, mirror the change here. Source of truth =
`integrations/fleet/entrypoints/main_fleet_tinker.py` + `skyrl/train/generators/utils.py`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from tinker import types
from tinker.types.tensor_data import TensorData


class RolloutOutput(BaseModel):
    """One (prompt, response) training record. For witness, one per ORAI call."""
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float]
    loss_mask: List[int]
    reward: float
    task_key: str
    env_key: str = "unknown"
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    stop_reason: str = "stop"
    duration: float = 0.0
    total_gen_time: float = 0.0
    total_step_time: float = 0.0
    total_tokens: int = 0
    error: Optional[str] = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_advantages(advantages: List[float]) -> List[float]:
    if not advantages or len(advantages) == 1:
        return advantages
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return [0.0] * len(advantages)
    return [(a - mean) / (std + 1e-8) for a in advantages]


def compute_advantages_grpo(rewards: List[float], group_size: int = None, normalize: bool = True) -> List[float]:
    """GRPO: advantage = reward − group mean, per consecutive group of `group_size`."""
    rewards = np.array(rewards)
    if group_size is None:
        group_size = len(rewards)
    n_groups = len(rewards) // group_size
    advantages: List[float] = []
    for i in range(n_groups):
        g = rewards[i * group_size:(i + 1) * group_size]
        advantages.extend((g - g.mean()).tolist())
    remaining = len(rewards) % group_size
    if remaining > 0:
        g = rewards[-remaining:]
        advantages.extend((g - g.mean()).tolist())
    if normalize:
        advantages = normalize_advantages(advantages)
    return advantages


def apply_overlong_filtering(loss_masks: List[List[int]], stop_reasons: List[str]) -> List[List[int]]:
    """DAPO overlong filtering: zero a trajectory's loss mask if it didn't stop normally."""
    assert len(loss_masks) == len(stop_reasons)
    return [[0] * len(m) if sr != "stop" else m[:] for m, sr in zip(loss_masks, stop_reasons)]


def tokenize_chat(tokenizer, chat_history: List[Dict], add_generation_prompt: bool = True) -> List[int]:
    result = tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt, tokenize=True)
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    if isinstance(result, dict) and "input_ids" in result:
        return list(result["input_ids"])
    return list(result)


def prepare_training_data(rollouts, advantages, tokenizer, max_sequence_length: int):
    """Build tinker Datums (prompt+response, target-shift, masked advantages). Verbatim from
    Deniz's loop incl. the logprob/length-sync guards. Returns (datums, truncated_count)."""
    filtered_loss_masks = apply_overlong_filtering([r.loss_mask for r in rollouts],
                                                   [r.stop_reason for r in rollouts])
    training_datums, truncated_count = [], 0
    for idx, rollout in enumerate(rollouts):
        prompt_ids = rollout.prompt_ids
        response_ids = rollout.response_ids
        logprobs = rollout.logprobs
        loss_mask_data = filtered_loss_masks[idx]
        full_sequence = prompt_ids + response_ids
        prompt_len = len(prompt_ids)
        if len(full_sequence) > max_sequence_length:
            truncated_count += 1
            full_sequence = full_sequence[:max_sequence_length]
            response_len = len(full_sequence) - prompt_len
            response_ids = response_ids[:response_len]
            logprobs = logprobs[:response_len] if logprobs else []
            loss_mask_data = loss_mask_data[:response_len]
        if len(logprobs) != len(response_ids):
            if len(logprobs) > len(response_ids):
                logprobs = logprobs[:len(response_ids)]
            else:
                logprobs = logprobs + [0.0] * (len(response_ids) - len(logprobs))
        target_tokens = full_sequence[1:]
        seq_len = len(target_tokens)
        full_logprobs = ([0.0] * prompt_len + logprobs)[1:]
        full_mask = ([0] * prompt_len + loss_mask_data)[1:]
        full_logprobs = full_logprobs[:seq_len] + [0.0] * max(0, seq_len - len(full_logprobs))
        full_mask = full_mask[:seq_len] + [0] * max(0, seq_len - len(full_mask))
        advantage_value = advantages[idx]
        full_advantages = torch.zeros(len(full_sequence))
        for i in range(prompt_len, len(full_sequence)):
            if i - 1 < len(full_mask) and full_mask[i - 1] > 0:
                full_advantages[i] = advantage_value
        full_advantages = full_advantages[1:]
        training_datums.append(types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(full_logprobs)),
                "advantages": TensorData.from_torch(full_advantages),
            },
        ))
    return training_datums, truncated_count


def compute_pass_at_n(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> float:
    """Fraction of task groups with at least one reward>0 (simplified from reward_metrics)."""
    groups = defaultdict(list)
    for r in rollouts:
        groups[r.get("task_key", "?")].append(r.get("reward", 0.0))
    passes = [1.0 if any(x > 0 for x in g) else 0.0 for g in groups.values()]
    return float(np.mean(passes)) if passes else 0.0


def compute_per_env_metrics(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> Dict[str, float]:
    """Mean reward per game (env_key). Simplified from reward_metrics."""
    by = defaultdict(list)
    for r in rollouts:
        by[r.get("env_key", "?")].append(r.get("reward", 0.0))
    return {f"reward/{k}/mean": float(np.mean(v)) for k, v in by.items()}
