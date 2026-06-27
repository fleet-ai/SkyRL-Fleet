import json
import math
import os
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import ray
import torch
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_utils import (
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY,
)
from skyrl.train.config import SkyRLTrainConfig, TrainerConfig
from skyrl.train.dataset import PromptDataset
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
)

BasicType = Union[int, float, str, bool, type(None)]

GLOBAL_STEP_PREFIX = "global_step_"


def finalize_minibatch_rollout_logprob_diff_std(metrics: Dict[str, float]) -> None:
    """Reconstruct the logprob-diff std from its reduced first/second moments, in place.

    Std can't be mean-reduced across micro-batches/DP/mini-batches, so the workers emit the
    moments and we derive ``std = sqrt(E[x^2] - E[x]^2)`` here. Replaces the second-moment key
    with the std; no-op when the moments are absent (e.g. critic training, or no rollout logprobs).
    """
    if (
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY not in metrics
        or MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY not in metrics
    ):
        return
    mean = metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY]
    sq_mean = metrics.pop(MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY)
    # max(0, ...) guards tiny negatives from float round-off.
    metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY] = math.sqrt(max(0.0, sq_mean - mean**2))


class ResumeMode(Enum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        return super()._missing_(value)


def get_node_ids(
    policy_model: PPORayActorGroup, critic_model: Optional[PPORayActorGroup], ref_model: Optional[PPORayActorGroup]
) -> List[str]:
    """Get the node ids of the policy, critic, and ref models.

    Args:
        policy_model: Policy model actor group
        critic_model: Critic model actor group (Optional)
        ref_model: Ref model actor group (Optional)
    """
    policy_node_ids: List[str] = ray.get(policy_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    if critic_model is not None:
        critic_node_ids: List[str] = ray.get(critic_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        critic_node_ids = []
    if ref_model is not None:
        ref_node_ids: List[str] = ray.get(ref_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        ref_node_ids = []

    unique_node_ids = list(set(policy_node_ids + critic_node_ids + ref_node_ids))
    return unique_node_ids


def run_on_each_node(node_ids: List[str], fn: Callable, *args, **kwargs):
    """Simple helper to run a function on each node.

    Args:
        node_ids: List of node ids to run the function on
        fn: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    node_ids = list(set(node_ids))
    task = ray.remote(num_cpus=0.25)(fn)
    refs = []

    for node_id in node_ids:
        node_task = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        )
        refs.append(node_task.remote(*args, **kwargs))

    return ray.get(refs)


def extract_step_from_path(path: str) -> int:
    basename = os.path.basename(path)
    if basename.startswith(GLOBAL_STEP_PREFIX):
        return int(basename.split(GLOBAL_STEP_PREFIX)[1])
    return -1


def list_checkpoint_dirs(checkpoint_base_path: str) -> list[str]:
    """
    List all checkpoint directories in the base path.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored

    Returns:
        list[str]: List of checkpoint directory names
    """
    if not io.exists(checkpoint_base_path):
        return []

    try:
        all_items = io.list_dir(checkpoint_base_path)

        # Filter for directories that match the global_step_* pattern
        checkpoint_dirs = []
        for item in all_items:
            # Get just the basename for pattern matching
            basename = os.path.basename(item)
            if basename.startswith("global_step_") and io.isdir(os.path.join(checkpoint_base_path, basename)):
                checkpoint_dirs.append(basename)

        return sorted(checkpoint_dirs)
    except Exception as e:
        logger.warning(f"Failed to list checkpoint directories from {checkpoint_base_path}: {e}")
        return []


def cleanup_old_checkpoints(checkpoint_base_path: str, max_checkpoints: int) -> None:
    """
    Clean up old checkpoints, keeping only the most recent `max_checkpoints` checkpoints.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored
        max_checkpoints: Maximum number of checkpoints to keep
    """
    if max_checkpoints < 0:
        return

    checkpoint_dirs = list_checkpoint_dirs(checkpoint_base_path)

    if len(checkpoint_dirs) <= max_checkpoints:
        return

    # Sort by step number (extract number from global_step_N)
    def extract_step(dirname):
        try:
            return int(dirname.split("global_step_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoint_dirs.sort(key=extract_step)

    # Remove oldest checkpoints
    dirs_to_remove = checkpoint_dirs[:-max_checkpoints] if max_checkpoints > 0 else checkpoint_dirs

    for dir_name in dirs_to_remove:
        full_path = os.path.join(checkpoint_base_path, dir_name)
        try:
            io.remove(full_path)
            step_num = extract_step(dir_name)
            logger.info(f"Cleaned up old checkpoint: global_step_{step_num} at {full_path}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {full_path}: {e}")


def validate_consistency_for_latest_checkpoint(
    root_ckpt_folder: str, ckpt_iteration: int, checkpoint_path: str, latest_checkpoint_file: str, save_interval: int
):
    """Validate that the checkpoint folder is consistent with the latest checkpoint file.

    Asserts that the folder with the highest global step is the latest checkpoint tracked by `latest_checkpoint_file`.
    Otherwise, the folder state is inconsistent and the user should delete other checkpoints.
    """
    if io.exists(root_ckpt_folder):
        checkpoint_dirs = list_checkpoint_dirs(root_ckpt_folder)
        if checkpoint_dirs:
            global_step_values = [extract_step_from_path(d) for d in checkpoint_dirs]
            max_global_step_in_folder = max(global_step_values)
            # NOTE (sumanthrh): We allow a checkpoint folder to be `save_interval` steps ahead of the latest checkpoint
            # in `latest_checkpoint_file`. This is because the last checkpoint can be an incomplete checkpoint.
            if max_global_step_in_folder - ckpt_iteration > save_interval:
                max_global_step_in_folder_path = os.path.join(
                    root_ckpt_folder, f"{GLOBAL_STEP_PREFIX}{max_global_step_in_folder}"
                )
                raise ValueError(
                    f"Inconsistent checkpoint folder. Latest checkpoint file {latest_checkpoint_file} points to "
                    f"{ckpt_iteration}, but the folder has checkpoints with higher global step - Found global steps "
                    f"{max_global_step_in_folder_path}. This is likely because checkpoint "
                    f"{max_global_step_in_folder_path} was created in a previous run while the latest run is at "
                    f"{checkpoint_path}. Please delete/move checkpoints from older runs and try again."
                )


def sanitize_data_source(data_source: str) -> str:
    """Sanitize data source name for use in file paths."""
    if data_source is None:
        return "unknown"
    return data_source.replace("/", "_")


def mean_numeric_env_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean-aggregate numeric fields across a list of per-episode env metric dicts.

    Mirrors the env-side default aggregator (skyrl_gym.metrics.default_aggregate_metrics)
    but lives trainer-side so the eval path can surface per-episode environment metrics
    (e.g. negotiation you_norm / them_norm / joint_efficiency / pareto). These are
    otherwise dropped during eval: concatenate_generator_outputs re-derives
    rollout_metrics from rewards alone, so the env_metrics never reach wandb unless we
    aggregate them here. Keys present on only some episodes are averaged over the
    episodes that reported them.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for m in metrics:
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, bool):
                v = float(v)
            elif not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def calculate_per_dataset_metrics(
    concat_generator_outputs: GeneratorOutput,
    concat_uids: List[str],
    concat_data_sources: List[str],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Calculate metrics per data source."""
    eval_metrics = {}

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Calculate metrics for each data source
    for data_source, indices in data_source_indices.items():
        # Extract subset for this data source
        subset_generator_output = {
            key: [value[i] for i in indices]
            for key, value in concat_generator_outputs.items()
            if isinstance(value, list)
        }
        subset_uids = [concat_uids[i] for i in indices]

        # Calculate metrics for this subset
        overall_metrics = get_metrics_from_generator_output(subset_generator_output, subset_uids)

        # Add to eval metrics with proper naming
        sanitized_data_source = sanitize_data_source(data_source)
        eval_metrics[f"eval/{sanitized_data_source}/avg_score"] = overall_metrics["avg_score"]
        eval_metrics[f"eval/{sanitized_data_source}/pass_at_{n_samples_per_prompt}"] = overall_metrics["pass_at_n"]
        eval_metrics[f"eval/{sanitized_data_source}/mean_positive_reward"] = overall_metrics["mean_positive_reward"]

        # Per-dataset environment metrics (negotiation: you_norm / them_norm /
        # joint_efficiency / pareto / agreed / ...). These live in env_metrics, which
        # concatenate_generator_outputs does NOT fold into rollout_metrics, so without
        # this they never reach wandb at eval time. Surfacing them per dataset lets us
        # read the pie-expansion signal (joint_efficiency / pareto) and the opponent's
        # score (them_norm) next to avg_score on each held-out eval set.
        subset_env_metrics = subset_generator_output.get("env_metrics") or []
        for key, mean_val in mean_numeric_env_metrics(subset_env_metrics).items():
            eval_metrics[f"eval/{sanitized_data_source}/{key}"] = mean_val

    return eval_metrics


def get_intra_group_completion_time_std_cv(
    generator_output: GeneratorOutput,
) -> Tuple[Optional[float], Optional[float]]:
    """Intra-group spread of per-trajectory completion times for a single group.

    Returns ``(std, cv)`` where ``std`` is the population standard deviation (seconds) and ``cv``
    is the coefficient of variation (``std / mean``). Both are ``None`` when the group recorded no
    completion times or has fewer than two trajectories.
    """
    traj_times = generator_output.get("trajectory_generation_times")
    group_std = None
    group_cv = None
    if traj_times and len(traj_times) > 1:
        # For step wise training, each turn /step contributes one entry.
        # Only take the metrics from the last step
        is_last_step = generator_output.get("is_last_step")
        if is_last_step:
            traj_times = [t for t, last in zip(traj_times, is_last_step) if last]
        traj_times_arr = np.array(traj_times, dtype=np.float64)
        # Population std of per-trajectory completion times within this group (seconds).
        group_std = float(traj_times_arr.std())
        # Coefficient of variation = std / mean. Guard against div-by-zero.
        mean_traj_time = float(traj_times_arr.mean())
        if mean_traj_time > 0:
            group_cv = group_std / mean_traj_time
    return group_std, group_cv


def get_group_completion_metrics(
    group_completion_times: Optional[List[float]],
    intra_group_stds: Optional[List[float]],
    intra_group_cvs: Optional[List[float]],
) -> Dict[str, float]:
    """Per-group completion-time statistics for the groups consumed in a step.

    These surface generation load-balancing behavior (e.g. across vllm-router routing policies):
    tail group latency (p90/max) and how unevenly trajectories within a group finish (intra-group
    coefficient of variation). Each input may be empty/None, in which case the corresponding metrics
    are omitted.
    """
    metrics = {}
    if group_completion_times:
        group_times_arr = np.array(group_completion_times, dtype=np.float64)
        metrics.update(
            {
                "generate/group_completion_time_mean": float(group_times_arr.mean()),
                "generate/group_completion_time_p90": float(np.percentile(group_times_arr, 90)),
                "generate/group_completion_time_max": float(group_times_arr.max()),
            }
        )
    if intra_group_stds:
        metrics.update({"generate/intra_group_completion_time_std_mean": float(np.mean(intra_group_stds))})
    if intra_group_cvs:
        metrics.update({"generate/intra_group_completion_time_cv_mean": float(np.mean(intra_group_cvs))})
    return metrics


def dump_per_dataset_eval_results(
    dump_dir_path: Path,
    tokenizer: AutoTokenizer,
    concat_generator_outputs: GeneratorOutput,
    concat_data_sources: List[str],
    concat_all_envs: List[str],
    concat_env_extras: List[Dict[str, Any]],
    eval_metrics: Dict[str, float],
):
    """Dump evaluation results per dataset and overall aggregated results."""

    # Prepare common data
    input_prompts = [tokenizer.decode(prompt) for prompt in concat_generator_outputs["prompt_token_ids"]]
    output_responses = [tokenizer.decode(response) for response in concat_generator_outputs["response_ids"]]
    multi_modal_data_list = concat_generator_outputs.get("multi_modal_data") or []

    # Save screenshots if any trajectories have images
    images_dir = dump_dir_path / "images"
    has_any_images = any(
        mm and mm.get("images") for mm in multi_modal_data_list if isinstance(mm, dict)
    )
    if has_any_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Dump per-dataset files
    total_images_saved = 0
    for data_source, indices in data_source_indices.items():
        sanitized_data_source = sanitize_data_source(data_source)
        filename = dump_dir_path / f"{sanitized_data_source}.jsonl"

        with open(filename, "w") as f:
            for i in indices:
                # Save screenshots for this eval trajectory
                image_paths = []
                mm_data = multi_modal_data_list[i] if i < len(multi_modal_data_list) else None
                if isinstance(mm_data, dict) and mm_data.get("images"):
                    for j, img in enumerate(mm_data["images"]):
                        img_filename = f"eval_{i:04d}_img_{j:03d}.jpg"
                        img_path = images_dir / img_filename
                        try:
                            if hasattr(img, "save"):
                                img.save(str(img_path), "JPEG", quality=85)
                                image_paths.append(str(img_path))
                                total_images_saved += 1
                            elif isinstance(img, str) and img.startswith(("http://", "https://")):
                                image_paths.append(img)
                                total_images_saved += 1
                            elif isinstance(img, bytes):
                                with open(img_path, "wb") as img_f:
                                    img_f.write(img)
                                image_paths.append(str(img_path))
                                total_images_saved += 1
                        except Exception as e:
                            logger.warning(f"Failed to save eval image {j} for trajectory {i}: {e}")

                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": concat_generator_outputs["rewards"][i],
                    "stop_reason": concat_generator_outputs.get("stop_reasons", [None] * len(input_prompts))[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "data_source": data_source,
                }
                if image_paths:
                    entry["image_paths"] = image_paths
                    entry["num_screenshots"] = len(image_paths)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dumped eval data for {data_source} to {filename}")

    if total_images_saved:
        logger.info(f"Saved {total_images_saved} eval screenshots to {images_dir}")

    # Dump aggregated results file
    aggregated_filename = dump_dir_path / "aggregated_results.jsonl"
    with open(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    logger.info(f"Dumped aggregated eval metrics to {aggregated_filename}")


def dump_training_trajectories(
    dump_dir: str,
    tokenizer: AutoTokenizer,
    generator_output: GeneratorOutput,
    env_extras: List[Dict[str, Any]],
    global_step: int,
) -> str:
    """Dump training trajectories to a JSONL file for analysis.

    Each line contains: step, env_key, data_source, stop_reason, reward, turns, tokens, prompt, text, timestamp.
    """
    traj_dir = Path(dump_dir) / "dumped_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    filename = traj_dir / f"global_step_{global_step}.jsonl"

    env_metrics_list = generator_output.get("env_metrics") or []
    multi_modal_data_list = generator_output.get("multi_modal_data") or []
    rewards_list = generator_output["rewards"]
    stop_reasons = generator_output.get("stop_reasons") or []
    ts = time.time()

    # Save screenshots alongside JSONL if any trajectories have images
    images_dir = traj_dir / f"global_step_{global_step}_images"
    has_any_images = any(
        mm and mm.get("images") for mm in multi_modal_data_list if isinstance(mm, dict)
    )
    if has_any_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        for i in range(len(generator_output["response_ids"])):
            env_m = env_metrics_list[i] if i < len(env_metrics_list) and env_metrics_list[i] else {}
            env_key = env_m.get("env_key", "unknown")
            turns = env_m.get("turns", env_m.get("num_turns", 0))
            extras = env_extras[i] if i < len(env_extras) else {}
            data_source = extras.get("data_source", "unknown") if isinstance(extras, dict) else "unknown"

            reward = rewards_list[i]
            if isinstance(reward, list):
                reward = float(sum(reward))
            else:
                reward = float(reward)

            stop_reason = stop_reasons[i] if i < len(stop_reasons) else "unknown"
            tokens = len(generator_output["response_ids"][i])

            # Save screenshots for this trajectory
            image_paths = []
            mm_data = multi_modal_data_list[i] if i < len(multi_modal_data_list) else None
            if isinstance(mm_data, dict) and mm_data.get("images"):
                for j, img in enumerate(mm_data["images"]):
                    img_filename = f"traj_{i:03d}_img_{j:03d}.jpg"
                    img_path = images_dir / img_filename
                    try:
                        if hasattr(img, "save"):
                            # PIL Image
                            img.save(str(img_path), "JPEG", quality=85)
                            image_paths.append(str(img_path))
                        elif isinstance(img, str) and img.startswith(("http://", "https://")):
                            # URL — store the URL, don't download during training
                            image_paths.append(img)
                        elif isinstance(img, bytes):
                            with open(img_path, "wb") as img_f:
                                img_f.write(img)
                            image_paths.append(str(img_path))
                    except Exception as e:
                        logger.warning(f"Failed to save image {j} for trajectory {i}: {e}")

            entry = {
                "step": global_step,
                "env_key": env_key,
                "data_source": data_source,
                "stop_reason": stop_reason,
                "reward": reward,
                "turns": turns,
                "tokens": tokens,
                "prompt": tokenizer.decode(generator_output["prompt_token_ids"][i]),
                "text": tokenizer.decode(generator_output["response_ids"][i]),
                "timestamp": ts,
            }
            # Some multi-turn envs (e.g. negotiation) end the episode on an opponent
            # turn that is never returned to the policy as an observation, so it is
            # absent from `text` (the decoded response_ids). When the env surfaces that
            # closing turn + resolved outcome in its metrics, attach them here so the
            # trace viewer can show how the episode actually ended. These are kept OUT
            # of `text` on purpose — the policy was not trained on them.
            closing = env_m.get("closing_observation")
            if closing:
                entry["closing_turn"] = closing
            outcome_info = {
                k: env_m[k]
                for k in ("you_take", "them_take", "accepted_by", "item_names", "outcome")
                if env_m.get(k) is not None
            }
            if outcome_info:
                entry["outcome_info"] = outcome_info
            if image_paths:
                entry["image_paths"] = image_paths
                entry["num_screenshots"] = len(image_paths)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_images = sum(
        len(entry.get("images", []))
        for mm in multi_modal_data_list
        if isinstance(mm, dict)
        for entry in [mm]
    )
    logger.info(
        f"Dumped {len(generator_output['response_ids'])} training trajectories to {filename}"
        + (f" ({n_images} screenshots saved)" if has_any_images else "")
    )
    return str(filename)


class DynamicSamplingState(TypedDict, total=False):
    """Schema for dynamic sampling state dictionary.

    Fields:
        sample_batch_count: Counter for the number of sample batches processed
        collected_generator_output: Accumulated generator output (filter strategy only)
        collected_uids: Accumulated UIDs (filter strategy only)
        num_prompts_in_batch: Number of prompts collected so far (filter strategy only)
    """

    sample_batch_count: int
    collected_generator_output: Optional[GeneratorOutput]
    collected_uids: Optional[List[str]]
    num_prompts_in_batch: Optional[int]


def handle_dynamic_sampling(
    generator_output: GeneratorOutput,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: Optional[DynamicSamplingState] = None,
) -> Tuple[GeneratorOutput, List[str], bool, Optional[DynamicSamplingState]]:
    """
    Handle dynamic sampling with different strategies (filter, replace).

    filter (used in DAPO) - filter out groups with std == 0 and group size > 1 and resample until we have enough prompts
    replace (used in POLARIS, WebSailor) - replace bad (std == 0) samples with good (std > 0) samples

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches (for filter strategy)

    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling, updated_state)
    """
    sampling_type = sampling_config.get("type", None)

    if sampling_type is None:
        return generator_output, uids, False, None

    if sampling_type == "replace":
        # For "replace" strategy, the collected state is not used.
        processed_output, processed_uids, keep_sampling = handle_replace_sampling(
            generator_output, uids, sampling_config
        )
        return processed_output, processed_uids, keep_sampling, collected_state
    elif sampling_type == "filter":
        # For filter strategies, accumulate the generator output and UIDs
        # across batches in collected_state if we are sampling repeatedly.
        return handle_filter_sampling(generator_output, uids, sampling_config, collected_state)
    else:
        raise ValueError(f"Invalid dynamic sampling type: {sampling_type}")


def handle_replace_sampling(
    generator_output: GeneratorOutput, uids: List[str], sampling_config: Dict[str, Any]
) -> Tuple[GeneratorOutput, List[str], bool]:
    """
    Handle replace sampling strategy based on POLARIS implementation

    Reference: https://github.com/ChenxinAn-fdu/POLARIS/blob/8c82adb16b8e45c1a34f6d0e23e35deb66dd1ae7/verl/verl/trainer/ppo/ray_trainer.py#L995-L1022.

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling)
    """
    n_samples_per_prompt = sampling_config["n_samples_per_prompt"]
    min_replace_ratio = sampling_config["min_replace_ratio"]

    # Extract rewards and convert to sequence-level if needed
    rewards_list = generator_output["rewards"]
    if rewards_list and isinstance(rewards_list[0], list):
        # Token-level rewards: sum to get sequence rewards
        rewards = np.array([sum(r) for r in rewards_list])
    else:
        rewards = np.array(rewards_list)

    # get mapping of uids to list of indices and metrics
    uid2indices = defaultdict(list)
    uid2metric_vals = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid2indices[uid].append(idx)
        uid2metric_vals[uid].append(rewards[idx])

    # Group by UID and calculate metrics
    uid2metric_std = {}
    for uid, metric_vals in uid2metric_vals.items():
        uid2metric_std[uid] = np.std(metric_vals)

    # Determine good UIDs: those with std > 0 (or group size == 1)
    good_uids = set([uid for uid, std in uid2metric_std.items() if std > 0 or n_samples_per_prompt == 1])
    bad_uids = set([uid for uid, std in uid2metric_std.items() if std == 0 and n_samples_per_prompt > 1])

    logger.info(f"Replace sampling: {len(good_uids)} good UIDs out of {len(uid2metric_vals)} total prompts")

    # Check if we have enough good UIDs (more than min_replace_ratio of the batch)
    if len(good_uids) > len(uid2metric_vals) * min_replace_ratio:
        logger.info("============= Dynamic sampling replace ===========")
        logger.info(f"Number of good prompts: {len(good_uids)}")
        logger.info(f"Number of bad prompts: {len(bad_uids)}")

        # Get good uids to replace the bad uids (length of bad uids)
        replacement_uids = get_bad_sample_replacements(good_uids, bad_uids)  # uids to replace the bad uids
        # get replacement indices
        replacement_indices = []
        for uid in replacement_uids:
            replacement_indices.extend(uid2indices[uid])
        # get bad indices
        bad_indices = []
        for uid in bad_uids:
            bad_indices.extend(uid2indices[uid])

        # Replace bad samples with good ones (modify in place because replacement_idx and bad_idx should not overlap)
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            generator_output["prompt_token_ids"][bad_idx] = generator_output["prompt_token_ids"][replacement_idx].copy()
            generator_output["response_ids"][bad_idx] = generator_output["response_ids"][replacement_idx].copy()
            replacement_reward = generator_output["rewards"][replacement_idx]
            generator_output["rewards"][bad_idx] = (
                replacement_reward.copy() if isinstance(replacement_reward, list) else replacement_reward
            )
            generator_output["loss_masks"][bad_idx] = generator_output["loss_masks"][replacement_idx].copy()
            if generator_output["stop_reasons"]:
                generator_output["stop_reasons"][bad_idx] = generator_output["stop_reasons"][replacement_idx]

            if generator_output["rollout_logprobs"]:
                generator_output["rollout_logprobs"][bad_idx] = generator_output["rollout_logprobs"][replacement_idx]

        # Update UIDs accordingly
        replaced_uids = uids.copy()
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            replaced_uids[bad_idx] = uids[replacement_idx]

        logger.info(f"After replacement - Replaced {len(bad_indices) // n_samples_per_prompt} bad prompts")
        logger.info("==================================================")

        return generator_output, replaced_uids, False
    else:
        logger.warning("===================== Warning (Dynamic sampling replace) ====================")
        logger.warning("In this mini-batch, most training samples receive low variance rewards.")
        logger.warning("If you continue to see this warning, please check your data difficulty distribution.")
        logger.warning("==================================================")

        return generator_output, uids, True


def handle_filter_sampling(
    generator_output: GeneratorOutput,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: DynamicSamplingState,
) -> Tuple[GeneratorOutput, List[str], bool, DynamicSamplingState]:
    """
    Handle filter-based sampling strategy (like DAPO).

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches

    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling, updated_state)
    """
    target_batch_size = sampling_config["train_batch_size"]
    n_samples_per_prompt = sampling_config["n_samples_per_prompt"]

    # Extract rewards from collected output
    rewards_list = generator_output["rewards"]
    if rewards_list and isinstance(rewards_list[0], list):
        # Token-level rewards: sum to get sequence rewards
        rewards = np.array([sum(r) for r in rewards_list])
    else:
        rewards = np.array(rewards_list)

    # Group by UID and calculate standard deviation
    uid2metric_vals = defaultdict(list)
    for uid, reward in zip(uids, rewards):
        uid2metric_vals[uid].append(reward)

    uid2metric_std = {}
    for uid, metric_vals in uid2metric_vals.items():
        uid2metric_std[uid] = np.std(metric_vals)

    # Filter out groups with std == 0 and group size > 1
    kept_uids = [uid for uid, std in uid2metric_std.items() if std > 0 or n_samples_per_prompt == 1]
    kept_uids_set = set(kept_uids)

    # Filter trajectories based on kept UIDs
    kept_traj_idxs = []
    for idx, traj_uid in enumerate(uids):
        if traj_uid in kept_uids_set:
            kept_traj_idxs.append(idx)

    # Apply filtering to generator output
    filtered_output = filter_generator_output(generator_output, kept_traj_idxs)
    filtered_uids = [uids[idx] for idx in kept_traj_idxs]

    if "collected_generator_output" not in collected_state:
        collected_state.update(
            {
                "collected_generator_output": filtered_output,
                "collected_uids": filtered_uids.copy(),
                "num_prompts_in_batch": len(kept_uids),
            }
        )
    else:
        collected_state["collected_generator_output"] = concatenate_generator_outputs(
            [collected_state["collected_generator_output"], filtered_output]
        )
        collected_state["collected_uids"].extend(filtered_uids)
        collected_state["num_prompts_in_batch"] += len(kept_uids)

    # Check if we have enough prompts
    if collected_state["num_prompts_in_batch"] < target_batch_size:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(f"Dynamic sampling: {collected_state['num_prompts_in_batch']} < {target_batch_size} prompts")
        logger.info(f"Resample batch {collected_state['sample_batch_count']}, continue sampling...")
        logger.info("==================================================")
        return generator_output, uids, True, collected_state
    else:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(
            f"Dynamic sampling: collected {collected_state['num_prompts_in_batch']} >= {target_batch_size} prompts"
        )
        logger.info("==================================================")
        # Truncate to exact batch size if needed
        n_samples_per_prompt = sampling_config.get("n_samples_per_prompt", 1)
        max_trajectories = target_batch_size * n_samples_per_prompt
        final_output = collected_state["collected_generator_output"]
        final_uids = collected_state["collected_uids"]

        if len(final_uids) > max_trajectories:
            final_output = filter_generator_output(final_output, list(range(max_trajectories)))
            final_uids = final_uids[:max_trajectories]

        return final_output, final_uids, False, None


def get_bad_sample_replacements(good_uids: List[str], bad_uids: List[str]) -> List[str]:
    num_replacements = len(bad_uids)
    num_candidates = len(good_uids)

    if num_candidates >= num_replacements:
        perm = np.random.permutation(num_candidates)
        chosen_replacement_uids = np.array(list(good_uids))[perm[:num_replacements]]
    else:
        indices = np.random.randint(low=0, high=num_candidates, size=(num_replacements,))
        chosen_replacement_uids = np.array(list(good_uids))[indices]

    return chosen_replacement_uids


def filter_generator_output(output: GeneratorOutput, kept_indices: List[int]) -> GeneratorOutput:
    """Filter GeneratorOutput based on kept indices."""
    filtered = {
        "prompt_token_ids": [output["prompt_token_ids"][i] for i in kept_indices],
        "response_ids": [output["response_ids"][i] for i in kept_indices],
        "rewards": [output["rewards"][i] for i in kept_indices],
        "loss_masks": [output["loss_masks"][i] for i in kept_indices],
        "stop_reasons": None,
        "rollout_metrics": output.get("rollout_metrics"),
        "rollout_logprobs": (
            [output["rollout_logprobs"][i] for i in kept_indices] if output["rollout_logprobs"] else None
        ),
    }

    if output.get("stop_reasons"):
        filtered["stop_reasons"] = [output["stop_reasons"][i] for i in kept_indices]

    filtered["env_metrics"] = (
        [output["env_metrics"][i] for i in kept_indices] if output.get("env_metrics") else None
    )

    return filtered


def zero_variance_filter(
    rewards: List[float],
    uids: List[str],
    loss_masks: Optional[List[List[int]]] = None,
    tol: float = 0.0,
) -> List[int]:
    """
    Given trajectory-level rewards and uids, return the indices of the trajectories to keep.

    A group (trajectories sharing a uid) is dropped only when it has >1 *live* trajectory and their
    reward spread is within ``tol`` (no GRPO signal); groups with <=1 live trajectory are always kept.
    A trajectory is "live" if ``sum(loss_mask) > 0`` (or all live when ``loss_masks`` is None) -- so
    trajectories masked upstream don't make a genuine zero-variance group look varied.

    Args:
        rewards: List[float]
        uids: List[str]
        loss_masks: Optional per-trajectory loss masks, used to determine which trajectories are live.
        tol: Two rewards within this absolute tolerance count as equal. 0.0 reproduces exact
            (``np.std > 0``) behavior; set a small value (e.g. 1e-6) for float (LLM-judge) rewards.

    Returns:
        List[int]
    """
    is_live = [True] * len(rewards) if loss_masks is None else [sum(mask) > 0 for mask in loss_masks]

    # Group live rewards by UID.
    uid2live_rewards = defaultdict(list)
    for uid, reward, live in zip(uids, rewards, is_live):
        if live:
            uid2live_rewards[uid].append(reward)

    def _is_zero_variance(uid: str) -> bool:
        vals = uid2live_rewards.get(uid, [])
        return len(vals) > 1 and (max(vals) - min(vals)) <= tol

    # Keep everything except groups with >1 live trajectory and no reward spread.
    kept_uids_set = {uid for uid in set(uids) if not _is_zero_variance(uid)}

    # Return indices of trajectories with kept UIDs
    return [i for i, uid in enumerate(uids) if uid in kept_uids_set]


def validate_generator_output(num_prompts: int, generator_output: GeneratorOutput, step_wise: bool = False):
    """Validate the generator output.

    Args:
        num_prompts: Number of input prompts used to produce this output.
        generator_output: The generated output batch to validate.
        step_wise: If True, validate step-wise specific fields (is_last_step, trajectory_ids,
            contiguous ordering). In step-wise mode, num_responses may exceed num_prompts
            because each trajectory is expanded into multiple per-turn samples.
    """
    if len(generator_output["response_ids"]) <= 0:
        raise RuntimeError("No outputs generated")

    num_responses = len(generator_output["response_ids"])
    num_prompt_tokens = len(generator_output["prompt_token_ids"])

    if not step_wise:
        assert num_prompts == num_responses, f"Mismatch between prompts ({num_prompts}) and responses ({num_responses})"

    assert (
        num_responses == num_prompt_tokens
    ), f"Mismatch between responses ({num_responses}) and prompt_token_ids ({num_prompt_tokens})"

    # make sure all batch elements have the same length as response_ids (which should be non-zero)
    for key in generator_output:
        if isinstance(generator_output[key], list) and key in [
            "response_ids",
            "loss_masks",
            "rewards",
            "rollout_logprobs",
            "stop_reasons",
            "trajectory_ids",
            "rollout_expert_indices",
            "is_last_step",
            "pixel_values",
            "image_grid_thw",
        ]:
            assert len(generator_output[key]) == len(generator_output["response_ids"]), (
                f"Generator output {key} length must be equal to response_ids length, "
                f"got {len(generator_output[key])} and {len(generator_output['response_ids'])}"
            )

    # make sure that each element of response ids and loss masks are all the same length
    # (and token level rewards if used)
    for i, (response_ids, loss_masks, rewards) in enumerate(
        zip(generator_output["response_ids"], generator_output["loss_masks"], generator_output["rewards"])
    ):
        assert len(response_ids) == len(loss_masks), (
            f"Response ids and loss masks must have the same length, "
            f"for sample {i} got {len(response_ids)} and {len(loss_masks)}"
        )
        if isinstance(rewards, list):
            assert len(rewards) == len(response_ids), (
                f"Token rewards and response ids must have the same length, "
                f"for sample {i} got {len(rewards)} and {len(response_ids)}"
            )

        if generator_output["rollout_logprobs"]:
            assert len(response_ids) == len(generator_output["rollout_logprobs"][i]), (
                f"Response ids and rollout logprobs must have the same length, "
                f"for sample {i} got {len(response_ids)} and {len(generator_output['rollout_logprobs'][i])}"
            )

    # loss masks should be non-zero for at least one element for trainer
    if np.concatenate(generator_output["loss_masks"]).sum() == 0:
        logger.warning("All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!")

    # check that the rewards are either List[float-like] or List[List[float-like]]
    rewards = generator_output["rewards"]
    if isinstance(rewards[0], list):
        assert all(
            isinstance(reward, list) for reward in rewards
        ), "rewards must be `List[float]` or `List[List[float]]`"
    else:
        assert all(
            not isinstance(reward, list) for reward in rewards
        ), "rewards must be `List[float]` or `List[List[float]]`"

    if step_wise:
        _validate_step_wise_fields(generator_output, num_responses)


def _validate_step_wise_fields(generator_output: GeneratorOutput, num_responses: int):
    """Validate step-wise specific fields in the generator output.

    Checks that is_last_step and trajectory_ids are present, correctly sized,
    contiguously ordered, and that is_last_step boundaries align with trajectory_id changes.

    The contiguity check is critical: the trainer's advantage broadcast uses
    ``cumsum(shifted_is_last_step)`` to map each step to its trajectory, which
    silently produces wrong results if steps from the same trajectory are interleaved
    with steps from other trajectories.

    For more, see https://docs.skyrl.ai/docs/tutorials/step-wise-training#generatoroutput-format
    """
    assert (
        generator_output.get("is_last_step") is not None
    ), "step_wise=True but `is_last_step` is missing from generator output"
    assert (
        generator_output.get("trajectory_ids") is not None
    ), "step_wise=True but `trajectory_ids` is missing from generator output"

    is_last_step = generator_output["is_last_step"]
    trajectory_ids = generator_output["trajectory_ids"]

    assert (
        len(is_last_step) == num_responses
    ), f"is_last_step length ({len(is_last_step)}) must equal response_ids length ({num_responses})"
    assert (
        len(trajectory_ids) == num_responses
    ), f"trajectory_ids length ({len(trajectory_ids)}) must equal response_ids length ({num_responses})"

    assert (
        is_last_step[-1] is True
    ), "is_last_step[-1] must be True (the last sample must be the final step of a trajectory)"

    num_trajectories = sum(1 for x in is_last_step if x)
    assert num_trajectories >= 1, "is_last_step must contain at least one True value"

    # Validate contiguous ordering: all steps of the same trajectory must be adjacent.
    seen_trajectory_ids = set()
    prev_tid = None
    for i, tid in enumerate(trajectory_ids):
        tid_key = tid.to_string() if hasattr(tid, "to_string") else str(tid)
        if tid_key != prev_tid:
            assert tid_key not in seen_trajectory_ids, (
                f"Non-contiguous trajectory at index {i}: trajectory '{tid_key}' appeared before "
                f"(at earlier indices), then a different trajectory, then again here. "
                f"Step-wise training requires all steps of the same trajectory to be adjacent."
            )
            if prev_tid is not None:
                seen_trajectory_ids.add(prev_tid)
            prev_tid = tid_key
    if prev_tid is not None:
        seen_trajectory_ids.add(prev_tid)

    # Validate is_last_step aligns with trajectory boundaries (both directions)
    for i in range(num_responses - 1):
        tid_cur = trajectory_ids[i].to_string() if hasattr(trajectory_ids[i], "to_string") else str(trajectory_ids[i])
        tid_next = (
            trajectory_ids[i + 1].to_string()
            if hasattr(trajectory_ids[i + 1], "to_string")
            else str(trajectory_ids[i + 1])
        )
        if tid_cur != tid_next:
            assert is_last_step[i] is True, (
                f"Trajectory boundary at index {i} ('{tid_cur}' → '{tid_next}') "
                f"but is_last_step[{i}] is False. Must be True at trajectory boundaries."
            )
        else:
            assert is_last_step[i] is not True, (
                f"is_last_step[{i}] is True but trajectory continues "
                f"(trajectory '{tid_cur}' at index {i} and {i+1}). "
                f"is_last_step must only be True at the final step of a trajectory."
            )


class HybridEnvSampler(torch.utils.data.Sampler):
    """Ensures minimum representation from each environment per batch.

    Prevents batches dominated by large envs (zillow 1000 tasks) while small
    envs (rops-mail 93 tasks) get zero samples. Each batch gets at least
    min_samples_per_env from every env, remaining slots filled proportionally.

    Ported from fleet-ai/SkyRL-archived.
    """

    def __init__(self, dataset, batch_size, min_samples_per_env=1, generator=None, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_env = min_samples_per_env
        self.generator = generator
        self.drop_last = drop_last

        self.env_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.dataframe[idx]
            group = row.get("data_source") or row.get(dataset.env_class_key, "unknown")
            self.env_to_indices[group].append(idx)

        self.env_classes = list(self.env_to_indices.keys())
        self.num_envs = len(self.env_classes)

        min_required = self.num_envs * min_samples_per_env
        if min_required > batch_size:
            logger.warning(
                f"HybridEnvSampler: {self.num_envs} envs × {min_samples_per_env} = {min_required} "
                f"> batch_size {batch_size}. Reducing min_samples_per_env."
            )
            self.min_samples_per_env = max(1, batch_size // self.num_envs)

        total_samples = len(dataset)
        self.env_weights = {env: len(indices) / total_samples for env, indices in self.env_to_indices.items()}

        logger.info(f"HybridEnvSampler: {self.num_envs} envs, batch_size={batch_size}, min_per_env={self.min_samples_per_env}")
        for env, indices in sorted(self.env_to_indices.items()):
            logger.info(f"  {env}: {len(indices)} samples ({self.env_weights[env]*100:.1f}%)")

    def __iter__(self):
        env_indices_shuffled = {}
        for env, indices in self.env_to_indices.items():
            shuffled = indices.copy()
            perm = torch.randperm(len(shuffled), generator=self.generator).tolist()
            env_indices_shuffled[env] = [shuffled[i] for i in perm]

        env_positions = {env: 0 for env in self.env_classes}

        min_batches_per_env = [len(indices) // self.min_samples_per_env for indices in self.env_to_indices.values()]
        num_batches = min(min_batches_per_env)
        total_samples = sum(len(indices) for indices in self.env_to_indices.values())
        num_batches = min(num_batches, total_samples // self.batch_size)

        for _ in range(num_batches):
            batch_indices = []

            for env in self.env_classes:
                available = len(env_indices_shuffled[env]) - env_positions[env]
                samples_to_take = min(self.min_samples_per_env, available)
                for _ in range(samples_to_take):
                    batch_indices.append(env_indices_shuffled[env][env_positions[env]])
                    env_positions[env] += 1

            remaining = self.batch_size - len(batch_indices)
            if remaining > 0:
                available_by_env = {env: env_indices_shuffled[env][env_positions[env]:] for env in self.env_classes}
                for _ in range(remaining):
                    envs_with_samples = [env for env, avail in available_by_env.items() if avail]
                    if not envs_with_samples:
                        break
                    weights = [self.env_weights[env] for env in envs_with_samples]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    rand_val = torch.rand(1, generator=self.generator).item()
                    cumsum = 0
                    chosen = envs_with_samples[-1]
                    for env, w in zip(envs_with_samples, weights):
                        cumsum += w
                        if rand_val < cumsum:
                            chosen = env
                            break
                    batch_indices.append(available_by_env[chosen].pop(0))
                    env_positions[chosen] += 1

            perm = torch.randperm(len(batch_indices), generator=self.generator).tolist()
            yield [batch_indices[i] for i in perm]

    def __len__(self):
        min_batches_per_env = [len(indices) // self.min_samples_per_env for indices in self.env_to_indices.values()]
        num_batches = min(min_batches_per_env)
        total_samples = sum(len(indices) for indices in self.env_to_indices.values())
        return min(num_batches, total_samples // self.batch_size)


def build_dataloader(
    cfg: SkyRLTrainConfig, dataset: PromptDataset, is_train: bool = True, is_fully_async: bool = False
) -> StatefulDataLoader:
    """
    Build the dataloader for the training or evaluation dataset.

    Args:
        cfg: Config object
        dataset: Dataset object
        is_train: Whether to build the dataloader for training or evaluation
        is_fully_async: If is_train, whether to build the dataloader for fully async training, which
            mainly makes the batch size 1.
    """
    # prepare dataloader
    batch_size = cfg.trainer.train_batch_size if is_train else cfg.trainer.eval_batch_size

    # Seed the dataloader for reproducibility.
    seeded_generator = torch.Generator()
    seeded_generator.manual_seed(cfg.trainer.seed)

    num_workers = cfg.data.dataloader.num_workers
    assert num_workers is not None, "dataloader `num_workers` should be non-null"

    use_hybrid_sampling = (
        is_train
        and not is_fully_async
        and getattr(cfg.trainer, "use_hybrid_env_sampling", False)
        and hasattr(dataset, "dataframe")
        and hasattr(dataset, "env_class_key")
    )

    if use_hybrid_sampling:
        from torch.utils.data import DataLoader

        min_samples_per_env = getattr(cfg.trainer, "min_samples_per_env", 1)
        sampler = HybridEnvSampler(
            dataset=dataset,
            batch_size=batch_size,
            min_samples_per_env=min_samples_per_env,
            generator=seeded_generator,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
        )
        logger.info(f"Using HybridEnvSampler with min_samples_per_env={min_samples_per_env}")
    else:
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size if not is_fully_async else 1,
            shuffle=True if is_train else False,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            # Unlike `shuffle`/`drop_last`, not branched on `is_train`: both dataloaders are
            # reused (train across epochs, eval across evaluations) to avoid worker respawn
            persistent_workers=cfg.data.dataloader.persistent_workers,
            drop_last=True if is_train else False,
            generator=seeded_generator,
            # NOTE (sumanthrh): We use ray and thus use `spawn` start method.
            # forking within ray leads to undefined behaviour and often causes hard to debug
            # memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    if is_train:
        if not is_fully_async:
            logger.info(f"Total steps: {len(dataloader) * cfg.trainer.epochs}")
        else:
            logger.info(f"Total steps: {len(dataloader) // cfg.trainer.train_batch_size * cfg.trainer.epochs}")
    else:
        logger.info(f"Validation set size: {len(dataloader)}")

    return dataloader


def get_rope_scaling_config(trainer_cfg: TrainerConfig) -> dict[str, Any]:
    return trainer_cfg.rope_scaling


def get_rope_theta_config(trainer_cfg: TrainerConfig) -> int | None:
    return trainer_cfg.rope_theta
