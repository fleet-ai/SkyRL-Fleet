"""
SFT trainer for ARC-witness B6.

Reads ChatML pairs (`messages` + `metadata`) from
`artifacts/arc-witness-runs/sft_data/v3/sft_pairs_judged.jsonl`, filters on
`metadata.judge_score`, tokenizes via `tokenizer.apply_chat_template`, and
fine-tunes the policy with `WorkerDispatch.forward_backward(loss_fn="cross_entropy")`.

Mirrors the structure of `examples/train/sft/sft_trainer.py` but adapted for
multi-GPU FSDP, ChatML data, judge filtering, wandb logging, and checkpointing.

Local CPU/single-GPU smoke (Qwen2.5-0.5B):
  uv run --extra fsdp python examples/train_integrations/witness/sft_trainer_v3.py \\
      --data /path/to/sft_pairs_judged.jsonl \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --num_gpus 1 --epochs 1 --batch_size 2

Multi-GPU FSDP (Qwen3.5-9B on H200x8) — invoked from SkyPilot yaml:
  python examples/train_integrations/witness/sft_trainer_v3.py \\
      --data $HOME/data/sft_pairs_judged.jsonl \\
      --model Qwen/Qwen3.5-9B \\
      --output_dir $HOME/ckpts/witness_sft \\
      --num_gpus 8 --epochs 3 --batch_size 8 --lr 1e-5 \\
      --wandb_project arc-agi-3 --wandb_run_name witness_sft_v3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import ray
import torch
from loguru import logger
from ray.util.placement_group import placement_group
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import ResolvedPlacementGroup, initialize_ray, validate_cfg


# ---------------------------------------------------------------------------
# Data: load + filter + tokenize
# ---------------------------------------------------------------------------


def load_judged_pairs(path: str, judge_score_min: int) -> list[dict]:
    """Load ChatML SFT pairs, keep only those with judge_score >= min."""
    pairs: list[dict] = []
    n_total = n_kept = n_no_score = 0
    score_dist: dict = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            score = rec.get("metadata", {}).get("judge_score")
            score_dist[score] = score_dist.get(score, 0) + 1
            if score is None:
                n_no_score += 1
                continue
            if score >= judge_score_min:
                pairs.append(rec)
                n_kept += 1
    logger.info(
        f"[data] loaded {n_total} pairs, kept {n_kept} with judge_score >= {judge_score_min} "
        f"(skipped {n_no_score} with no score). Distribution: {dict(sorted(score_dist.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
    )
    return pairs


def chatml_to_sft_example(
    pair: dict, tokenizer, max_length: int = 4096
) -> dict | None:
    """Tokenize a ChatML pair into {input_ids, attention_mask, num_actions}.

    `num_actions` = number of assistant tokens at the tail of the sequence.
    The loss mask in `collate_sft_batch` only fires on those tokens, so the
    model never gets penalized for predicting prompt tokens.
    """
    msgs = pair["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        return None

    # Full sequence: system + user + assistant
    full_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    # Prompt-only: system + user (with generation prompt header so the
    # split aligns exactly with where the assistant turn starts).
    prompt_msgs = msgs[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )

    num_actions = len(full_ids["input_ids"]) - len(prompt_ids["input_ids"])
    if num_actions <= 0:
        return None

    return {
        "input_ids": full_ids["input_ids"],
        "attention_mask": full_ids["attention_mask"],
        "num_actions": num_actions,
    }


def collate_sft_batch(examples: list[dict], tokenizer) -> TrainingInputBatch:
    """Collate tokenized examples into a TrainingInputBatch.

    Left-pads sequences (SkyRL convention). loss_mask=1 only on assistant tokens.
    """
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)

    sequences, attention_masks, loss_masks = [], [], []
    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        sequences.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + ex["attention_mask"])
        action_pad = max_num_actions - ex["num_actions"]
        loss_masks.append([0] * action_pad + [1] * ex["num_actions"])

    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": max_num_actions}
    return batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_cfg(args) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.placement.policy_num_gpus_per_node = args.num_gpus
    # SkyRL's validate_cfg requires num_policy_gpus == num_rollout_gpus
    # when colocate_all=True (the default). SFT doesn't actually run
    # rollout, but the validator doesn't differentiate. Match the rollout
    # engine count to the policy GPU count — same convention as the GRPO
    # yamls (e.g. tasks/witness-grpo-*.yaml: num_inference_engines = TOTAL_POLICY_GPUS).
    # See utils/utils.py:341-350.
    cfg.generator.inference_engine.num_engines = args.num_gpus
    # SkyRL's HFModelWrapper asserts attn_implementation == "flash_attention_2"
    # when use_sample_packing=True (default). We don't enable flash-attn 2
    # because the GRPO yamls explicitly disable it on Qwen3.5-9B (transformers
    # 5.3 + Qwen3.5 VLM combo has flash-attn issues). So we mirror the GRPO
    # yamls and disable sample packing too — for ~63 SFT steps the perf
    # cost from padding is negligible. See model_wrapper.py:124-127.
    cfg.trainer.use_sample_packing = False
    # CRITICAL: chunk the lm_head forward to avoid materializing full (B, S, V)
    # logits tensor. Qwen3.5-9B has vocab ~150k; at B=8, S=4096, bf16 the full
    # logits tensor is ~9.6 GB per forward — likely OOM on 9B+optim_states FSDP.
    # With chunked path (model_wrapper.py:388-432), lm_head is replaced with
    # identity, hidden states stream through `_chunked_lm_head_forward` and
    # log_probs are computed in chunks of `loss_chunk_size`. Mirrors GRPO yaml.
    cfg.trainer.loss_chunk_size = 4096
    # Mirror GRPO yaml convention. Doesn't actually affect cross_entropy path
    # (ppo_utils.py:948 cross_entropy_loss is just (-log_probs * loss_mask).sum()
    # — ignores loss_reduction), but harmless and keeps config in lockstep with
    # the rest of the witness training stack.
    cfg.trainer.algorithm.loss_reduction = "sequence_mean"
    cfg.trainer.micro_train_batch_size_per_gpu = args.micro_batch_size
    cfg.trainer.policy.optimizer_config.lr = args.lr
    cfg.trainer.strategy = args.strategy  # "fsdp2" for multi-GPU, "fsdp" for single
    cfg.trainer.flash_attn = args.flash_attn
    if args.logger:
        cfg.trainer.logger = args.logger
    validate_cfg(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    # Mode
    p.add_argument(
        "--recovery_mode", action="store_true",
        help="Skip training; load existing FSDP shards from --load_ckpt_dir and "
             "save HF format to --export_dir. Used to recover from a previous "
             "training run whose ckpt didn't get merged. Same code path as "
             "GRPO's save_models() (trainer.py:1432) — uses dispatch.save_hf_model.",
    )
    p.add_argument(
        "--load_ckpt_dir", default=None,
        help="(recovery_mode only) FSDP shards directory to load. Should contain "
             "model_world_size_*_rank_*.pt files. Typically <s3_root>/fsdp/global_step_N_final/policy.",
    )
    p.add_argument(
        "--export_dir", default=None,
        help="HF format export directory. In normal training, defaults to "
             "<output_dir>/merged_hf and is also written at end of training. "
             "In recovery_mode, REQUIRED.",
    )
    p.add_argument(
        "--s3_export_dest", default=None,
        help="S3 prefix to sync --export_dir to after save. e.g. "
             "s3://fleet-guanghan/witness_sft_v3/merged/. Skipped if not set.",
    )
    # Data
    p.add_argument("--data", required=False, default=None,
                   help="path to sft_pairs_judged.jsonl (required unless --recovery_mode)")
    p.add_argument(
        "--judge_score_min",
        type=int,
        default=2,
        help="keep pairs with metadata.judge_score >= this (default 2 → 173/247 pairs)",
    )
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--model", required=True)
    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8, help="global batch size")
    p.add_argument("--micro_batch_size", type=int, default=1, help="per-GPU micro batch")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--strategy", default="fsdp2", choices=["fsdp", "fsdp2"])
    p.add_argument("--flash_attn", action="store_true", default=False)
    # Output
    p.add_argument("--output_dir", default=None, help="dir to save final FSDP shards")
    p.add_argument("--save_every", type=int, default=0, help="save ckpt every N steps (0=only at end)")
    p.add_argument(
        "--s3_raw_sync_dest", default=None,
        help="S3 prefix for eager raw FSDP shard sync (defense against bash merge failure). "
             "e.g. s3://fleet-guanghan/witness_sft_v3/fsdp/. Skipped if not set.",
    )
    # Logging
    p.add_argument("--logger", default="console", help="console | wandb | tensorboard")
    p.add_argument("--wandb_project", default="arc-agi-3")
    p.add_argument("--wandb_run_name", default="witness_sft_v3")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def _init_policy_worker(args, cfg):
    """Set up Ray placement group + PolicyWorker actor group + WorkerDispatch.

    Shared between training and recovery_mode entry points.
    Returns (actor_group, dispatch).
    """
    logger.info(f"[ray] initializing policy worker (num_gpus={args.num_gpus})")
    raw_pg = placement_group(
        [{"GPU": args.num_gpus, "CPU": args.num_gpus}], strategy="PACK"
    )
    get_ray_pg_ready_with_timeout(raw_pg, timeout=60)
    pg = ResolvedPlacementGroup(raw_pg)

    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=args.num_gpus,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1.0,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(actor_group.async_init_model(args.model))
    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
    return actor_group, dispatch


def _sync_to_s3(local_dir: str, s3_dest: str, label: str) -> None:
    """aws s3 sync with try/except + verbose logging. Non-fatal on failure
    (so we don't crash after a successful save just because of S3)."""
    import subprocess

    s3_dest = s3_dest.rstrip("/") + "/"
    logger.info(f"[s3] syncing {local_dir} -> {s3_dest} ({label})")
    try:
        subprocess.check_call(["aws", "s3", "sync", local_dir, s3_dest])
        logger.info(f"[s3] {label} synced OK")
    except Exception as e:
        logger.error(
            f"[s3] {label} sync FAILED: {e}. Files still on disk at {local_dir}."
        )


def _run_recovery(args, cfg, tokenizer):
    """Recovery mode: load existing FSDP shards, save as HF format.

    This is the same code path GRPO uses to produce its `merged_hf/` directories
    (skyrl/train/trainer.py:1432 save_models() → dispatch.save_hf_model). We
    use it offline-from-shards by:
      1. Init Ray + PolicyWorker (loads Qwen3.5-9B base from HF Hub — wasted
         download but unavoidable; the FSDP wrapper needs to be alive).
      2. dispatch.load_checkpoint("policy", load_ckpt_dir, ...)  — overrides
         base weights with our SFT'd FSDP shards.
      3. dispatch.save_hf_model("policy", export_dir, tokenizer)  — proper
         FSDP2 gather + HF save_pretrained(safe_serialization=True). Works
         live because we have a real FSDP-wrapped model + process group.
      4. (Optional) S3 sync of export_dir.
    """
    actor_group, dispatch = _init_policy_worker(args, cfg)

    logger.info(f"[recovery] loading FSDP shards from {args.load_ckpt_dir}")
    # Don't try to load optimizer / scheduler — we excluded those from S3 sync
    # (only model weights are present), and we don't need them for HF export.
    dispatch.load_checkpoint(
        "policy",
        args.load_ckpt_dir,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    logger.info("[recovery] FSDP weights loaded into live model")

    os.makedirs(args.export_dir, exist_ok=True)
    logger.info(f"[recovery] saving HF format to {args.export_dir}")
    dispatch.save_hf_model("policy", args.export_dir, tokenizer)
    logger.info(f"[recovery] save_hf_model complete; listing:")
    for f in sorted(os.listdir(args.export_dir)):
        full = os.path.join(args.export_dir, f)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        logger.info(f"  {f:50s}  {size / 1e6:>10.1f} MB")

    if args.s3_export_dest:
        _sync_to_s3(args.export_dir, args.s3_export_dest, "merged_hf")

    ray.shutdown()
    logger.info("[recovery] done")


def main():
    args = parse_args()
    random.seed(args.seed)

    # Validate mode-specific args
    if args.recovery_mode:
        if not args.load_ckpt_dir:
            raise ValueError("--recovery_mode requires --load_ckpt_dir")
        if not args.export_dir:
            raise ValueError("--recovery_mode requires --export_dir")
        logger.info(f"[mode] recovery_mode=True; will load {args.load_ckpt_dir} → save HF to {args.export_dir}")
    else:
        if not args.data:
            raise ValueError("--data is required for training mode (use --recovery_mode to skip)")

    cfg = build_cfg(args)
    initialize_ray(cfg)

    # Wandb is optional — only init if logger=wandb (and not in recovery mode)
    if args.logger == "wandb" and not args.recovery_mode:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model": args.model,
                    "judge_score_min": args.judge_score_min,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "max_length": args.max_length,
                    "num_gpus": args.num_gpus,
                },
            )
        except Exception as e:
            logger.warning(f"[wandb] init failed: {e}; continuing with console only")

    # ---- Tokenizer (always — needed for save_hf_model in both modes) ----
    logger.info(f"[tok] loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # In recovery mode, skip data loading + tokenization + training loop.
    # We still init the policy worker below (FSDP-wrapped model needed for
    # load_checkpoint + save_hf_model).
    if args.recovery_mode:
        _run_recovery(args, cfg, tokenizer)
        return

    # ---- Data ----
    pairs = load_judged_pairs(args.data, args.judge_score_min)
    if not pairs:
        raise RuntimeError(f"No pairs survived judge filter (min={args.judge_score_min})")

    logger.info(f"[tok] tokenizing {len(pairs)} pairs (max_length={args.max_length})")
    tokenized = []
    n_dropped_truncated = 0
    n_at_max_length = 0  # silent partial-truncation: assistant tail may be cut
    for p in pairs:
        ex = chatml_to_sft_example(p, tokenizer, max_length=args.max_length)
        if ex is None:
            n_dropped_truncated += 1
            continue
        if len(ex["input_ids"]) >= args.max_length:
            n_at_max_length += 1
        tokenized.append(ex)
    logger.info(
        f"[tok] kept {len(tokenized)}/{len(pairs)} after tokenization "
        f"(dropped {n_dropped_truncated} as fully truncated)"
    )
    if n_at_max_length:
        logger.warning(
            f"[tok] {n_at_max_length} samples hit max_length={args.max_length} — "
            f"assistant tail may be partially cut. Consider raising --max_length."
        )

    # Sanity: decode boundary of first 3 examples to verify ChatML split
    # produces clean prompt/assistant separation (catches tokenizer quirks
    # that would silently misalign loss_mask).
    logger.info("[tok] tokenization boundary sanity (first 3 examples):")
    for i, ex in enumerate(tokenized[:3]):
        boundary = len(ex["input_ids"]) - ex["num_actions"]
        last_5_prompt = tokenizer.decode(ex["input_ids"][max(0, boundary - 5) : boundary])
        first_5_assistant = tokenizer.decode(ex["input_ids"][boundary : boundary + 5])
        logger.info(
            f"  [{i}] full_len={len(ex['input_ids'])} num_actions={ex['num_actions']} | "
            f"prompt-tail={last_5_prompt!r} -> assistant-head={first_5_assistant!r}"
        )

    # Token-length stats — useful for diagnosing OOM / truncation issues
    full_lens = [len(ex["input_ids"]) for ex in tokenized]
    action_lens = [ex["num_actions"] for ex in tokenized]
    logger.info(
        f"[tok] full_len  min/median/p95/max = "
        f"{min(full_lens)}/{sorted(full_lens)[len(full_lens)//2]}/"
        f"{sorted(full_lens)[int(len(full_lens)*0.95)]}/{max(full_lens)}"
    )
    logger.info(
        f"[tok] action_len min/median/p95/max = "
        f"{min(action_lens)}/{sorted(action_lens)[len(action_lens)//2]}/"
        f"{sorted(action_lens)[int(len(action_lens)*0.95)]}/{max(action_lens)}"
    )

    # ---- Policy worker (shared with recovery mode) ----
    actor_group, dispatch = _init_policy_worker(args, cfg)

    # ---- Training loop ----
    bs = args.batch_size
    n = len(tokenized)
    steps_per_epoch = max(1, n // bs)
    total_steps = steps_per_epoch * args.epochs
    logger.info(
        f"[train] {args.epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} steps total"
    )

    global_step = 0
    for epoch in range(args.epochs):
        # Shuffle each epoch (deterministic given seed)
        rng = random.Random(args.seed + epoch)
        order = list(range(n))
        rng.shuffle(order)
        shuffled = [tokenized[i] for i in order]

        for step_in_epoch in tqdm(
            range(steps_per_epoch), desc=f"epoch {epoch+1}/{args.epochs}"
        ):
            start = step_in_epoch * bs
            batch_examples = shuffled[start : start + bs]
            if len(batch_examples) < bs:
                continue  # skip incomplete tail

            batch = collate_sft_batch(batch_examples, tokenizer)
            metrics = dispatch.forward_backward(
                "policy", batch, loss_fn="cross_entropy"
            )
            grad_norm = dispatch.optim_step("policy")

            global_step += 1

            if global_step % args.log_every == 0:
                loss_val = metrics.get("final_loss", metrics.get("loss"))
                if isinstance(loss_val, torch.Tensor):
                    loss_val = float(loss_val)
                logger.info(
                    f"[train] epoch={epoch+1} step={global_step}/{total_steps} "
                    f"loss={loss_val} grad_norm={grad_norm}"
                )
                if args.logger == "wandb":
                    try:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": loss_val,
                                "train/grad_norm": float(grad_norm)
                                if grad_norm is not None
                                else None,
                                "train/epoch": epoch + 1,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

            if args.save_every and args.output_dir and global_step % args.save_every == 0:
                # Mirror GRPO's path convention exactly (skyrl/train/trainer.py:1261-1268):
                #   global_step_folder/policy/  ← FSDP shards land HERE
                # `dispatch.save_checkpoint("policy", policy_save_dir, ...)`'s first
                # arg "policy" is the actor-group routing key, NOT a path component;
                # the /policy/ subdirectory must be in the path we pass.
                global_step_folder = os.path.join(args.output_dir, f"global_step_{global_step}")
                policy_save_dir = os.path.join(global_step_folder, "policy")
                os.makedirs(global_step_folder, exist_ok=True)
                logger.info(f"[ckpt] saving FSDP shards to {policy_save_dir}")
                dispatch.save_checkpoint("policy", policy_save_dir, tokenizer=tokenizer)

    # ---- Final checkpoint (GRPO path convention: <ckpt_path>/global_step_N/policy/) ----
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        global_step_folder = os.path.join(args.output_dir, f"global_step_{global_step}_final")
        policy_save_dir = os.path.join(global_step_folder, "policy")
        os.makedirs(global_step_folder, exist_ok=True)
        logger.info(f"[ckpt] saving final FSDP shards to {policy_save_dir}")
        dispatch.save_checkpoint("policy", policy_save_dir, tokenizer=tokenizer)

        # Verify the save actually wrote shards. We've been burned by silent
        # path mismatches before — fail loudly here instead of in bash later.
        try:
            shards = sorted(p for p in os.listdir(policy_save_dir) if p.startswith("model_world_size_"))
        except FileNotFoundError:
            shards = []
        if not shards:
            raise RuntimeError(
                f"[ckpt] save_checkpoint completed but no model_world_size_*_rank_*.pt "
                f"files found in {policy_save_dir}. Contents: "
                f"{os.listdir(global_step_folder) if os.path.isdir(global_step_folder) else 'parent missing'}"
            )
        logger.info(f"[ckpt] verified {len(shards)} FSDP shard files at {policy_save_dir}")

        # Defense in depth: sync raw FSDP shards to S3 BEFORE we hand off to bash
        # for merge. If merge fails, at least we still have the raw shards in S3
        # to merge offline. Spot VMs don't give us second chances.
        if args.s3_raw_sync_dest:
            import subprocess

            try:
                logger.info(
                    f"[ckpt] eagerly syncing raw FSDP shards to {args.s3_raw_sync_dest} "
                    f"(safety net before merge)"
                )
                subprocess.check_call(
                    [
                        "aws", "s3", "sync",
                        global_step_folder,
                        args.s3_raw_sync_dest.rstrip("/")
                        + f"/global_step_{global_step}_final/",
                        "--exclude", "*/optim*",
                        "--exclude", "*/scheduler*",
                    ]
                )
                logger.info("[ckpt] raw FSDP shards safe in S3")
            except Exception as e:
                logger.error(
                    f"[ckpt] WARNING: raw FSDP sync to S3 failed: {e}. "
                    f"Shards still on disk at {global_step_folder}; bash merge step "
                    f"will try them. If that fails, ckpt is lost when VM goes down."
                )

        # ── PRIMARY HF EXPORT — mirror GRPO's save_models() (trainer.py:1432) ──
        # Uses dispatch.save_hf_model which does fsdp2_get_full_state_dict +
        # HF model.save_pretrained(safe_serialization=True). This is the
        # production-validated path for producing eval/serving-ready HF
        # checkpoints. Avoids the offline merge_fsdp_checkpoint.py workflow
        # entirely (that script is a fallback, not the primary path).
        export_dir = args.export_dir or os.path.join(args.output_dir, "merged_hf")
        os.makedirs(export_dir, exist_ok=True)
        logger.info(f"[hf-export] saving HF format to {export_dir} via dispatch.save_hf_model")
        try:
            dispatch.save_hf_model("policy", export_dir, tokenizer)
            logger.info(f"[hf-export] save_hf_model complete. Contents:")
            for f in sorted(os.listdir(export_dir))[:15]:
                full = os.path.join(export_dir, f)
                size = os.path.getsize(full) if os.path.isfile(full) else 0
                logger.info(f"  {f:50s}  {size / 1e6:>10.1f} MB")

            if args.s3_export_dest:
                _sync_to_s3(export_dir, args.s3_export_dest, "merged_hf")
        except Exception as e:
            logger.error(
                f"[hf-export] save_hf_model FAILED: {e}. "
                f"Raw FSDP shards already in S3 (if --s3_raw_sync_dest was set), "
                f"recover offline via: python -m examples.train_integrations.witness.sft_trainer_v3 "
                f"--recovery_mode --load_ckpt_dir <fsdp_dir> --export_dir <out>"
            )

    logger.info("[done] SFT training complete")
    if args.logger == "wandb":
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
    ray.shutdown()


if __name__ == "__main__":
    main()
