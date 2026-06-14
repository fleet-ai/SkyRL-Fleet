"""
Negotiation warm-start SFT trainer (multi-turn, assistant-only supervision).

This script supervised-fine-tunes a policy model on a conversational SFT dataset and,
at the end, writes a HuggingFace-format checkpoint (config.json + *.safetensors +
tokenizer files) that a downstream RL run can load via ``trainer.policy.model.path``.

It mirrors the structure of the minimal demo ``examples/train/sft/sft_trainer.py``
(PPORayActorGroup + PolicyWorker + WorkerDispatch.forward_backward + optim_step) but
adds everything required for a real warm-start run:

  * Multi-turn chat-template tokenization with **assistant-only** loss masking. Assistant
    turns are interspersed throughout the conversation (not just a trailing block), so we
    build a per-token supervision mask via incremental chat-template rendering.
  * FSDP2 full-shard topology across N GPUs (no tensor parallel, no inference engines),
    matching the downstream RL trainer's policy topology.
  * Stable token-mean cross-entropy loss (see "LOSS NORMALIZATION" below).
  * Correct multi-call gradient accumulation against SkyRL's worker semantics.
  * HF checkpoint export via ``dispatch.save_hf_model``.

------------------------------------------------------------------------------------------
INPUT DATA CONTRACT (parquet)
------------------------------------------------------------------------------------------
Each row:
    {
      "messages": [ {"role": "system", "content": str},
                    {"role": "user", "content": str},
                    {"role": "assistant", "content": str}, ... ],   # alternating after system
      "data_source": "sft_casino" | "sft_dnd",
      "extra_info": {...},
    }
Supervision is on ASSISTANT tokens only.

------------------------------------------------------------------------------------------
TENSOR / BATCH FORMAT (verified against SkyRL source)
------------------------------------------------------------------------------------------
``forward_backward("policy", batch, loss_fn=...)`` consumes a ``TrainingInputBatch`` with:
  * ``sequences``       : (B, S) long, LEFT-padded (SkyRL convention).
  * ``attention_mask``  : (B, S), 1 for real tokens, 0 for left-pad.
  * ``loss_mask``       : (B, num_actions), aligned to the LAST ``num_actions`` positions.
  * ``batch.metadata["response_length"] = num_actions`` (a single int for the whole batch).

The model wrapper computes ``action_log_probs = log_probs[:, -num_actions-1:-1]`` where
``log_probs[:, j]`` is the log-prob of predicting token ``j+1`` (the sequence is rolled by
-1 before ``logprobs_from_logits``; see ``model_wrapper.py`` ~L374/L505). The loss is
``(-action_log_probs * loss_mask)`` reduced over the action positions.

MULTI-TURN MASKING TRICK (the key off-by-one):
  We set ``num_actions = S - 1`` so that
      ``log_probs[:, -(S-1)-1 : -1] == log_probs[:, 0 : S-1]`` (the whole sequence except the
      final, dropped position).
  We first build a per-token boolean ``assistant_token_mask`` of shape (B, S): 1 where the
  token at position ``t`` belongs to an assistant turn (its content + the closing
  ``<|im_end|>``), else 0 (left-pad and non-assistant positions are 0).
  Because ``action_log_probs[:, j]`` supervises the prediction of token ``j+1``, the loss
  mask we pass must be ``loss_mask[:, j] = assistant_token_mask[:, j+1]``, i.e.
      ``loss_mask = assistant_token_mask[:, 1:]``   (shape (B, S-1)).
  Position ``j`` therefore supervises the prediction of token ``j+1`` — exactly the
  assistant tokens. (Token 0 is never a prediction target, which is correct since the very
  first token is always the system/BOS header, never an assistant token.)

------------------------------------------------------------------------------------------
LOSS NORMALIZATION
------------------------------------------------------------------------------------------
SkyRL's built-in ``cross_entropy`` loss returns a SUM over masked tokens (Tinker semantics),
so the gradient scale grows with the number of supervised tokens in a micro-batch — long
dialogues dominate and the effective LR becomes batch-composition dependent. We instead
register a custom **token-mean** policy loss ``sft_token_mean_ce`` that divides the masked
sum by ``loss_mask.sum().clamp(min=1)``. This yields a stable per-token NLL whose scale is
independent of sequence length / batch composition.

Why a NEW name (not overriding ``cross_entropy``): each worker process re-runs the
``@register_policy_loss`` decorators at import and would clobber any override of a built-in
name on the shared Ray registry actor. A brand-new name is never touched by those
decorators, so it survives. The trade-off is that a non-``cross_entropy`` name takes the
worker's "RL path" in ``_forward_backward_micro``; we neutralize that by disabling KL and
entropy losses (``use_kl_loss=False``, ``use_entropy_loss=False``), so the effective loss is
exactly ``policy_loss`` = our token-mean CE. (The entropy *tensor* is computed in the model
forward in BOTH paths regardless, so there is no extra compute cost from this choice.)

------------------------------------------------------------------------------------------
GRADIENT ACCUMULATION (verified against worker.py + dispatch.py)
------------------------------------------------------------------------------------------
``PolicyWorkerBase.forward_backward`` micro-batches its input by
``cfg.micro_train_batch_size_per_gpu`` and *accumulates* gradients across micro-batches,
incrementing ``_micro_batches_accumulated``. ``optim_step`` scales the accumulated grads by
``1 / _micro_batches_accumulated`` and only THEN resets the counter. The counter therefore
persists across multiple ``forward_backward`` calls until the next ``optim_step``.

``MeshDispatch`` requires ``len(data) % dp_size == 0`` and splits the batch evenly across the
``dp_size`` DP ranks. To keep padding tight and respect divisibility, per optimizer step we
issue ``grad_accum`` separate ``forward_backward`` calls, each on exactly
``micro_batch_size * dp_size`` samples (so each DP rank receives ``micro_batch_size`` samples
== one micro-batch, given ``micro_train_batch_size_per_gpu = micro_batch_size``). After
``grad_accum`` calls, ``_micro_batches_accumulated == grad_accum`` on every rank, and a single
``optim_step`` averages over them. FSDP2 averages gradients across DP ranks, so the net update
is the mean of ``dp_size * grad_accum`` per-micro-batch token-mean losses — a clean, stable
mean. Each ``forward_backward`` call pads only to the max length within its own
``micro_batch_size * dp_size`` samples, minimizing wasted compute.

------------------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------------------
    uv run --isolated --extra fsdp python examples/train/sft/negotiation_sft_trainer.py \
        --data train.parquet \
        --model_path Qwen/Qwen3.5-35B-A3B \
        --export_dir /path/to/hf_out \
        --epochs 2 --lr 1e-5 --weight_decay 0.0 --warmup_ratio 0.03 \
        --micro_batch_size 1 --grad_accum 8 --max_length 4096 --num_gpus 8

    # Validate the data + masking pipeline on CPU (no Ray, no model):
    python examples/train/sft/negotiation_sft_trainer.py --data train.parquet --dry_run

    # Validate the FULL pipeline cheaply on 1 GPU (tiny model, tiny data, export + reload):
    python examples/train/sft/negotiation_sft_trainer.py --data train.parquet --smoke
"""

import argparse
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Custom token-mean cross-entropy policy loss.
#
# Signature MUST match what PolicyWorkerBase._forward_backward_micro calls:
#   current_loss_fn(action_log_probs, old_action_log_probs, advantages,
#                   config=..., loss_mask=..., rollout_logprobs=...)
# Only `log_probs` (== action_log_probs) and `loss_mask` are used for SFT.
#
# Defined at module level so cloudpickle can serialize it to the Ray registry actor.
# Keep it self-contained (only torch) so it deserializes cleanly inside workers.
# ---------------------------------------------------------------------------
SFT_LOSS_NAME = "sft_token_mean_ce"


def sft_token_mean_cross_entropy(
    log_probs,
    old_log_probs,
    advantages,
    config=None,
    loss_mask=None,
    rollout_logprobs=None,
):
    """Token-mean negative log-likelihood for SFT.

    loss = sum(-log_probs * loss_mask) / max(loss_mask.sum(), 1)

    ``old_log_probs``, ``advantages``, ``rollout_logprobs`` are ignored (RL-only).
    Returns (loss, metrics_dict) to match the policy-loss interface.
    """
    elementwise_loss = -log_probs
    if loss_mask is not None:
        denom = loss_mask.sum().clamp(min=1.0)
        loss = (elementwise_loss * loss_mask).sum() / denom
    else:
        loss = elementwise_loss.mean()
    return loss, {"clip_ratio": 0.0}


def register_sft_loss() -> None:
    """Register the token-mean CE loss on the (Ray-backed) policy-loss registry.

    Must be called AFTER ``initialize_ray`` so the function is synced to the shared
    Ray registry actor and thus visible to all PolicyWorker processes. Idempotent.
    """
    from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry

    if SFT_LOSS_NAME not in PolicyLossRegistry.list_available():
        PolicyLossRegistry.register(SFT_LOSS_NAME, sft_token_mean_cross_entropy)
        logger.info(f"Registered custom policy loss '{SFT_LOSS_NAME}'.")
    else:
        logger.info(f"Policy loss '{SFT_LOSS_NAME}' already registered.")


# ---------------------------------------------------------------------------
# Tokenization + multi-turn assistant masking.
# ---------------------------------------------------------------------------
def _render(tokenizer, messages: List[Dict], add_generation_prompt: bool) -> List[int]:
    """Tokenize ``messages`` with the chat template (thinking disabled).

    Returns a flat list of token ids. ``enable_thinking=False`` is passed when the
    template supports it; unknown template kwargs are otherwise ignored by HF, but we
    guard with try/except for tokenizers whose signature rejects it outright.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )


def _is_prefix(prefix: List[int], full: List[int]) -> bool:
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def build_ids_and_assistant_mask(
    tokenizer, messages: List[Dict]
) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize a full conversation and build a per-token assistant-supervision mask.

    Strategy (robust across templates, supervises assistant CONTENT + closing token only):
      For each assistant message k, the supervised span is
          [ len(render(messages[:k], add_generation_prompt=True)),
            len(render(messages[:k+1], add_generation_prompt=False)) )
      i.e. everything the model must *generate* after the assistant generation prompt
      (``<|im_start|>assistant\\n``) up to and including the turn's ``<|im_end|>``. The role
      header itself is part of the generation prompt and is NOT supervised.

    We assert prefix-consistency: every partial render must be a token-level prefix of the
    full render. If the template is not prefix-stable we return None (sample is skipped) so
    we never produce a misaligned mask.

    Returns (input_ids, assistant_mask) with ``len(assistant_mask) == len(input_ids)``, or
    None if the conversation has no supervisable assistant tokens or the template is not
    prefix-stable.
    """
    full_ids = _render(tokenizer, messages, add_generation_prompt=False)
    mask = [0] * len(full_ids)

    for k, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        gen_prefix = _render(tokenizer, messages[: k], add_generation_prompt=True)
        full_prefix = _render(tokenizer, messages[: k + 1], add_generation_prompt=False)

        # Prefix-stability checks. If any fails, the template is not incrementally
        # decomposable and we cannot trust the mask -> skip this sample.
        if not (_is_prefix(gen_prefix, full_ids) and _is_prefix(full_prefix, full_ids)):
            return None
        if len(gen_prefix) > len(full_prefix):
            return None

        for i in range(len(gen_prefix), len(full_prefix)):
            mask[i] = 1

    if sum(mask) == 0:
        return None
    return full_ids, mask


def fit_to_max_length(
    tokenizer, messages: List[Dict], max_length: int
) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize + mask, dropping whole earliest turns until the sample fits ``max_length``.

    Never truncates mid-turn. Preserves a leading system message (if present) and drops the
    earliest non-system messages two at a time (a user/assistant exchange) to keep the
    user/assistant alternation intact. Returns None if a single sample still cannot fit (or
    has no assistant tokens), in which case the caller skips it.
    """
    msgs = [dict(m) for m in messages]
    sys_offset = 1 if msgs and msgs[0].get("role") == "system" else 0

    while True:
        built = build_ids_and_assistant_mask(tokenizer, msgs)
        if built is not None and len(built[0]) <= max_length:
            return built
        # Need to drop turns. Stop if only system + one turn remain.
        if len(msgs) - sys_offset <= 1:
            return None
        # Drop the earliest non-system exchange (up to 2 messages) to preserve alternation.
        del msgs[sys_offset : sys_offset + 2]


def prepare_dataset(
    data_path: str, tokenizer, max_length: int, limit: Optional[int] = None
) -> List[Dict]:
    """Load the parquet SFT dataset and tokenize into per-sample dicts.

    Each returned sample: ``{"input_ids": List[int], "assistant_mask": List[int],
    "data_source": str}``. Samples that don't fit ``max_length`` or have no assistant
    tokens are skipped.
    """
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=data_path, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    samples: List[Dict] = []
    n_skipped = 0
    for ex in ds:
        messages = ex["messages"]
        # Normalize to plain {role, content} dicts (datasets may return mapping-like rows).
        norm = [{"role": m["role"], "content": m["content"]} for m in messages]
        built = fit_to_max_length(tokenizer, norm, max_length)
        if built is None:
            n_skipped += 1
            continue
        input_ids, assistant_mask = built
        samples.append(
            {
                "input_ids": input_ids,
                "assistant_mask": assistant_mask,
                "data_source": ex.get("data_source", "unknown"),
            }
        )

    logger.info(
        f"Tokenized {len(samples)} samples from {data_path} "
        f"(skipped {n_skipped} that were empty/too long)."
    )
    if len(samples) == 0:
        raise ValueError(f"No usable samples in {data_path} after tokenization/filtering.")
    return samples


def collate_batch(samples: List[Dict], pad_token_id: int):
    """Collate tokenized samples into a TrainingInputBatch with assistant-only loss mask.

    Builds:
      * ``sequences``      : (B, S) long, LEFT-padded.
      * ``attention_mask`` : (B, S), 1 for real tokens, 0 for left-pad.
      * ``loss_mask``      : (B, S-1) float, the per-token assistant mask shifted by 1 so
                             that position ``j`` supervises the prediction of token ``j+1``
                             (see module docstring "MULTI-TURN MASKING TRICK").
      * ``metadata["response_length"] = S - 1`` so the model returns log-probs over the
        whole sequence (minus the dropped final position).
    """
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    S = max(len(s["input_ids"]) for s in samples)

    sequences, attention_masks, token_masks = [], [], []
    for s in samples:
        ids = s["input_ids"]
        amask = s["assistant_mask"]
        pad_len = S - len(ids)
        # Left-pad (SkyRL convention).
        sequences.append([pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
        # Per-token assistant mask over the FULL padded length (left-pad positions = 0).
        token_masks.append([0] * pad_len + amask)

    sequences_t = torch.tensor(sequences, dtype=torch.long)
    attention_t = torch.tensor(attention_masks, dtype=torch.long)
    token_mask_t = torch.tensor(token_masks, dtype=torch.float)  # (B, S)

    # Off-by-one alignment: action_log_probs[:, j] predicts token j+1, so to supervise an
    # assistant token at position t we use loss_mask index j = t-1. Hence drop column 0.
    loss_mask_t = token_mask_t[:, 1:].contiguous()  # (B, S-1)
    num_actions = S - 1

    batch = TrainingInputBatch(
        {
            "sequences": sequences_t,
            "attention_mask": attention_t,
            "loss_mask": loss_mask_t,
        }
    )
    batch.metadata = {"response_length": num_actions}
    return batch


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
def build_config(args, num_training_steps: int):
    """Build a SkyRLTrainConfig with FSDP2 full-shard SFT overrides."""
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    cfg = SkyRLTrainConfig()

    # Topology: FSDP2 full shard across num_gpus, no TP, no inference engines.
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = args.num_gpus
    # No colocation: we only have a policy actor group, and we init it directly (not via
    # dispatch.init_model), so we must NOT let WorkerDispatch try to offload/backload.
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False

    cfg.trainer.policy.model.path = args.model_path
    cfg.trainer.policy.sequence_parallel_size = 1
    cfg.trainer.policy.fsdp_config.fsdp_size = -1  # full shard
    cfg.trainer.policy.fsdp_config.reshard_after_forward = True
    cfg.trainer.policy.fsdp_config.cpu_offload = False

    # Optimizer / LR schedule (OptimizerConfig dataclass).
    cfg.trainer.policy.optimizer_config.lr = args.lr
    cfg.trainer.policy.optimizer_config.weight_decay = args.weight_decay
    cfg.trainer.policy.optimizer_config.scheduler = "cosine"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = max(
        0, int(round(args.warmup_ratio * num_training_steps))
    )
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0

    # Per-GPU micro batch -> one micro-batch per forward_backward call per rank.
    cfg.trainer.micro_train_batch_size_per_gpu = args.micro_batch_size

    # SFT: neutralize the worker's RL path so loss == our token-mean CE.
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.temperature = 1.0  # SFT: no temperature scaling of logits.

    cfg.trainer.seed = args.seed
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine.tensor_parallel_size = 1

    # validate_cfg also repopulates the loss registries; safe to call here. It performs
    # RL-oriented batch-size checks against defaults (train_batch_size=1024, mini=256),
    # which hold for the DP sizes we use.
    try:
        validate_cfg(cfg)
    except Exception as e:  # pragma: no cover - defensive; SFT doesn't use generator
        logger.warning(f"validate_cfg raised ({e}); continuing with SFT overrides.")
    return cfg


# ---------------------------------------------------------------------------
# Dry run (CPU, no Ray / no model): validate data + masking pipeline.
# ---------------------------------------------------------------------------
def run_dry(args, tokenizer) -> None:
    logger.info("[dry_run] Building dataset on CPU (no Ray, no model)...")
    samples = prepare_dataset(args.data, tokenizer, args.max_length, limit=args.dry_run_limit)

    # Group a single forward_backward batch: micro_batch_size * num_gpus samples.
    fb_size = max(1, args.micro_batch_size * args.num_gpus)
    first = samples[: min(fb_size, len(samples))]
    batch = collate_batch(first, tokenizer.pad_token_id)

    seq = batch["sequences"]
    attn = batch["attention_mask"]
    lm = batch["loss_mask"]
    logger.info("[dry_run] First batch shapes / stats:")
    logger.info(f"  sequences       : {tuple(seq.shape)} (dtype={seq.dtype})")
    logger.info(f"  attention_mask  : {tuple(attn.shape)} (sum/row={attn.sum(dim=1).tolist()})")
    logger.info(f"  loss_mask       : {tuple(lm.shape)} (supervised tokens/row={lm.sum(dim=1).int().tolist()})")
    logger.info(f"  response_length : {batch.metadata['response_length']}  (== S-1, S={seq.shape[1]})")

    # Sanity: show a decoded supervised span from sample 0 to eyeball the masking.
    ids0 = first[0]["input_ids"]
    amask0 = first[0]["assistant_mask"]
    sup_ids = [tok for tok, m in zip(ids0, amask0) if m == 1]
    logger.info(
        f"[dry_run] sample[0] data_source={first[0]['data_source']} "
        f"len={len(ids0)} supervised={sum(amask0)}"
    )
    logger.info(
        "[dry_run] sample[0] supervised text (first 300 chars):\n  "
        + repr(tokenizer.decode(sup_ids)[:300])
    )
    logger.info("[dry_run] OK: data + masking pipeline validated.")


# ---------------------------------------------------------------------------
# Optional validation loss (forward-only).
# ---------------------------------------------------------------------------
def evaluate(dispatch, samples: List[Dict], tokenizer, dp_size: int, micro_batch_size: int) -> Optional[float]:
    """Compute mean per-token NLL over the validation set using the forward-only path.

    ``WorkerDispatch.forward`` returns per-token log-probs (the ``output`` field) aligned to
    ``response_length`` positions. We mask with the same shifted assistant mask and average.
    Returns None if forward output cannot be interpreted.
    """
    import numpy as np

    fb_size = max(1, micro_batch_size * dp_size)
    total_nll, total_tokens = 0.0, 0
    n = (len(samples) // fb_size) * fb_size
    if n == 0:
        return None
    for start in range(0, n, fb_size):
        group = samples[start : start + fb_size]
        batch = collate_batch(group, tokenizer.pad_token_id)
        out = dispatch.forward("policy", batch)
        # out["output"] : (B, num_actions) log-probs of the actual next tokens.
        logp = out["output"]
        if not isinstance(logp, torch.Tensor):
            logp = torch.tensor(np.asarray(logp))
        loss_mask = batch["loss_mask"]
        nll = (-(logp.float()) * loss_mask.float()).sum().item()
        total_nll += nll
        total_tokens += int(loss_mask.sum().item())
    if total_tokens == 0:
        return None
    return total_nll / total_tokens


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def run_training(args) -> None:
    import ray
    from ray.util.placement_group import placement_group

    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
    from skyrl.train.utils.utils import initialize_ray, ResolvedPlacementGroup
    from skyrl.train.utils import get_ray_pg_ready_with_timeout

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets.
    train_samples = prepare_dataset(
        args.data, tokenizer, args.max_length, limit=args.max_samples
    )
    val_samples = None
    if args.val_data:
        val_samples = prepare_dataset(args.val_data, tokenizer, args.max_length)

    # DP size (sequence_parallel_size = 1, single node) -> one forward_backward batch is
    # micro_batch_size * dp_size samples.
    dp_size = args.num_gpus
    fb_size = args.micro_batch_size * dp_size
    fb_batches_per_epoch = len(train_samples) // fb_size
    optim_steps_per_epoch = fb_batches_per_epoch // args.grad_accum
    if optim_steps_per_epoch == 0:
        # Too little data for a full grad-accum cycle; do one optim step with whatever fits.
        logger.warning(
            f"Only {fb_batches_per_epoch} forward_backward batches/epoch < grad_accum="
            f"{args.grad_accum}; will take one optim step per epoch over available micro-batches."
        )
        optim_steps_per_epoch = 1 if fb_batches_per_epoch > 0 else 0
    if fb_batches_per_epoch == 0:
        raise ValueError(
            f"Not enough samples ({len(train_samples)}) for even one forward_backward batch "
            f"of size micro_batch_size*num_gpus = {fb_size}."
        )
    total_training_steps = max(1, optim_steps_per_epoch * args.epochs)
    logger.info(
        f"dp_size={dp_size} fb_size={fb_size} fb_batches/epoch={fb_batches_per_epoch} "
        f"optim_steps/epoch={optim_steps_per_epoch} total_optim_steps={total_training_steps}"
    )

    # Ray + config + custom loss registration.
    cfg = build_config(args, num_training_steps=total_training_steps)
    initialize_ray(cfg)
    register_sft_loss()

    # Placement group: single PACK bundle with num_gpus GPUs (matches the demo).
    logger.info("Initializing policy worker group ...")
    raw_pg = placement_group([{"GPU": args.num_gpus, "CPU": args.num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=120)
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
    # Pass num_training_steps so the LR scheduler (warmup + cosine) is built correctly.
    ray.get(
        actor_group.async_init_model(
            cfg.trainer.policy.model.path, num_training_steps=total_training_steps
        )
    )
    # Confirm the real DP size matches our assumption.
    real_dp = actor_group.actor_infos[0].rank.dp_size
    assert real_dp == dp_size, f"DP size mismatch: assumed {dp_size}, got {real_dp}"

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    logger.info("Starting SFT training ...")
    global_optim_step = 0
    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        # Build the per-epoch list of forward_backward groups (each fb_size samples).
        groups = [
            train_samples[i : i + fb_size]
            for i in range(0, fb_batches_per_epoch * fb_size, fb_size)
        ]

        accum_loss, accum_tokens, accum_count = 0.0, 0, 0
        for gi, group in enumerate(groups):
            batch = collate_batch(group, tokenizer.pad_token_id)
            metrics = dispatch.forward_backward("policy", batch, loss_fn=SFT_LOSS_NAME)
            # RL-path metrics: prefer "policy_loss"/"final_loss"; fall back to "loss".
            step_loss = metrics.get(
                "policy_loss", metrics.get("final_loss", metrics.get("loss", float("nan")))
            )
            n_tokens = int(batch["loss_mask"].sum().item())
            accum_loss += step_loss
            accum_tokens += n_tokens
            accum_count += 1

            is_last_group = gi == len(groups) - 1
            if accum_count == args.grad_accum or is_last_group:
                grad_norm = dispatch.optim_step("policy")
                avg_loss = accum_loss / max(1, accum_count)
                lr = metrics.get("policy_lr", metrics.get("lr", float("nan")))
                logger.info(
                    f"epoch={epoch} step={global_optim_step} loss={avg_loss:.4f} "
                    f"grad_norm={grad_norm} lr={lr:.3e} supervised_tokens={accum_tokens}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm if grad_norm is not None else 0.0,
                            "train/lr": lr,
                            "train/supervised_tokens": accum_tokens,
                            "epoch": epoch,
                        },
                        step=global_optim_step,
                    )
                global_optim_step += 1
                accum_loss, accum_tokens, accum_count = 0.0, 0, 0

        # Optional end-of-epoch validation (forward-only).
        if val_samples is not None:
            val_nll = evaluate(dispatch, val_samples, tokenizer, dp_size, args.micro_batch_size)
            if val_nll is not None:
                logger.info(f"epoch={epoch} val_token_mean_nll={val_nll:.4f}")
                if use_wandb:
                    wandb.log({"val/token_mean_nll": val_nll, "epoch": epoch}, step=global_optim_step)
            else:
                logger.info(f"epoch={epoch} validation skipped (could not compute forward NLL).")

    # ---- Export HF checkpoint (config + safetensors + tokenizer) ----
    logger.info(f"Exporting HuggingFace checkpoint to {args.export_dir} ...")
    os.makedirs(args.export_dir, exist_ok=True)
    dispatch.save_hf_model("policy", args.export_dir, tokenizer)
    # save_hf_model already writes the tokenizer when passed; do it again to be safe so the
    # export dir is fully self-contained and loadable as trainer.policy.model.path.
    tokenizer.save_pretrained(args.export_dir)
    logger.info(f"HF checkpoint exported to: {os.path.abspath(args.export_dir)}")

    if use_wandb:
        wandb.finish()
    ray.shutdown()


def verify_export_reloads(export_dir: str) -> None:
    """Assert the exported dir loads as a standard HF model + tokenizer (smoke check)."""
    from transformers import AutoModelForCausalLM

    logger.info(f"[verify] Reloading exported checkpoint from {export_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(export_dir)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[verify] OK: reloaded model with {n_params:,} params; tokenizer vocab={tok.vocab_size}.")
    del model


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Negotiation warm-start SFT trainer (SkyRL).")
    p.add_argument("--data", type=str, required=True, help="Path to train parquet.")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--export_dir", type=str, default=None, help="HF output dir.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--val_data", type=str, default=None, help="Optional validation parquet.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default=None, help="Optional wandb project.")
    p.add_argument("--max_samples", type=int, default=None, help="Cap #train samples (debug).")
    # dry_run: CPU-only data/masking validation (no Ray, no model).
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dry_run_limit", type=int, default=8, help="#rows to load in --dry_run.")
    # smoke: tiny end-to-end GPU run to validate the full pipeline + export reload.
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        # Tiny config so the FULL pipeline (tokenize -> fb -> optim -> export -> reload) is
        # validated cheaply without 35B-scale GPUs.
        logger.info("=== SMOKE MODE: tiny end-to-end pipeline validation ===")
        args.model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        args.num_gpus = 1
        args.epochs = 1
        args.micro_batch_size = 1
        args.grad_accum = 2
        args.max_length = 1024
        args.max_samples = 20
        if args.export_dir is None:
            args.export_dir = tempfile.mkdtemp(prefix="neg_sft_smoke_")
        run_training(args)
        verify_export_reloads(args.export_dir)
        logger.info(f"=== SMOKE PASSED. Export at {args.export_dir} ===")
        return

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        run_dry(args, tokenizer)
        return

    if args.export_dir is None:
        raise ValueError("--export_dir is required for a real training run.")
    run_training(args)


if __name__ == "__main__":
    main()
