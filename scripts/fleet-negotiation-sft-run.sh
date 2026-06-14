#!/usr/bin/env bash
# SFT warm-start stage for the SkyRL negotiation environment.
#
# This script is Stage 1 of the warm-start training pipeline:
#   Stage 1 (this script): SFT — installs a message-level behavioral prior from a
#     human-negotiation corpus. Two dataset options:
#       casino  CaSiNo (food/water/firewood; out-of-domain of the DnD RL env)
#       dnd     Deal or No Deal item-division (in-domain)
#     SFT teaches the model negotiation dialogue format and cooperative discourse
#     patterns without RL pressure, dramatically shortening the subsequent RL
#     exploration phase relative to the cold-start baseline.
#   Stage 2: Warm-Start RL — load $EXPORT_DIR as MODEL_PATH and run the standard
#     GRPO negotiation trainer (fleet-negotiation-35b-run.sh).
#
# Output: a HuggingFace checkpoint directory ($EXPORT_DIR) loadable via
#   AutoModelForCausalLM.from_pretrained($EXPORT_DIR)
# Pass it to the warm-start RL arm via MODEL_PATH=... (printed at end of run).
#
# Infrastructure: single node, 8×H200, FSDP2 full-shard via Ray (local cluster).
# No OpenRouter, no AWS, no WANDB_API_KEY required (wandb off by default).
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL repo root (scripts/ is directly under repo root)

# Source CUDA env overrides (toolkit paths, LD_LIBRARY_PATH, etc.).
# Same convention as fleet-common-run.sh --cuda-env "$HOME/.cuda_env".
if [ -f "$HOME/.cuda_env" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.cuda_env"
fi

# Raise the open-file-descriptor limit (Ray + FSDP2 open many handles).
ulimit -n 65536 2>/dev/null || true

# Force triton GDN prefill backend; Qwen3.5 GDN models can hang silently with
# FlashInfer GDN JIT on some cloud images (see fleet-negotiation-35b-run.sh).
export VLLM_GDN_PREFILL_BACKEND=triton

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Configurable env vars — override before launch or via SkyPilot envs block
# ---------------------------------------------------------------------------
export SFT_DATASET="${SFT_DATASET:-casino}"           # casino | dnd
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
export BOTH_SIDES="${BOTH_SIDES:-true}"               # train on both negotiator roles
export EPOCHS="${EPOCHS:-2}"
export LR="${LR:-1e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export NUM_GPUS="${NUM_GPUS:-8}"
export SEED="${SEED:-1}"
export VAL_FRAC="${VAL_FRAC:-0.0}"
# DATA_DIR must be set after SFT_DATASET so the default can interpolate it.
export DATA_DIR="${DATA_DIR:-$HOME/data/fleet/negotiation_sft_${SFT_DATASET}}"
# EXPORT_DIR is the HF checkpoint the downstream RL run loads as MODEL_PATH.
export EXPORT_DIR="${EXPORT_DIR:-$HOME/exports/sft_${SFT_DATASET}}"

# ---------------------------------------------------------------------------
# Validate SFT_DATASET
# ---------------------------------------------------------------------------
if [ "$SFT_DATASET" != "casino" ] && [ "$SFT_DATASET" != "dnd" ]; then
  echo "ERROR: SFT_DATASET must be 'casino' or 'dnd', got: '${SFT_DATASET}'" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  SFT Warm-Start — Negotiation"
echo "============================================================"
echo "  Dataset    : ${SFT_DATASET}  (both_sides=${BOTH_SIDES})"
echo "  Base model : ${MODEL_PATH}"
echo "  Epochs     : ${EPOCHS}  lr=${LR}  wd=${WEIGHT_DECAY}  warmup=${WARMUP_RATIO}"
echo "  Batch      : micro=${MICRO_BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  max_len=${MAX_LENGTH}"
echo "  GPUs       : ${NUM_GPUS}  seed=${SEED}  val_frac=${VAL_FRAC}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Export dir : ${EXPORT_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Prepare SFT dataset
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Preparing SFT dataset (${SFT_DATASET}) -> ${DATA_DIR}"

python skyrl-gym/skyrl_gym/envs/negotiation/prepare_sft_dataset.py \
  --dataset "${SFT_DATASET}" \
  --output_dir "${DATA_DIR}" \
  --both_sides "${BOTH_SIDES}" \
  --val_frac "${VAL_FRAC}" \
  --seed "${SEED}"

# ---------------------------------------------------------------------------
# Step 2: SFT training
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running SFT trainer (${NUM_GPUS} GPUs, FSDP2)"

# Pass --val_data only when a validation split was actually generated
# (i.e., VAL_FRAC > 0). awk handles the float comparison portably.
VAL_ARGS=()
if awk "BEGIN { exit !( ${VAL_FRAC} + 0 > 0 ) }"; then
  VAL_ARGS=(--val_data "${DATA_DIR}/validation.parquet")
fi

python examples/train/sft/negotiation_sft_trainer.py \
  --data "${DATA_DIR}/train.parquet" \
  --model_path "${MODEL_PATH}" \
  --export_dir "${EXPORT_DIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --max_length "${MAX_LENGTH}" \
  --num_gpus "${NUM_GPUS}" \
  --seed "${SEED}" \
  ${VAL_ARGS[@]+"${VAL_ARGS[@]}"}

# ---------------------------------------------------------------------------
# Done — print checkpoint path and the exact warm-start RL launch command
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SFT checkpoint written to: ${EXPORT_DIR}"
echo "============================================================"
echo ""
echo "To launch the warm-start RL arm from this checkpoint, run:"
echo ""
echo "  MODEL_PATH=${EXPORT_DIR} RUN_ID=ws_warm_${SFT_DATASET} bash scripts/fleet-negotiation-35b-run.sh trainer.ckpt_path=${HOME}/ckpts/ws_warm_${SFT_DATASET} trainer.resume_mode=latest"
echo ""
