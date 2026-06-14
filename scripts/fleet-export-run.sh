#!/usr/bin/env bash
# FSDP -> HuggingFace export run for negotiation checkpoints.
#
# Loads a SHARDED FSDP training checkpoint (per-rank .pt files, what training
# actually saves) back into the distributed policy model and re-emits it as HF
# safetensors under EXPORT_PATH/global_step_<N>/policy, which `vllm serve` (and
# the negotiation eval harnesses) can consume.
#
# This is the offline equivalent of `trainer.save_models()`; use it for runs
# that trained WITHOUT `trainer.hf_save_interval` enabled, or to re-export an
# older checkpoint.
#
# CRITICAL: run this on the SAME node/GPU topology (same world size) the
# checkpoint was trained with — FSDP shard filenames encode the world size and
# the full-state-dict gather is a collective across all ranks. For the 2-node
# x 8-GPU negotiation runs, launch on 2 x 8 GPUs (set the SkyPilot task
# num_nodes: 2, same as training).
#
# Required env vars: WANDB_API_KEY
# Selecting WHAT to export (pick one):
#   RESUME_RUN_NAME   W&B run name whose latest checkpoint to pull from S3 and
#                     export. Needs AWS creds + PROJECT_NAME + MODEL_PATH to
#                     match the training run's S3 prefix.
#   RESUME_PATH       Local global_step_<N> dir to export (sets resume_mode=from_path).
#   EXPORT_ALL_LOCAL_STEPS=1  Export every local global_step_* dir under CKPT_PATH.
#
# Optional env vars:
#   MODEL_PATH        Base HF model the run was trained from. Default: Qwen/Qwen3.5-9B
#   MODEL_TAG         Tag used in the default CKPT_PATH. Default: qwen35
#   PROJECT_NAME      W&B / S3 project prefix. Default: fleet-negotiation-grpo
#   CKPT_PATH         Local dir to download into / read shards from.
#                     Default: $HOME/ckpts/export_${MODEL_TAG}
#   EXPORT_PATH       Where HF safetensors are written. Default: $HOME/exports
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (required for S3 resume)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root

export LOGGER="${LOGGER:-console}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-negotiation}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export MODEL_TAG="${MODEL_TAG:-qwen35}"
export PROJECT_NAME="${PROJECT_NAME:-fleet-negotiation-grpo}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
# Qwen3.5 GDN models can hang in the FlashInfer GDN JIT on GCP/RunPod; force the
# triton GDN prefill backend (harmless elsewhere). Matches the training scripts.
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
export SKIP_IB_INTERSECTION="${SKIP_IB_INTERSECTION:-1}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/export_${MODEL_TAG}}"
EXPORT_PATH="${EXPORT_PATH:-$HOME/exports}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"
export EXPORT_ALL_LOCAL_STEPS="${EXPORT_ALL_LOCAL_STEPS:-}"

# Resolve resume_mode: from_path if RESUME_PATH given, else latest.
RESUME_PATH="${RESUME_PATH:-}"
if [ -n "$RESUME_PATH" ]; then
  RESUME_MODE="${RESUME_MODE:-from_path}"
else
  RESUME_MODE="${RESUME_MODE:-latest}"
fi

echo "=== Fleet FSDP -> HF Export Run ==="
echo "Model:          $MODEL_PATH"
echo "Project:        $PROJECT_NAME"
echo "Resume run:     ${RESUME_RUN_NAME:-(none — using local CKPT_PATH)}"
echo "Resume mode:    $RESUME_MODE"
echo "Resume path:    ${RESUME_PATH:-(n/a)}"
echo "Export all:     ${EXPORT_ALL_LOCAL_STEPS:-0}"
echo "Ckpt path:      $CKPT_PATH"
echo "Export path:    $EXPORT_PATH"

RESUME_ARGS=(trainer.resume_mode="$RESUME_MODE")
if [ "$RESUME_MODE" = "from_path" ]; then
  : "${RESUME_PATH:?Set RESUME_PATH (a global_step_<N> dir) for resume_mode=from_path}"
  RESUME_ARGS+=(trainer.resume_path="$RESUME_PATH")
fi

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint integrations.fleet.entrypoints.main_export \
  --env-class negotiation \
  --data-dir-name negotiation -- \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="export_${MODEL_TAG}" \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.eval_interval=0 \
  trainer.logger="$LOGGER" \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.run_engines_locally=true \
  generator.gpu_memory_utilization=0.4 \
  "${RESUME_ARGS[@]}" \
  "$@"
