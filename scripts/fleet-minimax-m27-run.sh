#!/usr/bin/env bash
# MiniMax-M2.7 (230B/10B MoE) GRPO training config.
#
# Model: MiniMaxAI/MiniMax-M2.7 (MoE, 230B total, 10B active, 256 experts, 8 active/token)
# TP=4 (4 GPUs per engine, 8 engines on 32 GPUs across 4 nodes)
# FSDP2 with CPU optimizer offload (required for 230B params)
# 128K context, gradient checkpointing mandatory
#
# Required env vars: FLEET_API_KEY, WANDB_API_KEY
# Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root

# Defaults for vars normally set by SkyPilot YAML envs block
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export DATA_VERSION="${DATA_VERSION:-v6}"
export MODALITY="${MODALITY:-tool_use}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-128000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
export ENV_KEYS="${ENV_KEYS:-}"
export DIFFICULTY="${DIFFICULTY:-}"
export RUN_ID="${RUN_ID:-}"
export MAX_TASKS="${MAX_TASKS:-}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${FLEET_API_KEY:?Set FLEET_API_KEY before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

# Derive batch sizes from GPU count (micro_train_batch_size_per_gpu=1)
TP_SIZE=4
TOTAL_GPUS=$(( ${SKYPILOT_NUM_GPUS_PER_NODE:-8} * ${NUM_NODES:-4} ))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$TOTAL_GPUS}
POLICY_MINI_BATCH_SIZE=${POLICY_MINI_BATCH_SIZE:-$TOTAL_GPUS}
echo "=== Batch sizing: ${TOTAL_GPUS} GPUs, train_batch=${TRAIN_BATCH_SIZE}, mini_batch=${POLICY_MINI_BATCH_SIZE} ==="

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 -- \
  environment.skyrl_gym.fleet_task.ttl_seconds=900 \
  environment.skyrl_gym.fleet_task.partial_reward=true \
  environment.skyrl_gym.fleet_task.enable_hints=false \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="MiniMaxAI/MiniMax-M2.7" \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.flash_attn=true \
  trainer.loss_chunk_size=2048 \
  trainer.use_sample_packing=false \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=1 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>"]' \
  'generator.eval_sampling_params.stop=["</tool_call>"]' \
  trainer.policy.optimizer_config.lr=1.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.zero_variance_filter=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=4 \
  generator.eval_n_samples_per_prompt=3 \
  generator.enforce_eager=true \
  generator.gpu_memory_utilization=0.75 \
  generator.inject_context_status=true \
  generator.context_warning_threshold=0.90 \
  generator.trajectory_timeout_seconds=900 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name="fleet_minimax_m27_${MODALITY}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_minimax_m27_${MODALITY}" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
