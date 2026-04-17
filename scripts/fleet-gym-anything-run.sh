#!/usr/bin/env bash
# Gym-Anything GRPO training config (Vision-Language / Desktop CUA).
# Based on fleet-vl-run.sh, adapted for gym-anything local Docker environments.
#
# Model: Qwen/Qwen3.5-9B (9B params, natively multimodal)
# TP=1 (single GPU per engine, 8 engines on 8x H200)
# Environments: gym-anything Docker containers (desktop software)
#
# Required env vars: WANDB_API_KEY
# Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root

export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-gym_anything}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-96000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export RUN_ID="${RUN_ID:-}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

export GYM_ANYTHING_REMOTE_URL="${GYM_ANYTHING_REMOTE_URL:?Set GYM_ANYTHING_REMOTE_URL before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --env-class gym_anything \
  --data-dir-name gym_anything -- \
  environment.skyrl_gym.gym_anything.tasks_file="$HOME/data/fleet/tasks_gym_anything.json" \
  environment.skyrl_gym.gym_anything.use_cache=true \
  environment.skyrl_gym.gym_anything.cache_level=post_start \
  environment.skyrl_gym.gym_anything.remote_url="${GYM_ANYTHING_REMOTE_URL}" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="Qwen/Qwen3.5-9B" \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  trainer.algorithm.loss_reduction="sequence_mean" \
  +generator.chat_template_kwargs='{enable_thinking:true}' \
  +generator.engine_init_kwargs.mm_processor_cache_gb=0 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>"]' \
  'generator.eval_sampling_params.stop=["</tool_call>"]' \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=4 \
  generator.eval_n_samples_per_prompt=3 \
  generator.gpu_memory_utilization=0.80 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-gym-anything-grpo" \
  trainer.run_name="fleet_qwen35_gym_anything_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_qwen35_gym_anything" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
