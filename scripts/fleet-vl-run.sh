#!/usr/bin/env bash
# VL/CUA (Vision-Language / Computer Use Agent) GRPO training config.
# Called by the SkyPilot YAML and by fleet-research run.sh.
#
# Based on working config from SkyRL PR #288 (feat/vl-support-clean),
# adapted to SkyRL-v2's fleet-common-run.sh pattern.
#
# Model: Qwen/Qwen3.5-9B (9B params, natively multimodal, GatedDeltaNet)
# TP=1 (single GPU per engine, 8 engines on 8x H200)
# Modality: browser_use (screenshots + coordinate normalization)
#
# Required env vars: FLEET_API_KEY, WANDB_API_KEY
# Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root (scripts/ is directly under repo root)

# Defaults for vars normally set by SkyPilot YAML envs block
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export DATA_VERSION="${DATA_VERSION:-v6}"
export MODALITY="${MODALITY:-browser_use}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export MAX_TURNS="${MAX_TURNS:-80}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-80000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
export ENV_KEYS="${ENV_KEYS:-}"
export DIFFICULTY="${DIFFICULTY:-}"
export RUN_ID="${RUN_ID:-}"
export PROJECT_NAME="${PROJECT_NAME:-fleet-taste-baselines}"
export RUN_NAME="${RUN_NAME:-fleet_qwen35_${MODALITY}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}}"
export MAX_TASKS="${MAX_TASKS:-}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"
export TASTE_REWARD="${TASTE_REWARD:-false}"
export TASTE_TEXT_MODEL="${TASTE_TEXT_MODEL:-claude-sonnet-4-5-20250929}"
export TASTE_VISUAL_MODEL="${TASTE_VISUAL_MODEL:-claude-sonnet-4-5-20250929}"
export TASTE_SKIP_VISUAL="${TASTE_SKIP_VISUAL:-false}"
export TASTE_N_SCREENSHOTS="${TASTE_N_SCREENSHOTS:-8}"
export TASTE_TIMEOUT="${TASTE_TIMEOUT:-60.0}"
export TASTE_JUDGE_REQUIRED="${TASTE_JUDGE_REQUIRED:-true}"
export SKYRL_FAIL_ON_ENV_INIT_ERROR="${SKYRL_FAIL_ON_ENV_INIT_ERROR:-true}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${FLEET_API_KEY:?Set FLEET_API_KEY before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf -- \
  environment.skyrl_gym.fleet_task.ttl_seconds=1800 \
  environment.skyrl_gym.fleet_task.partial_reward=false \
  environment.skyrl_gym.fleet_task.taste_reward=$TASTE_REWARD \
  environment.skyrl_gym.fleet_task.taste_text_model="$TASTE_TEXT_MODEL" \
  environment.skyrl_gym.fleet_task.taste_visual_model="$TASTE_VISUAL_MODEL" \
  environment.skyrl_gym.fleet_task.taste_skip_visual=$TASTE_SKIP_VISUAL \
  environment.skyrl_gym.fleet_task.taste_n_screenshots=$TASTE_N_SCREENSHOTS \
  environment.skyrl_gym.fleet_task.taste_timeout=$TASTE_TIMEOUT \
  environment.skyrl_gym.fleet_task.taste_judge_required=$TASTE_JUDGE_REQUIRED \
  environment.skyrl_gym.fleet_task.enable_hints=false \
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
  trainer.eval_batch_size=$EVAL_BATCH_SIZE \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=2 \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>"]' \
  'generator.eval_sampling_params.stop=["</tool_call>"]' \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.zero_variance_filter=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=$GRPO_GROUP_SIZE \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES \
  generator.gpu_memory_utilization=0.80 \
  generator.trajectory_timeout_seconds=900 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_qwen35_${MODALITY}" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
