#!/usr/bin/env bash
# GRPO baseline training run on Hendrycks MATH, difficulty levels 3-5.
#
# This is the "standard GRPO, terminal reward only" baseline referenced in
# integrations/fleet/experiment.md (Phase 4, strategy 1). It is the math-domain
# analogue of the agentic fleet-35b-vl-run.sh reference: same documented-header
# + env-var-defaults style, but it trains a single-turn math env on the standard
# SkyRL path (colocated GRPO via `skyrl.train.entrypoints.main_base`) rather than
# the Fleet browser/agentic stack.
#
# Data + grading:
#   - Data: ~/data/math/{train,validation}.parquet  (levels 3-5)
#       Build first:  bash examples/train/math/prepare_math_data.sh
#   - env_class=aime: the `aime` env (skyrl_gym.envs.aime) is the Hendrycks-MATH
#     grader (boxed / "Answer:" extraction + normalization). Reward is +1/-1.
#
# Defaults: Qwen2.5-7B-Instruct on 1 node x 8 GPUs (colocated, fsdp2). Override
# any knob via env vars or trailing Hydra-style `key=value` args:
#   MODEL_PATH=Qwen/Qwen2.5-Math-7B NUM_GPUS_PER_NODE=8 bash examples/train/math/run_math_levels3to5_baseline.sh
#
# Required env: WANDB_API_KEY (or set LOGGER=console to skip W&B).
set -euo pipefail
cd "$(dirname "$0")/../../.."  # cd to SkyRL repo root (this script is examples/train/math/)

# ----------------------------------------------------------------------------
# Config (env-var overridable)
# ----------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-qwen2.5-7b}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/math}"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
export VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation.parquet}"

export NUM_NODES="${NUM_NODES:-1}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"
export INFERENCE_TP="${INFERENCE_TP:-2}"

export NUM_EPOCHS="${NUM_EPOCHS:-15}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-3072}"

export LR="${LR:-1.0e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

export LOGGER="${LOGGER:-wandb}"
export PROJECT_NAME="${PROJECT_NAME:-skyrl-math-baseline}"
export RUN_ID="${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"
export RUN_NAME="${RUN_NAME:-math_l3to5_grpo_${MODEL_TAG}_${RUN_ID}}"
export CKPT_PATH="${CKPT_PATH:-${HOME}/ckpts/${RUN_NAME}}"

if [[ "${LOGGER}" == "wandb" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY (or set LOGGER=console) before running}"
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "ERROR: ${TRAIN_FILE} not found. Run: bash examples/train/math/prepare_math_data.sh" >&2
  exit 1
fi

set -x
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['${TRAIN_FILE}']" \
  data.val_data="['${VAL_FILE}']" \
  environment.env_class=aime \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.model.path="${MODEL_PATH}" \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes="${NUM_NODES}" \
  trainer.placement.policy_num_gpus_per_node="${NUM_GPUS_PER_NODE}" \
  generator.inference_engine.num_engines="${NUM_INFERENCE_ENGINES}" \
  generator.inference_engine.tensor_parallel_size="${INFERENCE_TP}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  generator.inference_engine.enforce_eager=true \
  generator.batched=true \
  generator.n_samples_per_prompt="${N_SAMPLES_PER_PROMPT}" \
  generator.eval_n_samples_per_prompt="${EVAL_N_SAMPLES_PER_PROMPT}" \
  generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  generator.sampling_params.temperature="${TEMPERATURE}" \
  generator.sampling_params.top_p="${TOP_P}" \
  generator.eval_sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  generator.eval_sampling_params.temperature=0.0 \
  trainer.epochs="${NUM_EPOCHS}" \
  trainer.train_batch_size="${TRAIN_BATCH_SIZE}" \
  trainer.policy_mini_batch_size="${MINI_BATCH_SIZE}" \
  trainer.micro_forward_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  trainer.micro_train_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  trainer.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=3 \
  trainer.policy.optimizer_config.lr="${LR}" \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=10 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.run_name="${RUN_NAME}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="${CKPT_PATH}" \
  "$@"
