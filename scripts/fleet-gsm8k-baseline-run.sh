#!/usr/bin/env bash
# GSM8k baseline GRPO training for Qwen3-1.7B (text-only, single node).
#
# This is a PLAIN BASELINE: standard GRPO on the upstream GSM8k env
# (skyrl_gym.envs.gsm8k) — no custom tool, no Fleet task env, no S3 task
# download. The dataset is prepared locally from HuggingFace via
# examples/train/gsm8k/gsm8k_dataset.py, and training runs through the
# upstream entrypoint skyrl.train.entrypoints.main_base.
#
# Env (one-step, single-turn):
#   skyrl_gym GSM8kEnv — action = the model's full response; reward =
#   utils.compute_score(response, ground_truth): strict regex extracts the
#   number after "#### " and exact-matches it against the gold answer
#   (1.0 correct / 0.0 otherwise). Always done after one step.
#
# Model: Qwen/Qwen3-1.7B  (Qwen3 1.7B post-trained chat model; "-Instruct" has no
#   separate HF repo — Qwen3 ships the instruct variant under the bare id, with
#   Qwen/Qwen3-1.7B-Base being the pretrained-only variant). Qwen3 is a hybrid
#   reasoning model; we DISABLE thinking for this short arithmetic task via
#   enable_thinking=false so the model emits the "#### <answer>" directly
#   instead of burning the response budget on a <think> block.
#
# Topology: 1 node x 8 GPUs, TP=1 -> 8 inference engines (set by NUM_INFERENCE_ENGINES).
#
# Required env vars: WANDB_API_KEY
# Optional env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (only if you want
#   S3 checkpoint upload via main_fleet — NOT used here; main_base writes ckpts locally).
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root (scripts/ is directly under repo root)

# --- Defaults (overridable via SkyPilot YAML envs / CLI) ---
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"          # modest baseline; GSM8k train ~7.5k examples
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-1024}"
# 1 node x 8 GPUs; TP=1 -> 8 engines.
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"

# Fixed run identity (required by the task spec — do not parametrize).
RUN_NAME="neeraj-rewind-gsm8k-baseline"
PROJECT_NAME="fleet-rewind-gsm8k"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

source .venv/bin/activate

# --- Prepare GSM8k data (HuggingFace -> parquet) ---
# Writes train.parquet + validation.parquet (test split) under $DATA_DIR.
DATA_DIR="${HOME}/data/gsm8k"
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then
  echo "=== Preparing GSM8k dataset into ${DATA_DIR} ==="
  python3 examples/train/gsm8k/gsm8k_dataset.py --output_dir "${DATA_DIR}"
else
  echo "=== GSM8k parquet files already present in ${DATA_DIR}; skipping prep ==="
fi

# Delegate Ray cluster bring-up + training launch to the shared runner.
# We use the UPSTREAM entrypoint (main_base) — this is a plain GRPO baseline,
# not a Fleet-task run, so we don't want main_fleet's S3/trace wrapping.
# --env-class gsm8k selects the registered GSM8kEnv.
# --data-dir-name is irrelevant here because we override data.{train,val}_data
# explicitly below (after --), so the absolute $DATA_DIR paths win.
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit \
  --no-pytorch-alloc-conf \
  --entrypoint skyrl.train.entrypoints.main_base \
  --env-class gsm8k \
  --data-dir-name gsm8k -- \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs="$NUM_EPOCHS" \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.max_input_length="$MAX_PROMPT_LENGTH" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=1 \
  generator.backend="$INFERENCE_BACKEND" \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.inference_engine.async_engine=false \
  generator.inference_engine.enforce_eager=true \
  generator.batched=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  "$@"
