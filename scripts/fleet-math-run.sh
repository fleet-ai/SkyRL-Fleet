#!/usr/bin/env bash
# MATH levels 3-5 GRPO baseline (Qwen2.5-7B-Instruct) on the Fleet multi-node infra.
#
# Infra-integrated counterpart of examples/train/math/run_math_levels3to5_baseline.sh:
# identical hyperparameters, but routed through scripts/fleet-common-run.sh so it gets
# the SAME infra as fleet-35b-vl-run.sh -- Ray multi-node bring-up, per-job NCCL_IB_HCA
# intersection, FSDP2 colocate placement across all nodes, ulimit / alloc-conf /
# NCCL-heartbeat handling, and crash diagnostics. The upstream baseline runs
# single-node via `uv run`; it never starts Ray across nodes, so it cannot span the
# allocation on its own.
#
# Differences from the upstream baseline script:
#   - CURRENT (flat) generator schema, not the legacy nested generator.inference_engine.*
#     form, because fleet-common-run.sh sets the flat generator.num_inference_engines and
#     mixing the two fights the runtime legacy->structured translation.
#   - placement (policy/ref num_gpus_per_node + num_nodes) and num_inference_engines are
#     derived by fleet-common-run.sh from the SkyPilot allocation, so not hard-coded here.
#   - env_class=aime (the Hendrycks-MATH grader, +1/-1) via base entrypoint
#     skyrl.train.entrypoints.main_base -- no Fleet browser/tool backend.
#
# Batch sizing (must divide GPU count = dp_size under FSDP2; see dispatch.py asserts):
#   train_batch_size=128 * n_samples_per_prompt=8 = 1024 trajectories; mini_batch=32.
#   On 32 GPUs (4 nodes): 1024/32=32 and 32/32=1 -- both divide cleanly.
#
# Required env: WANDB_API_KEY (or LOGGER=console).
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root (scripts/ is directly under it)

# ---- Config (env-var overridable; defaults mirror the upstream baseline) ----
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-qwen2.5-7b}"
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

# 32 GPUs / TP=2 = 16 inference engines (colocate_all). fleet-common-run.sh reads this.
export INFERENCE_TP="${INFERENCE_TP:-2}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-16}"

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

export PROJECT_NAME="${PROJECT_NAME:-fleet-math-baseline}"
export RUN_ID="${RUN_ID:-$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')}"
export RUN_NAME="${RUN_NAME:-math_l3to5_grpo_${MODEL_TAG}_${RUN_ID}}"
export LEVELS="${LEVELS:-3 4 5}"

# fleet-common-run.sh references MODALITY unconditionally (TASKS_FILE path) even though
# that path is only USED when env_class=fleet_task. Harmless value so `set -u` is happy;
# DATA_DIR_NAME is forced to math via --data-dir-name.
export MODALITY="${MODALITY:-math}"

if [[ "${LOGGER}" == "wandb" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY (or set LOGGER=console) before running}"
fi

# ---- Prepare MATH dataset on rank 0 (shared NFS, visible to all nodes) ----
# Mirror fleet-common-run.sh's DATA_ROOT/DATA_DIR resolution so the parquet lands exactly
# where the launch looks for it. Idempotent: pre-staging the data out-of-band (so it
# already exists) keeps rank 0 from delaying the Ray head past the workers' join timeout.
DATA_ROOT=""
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
  SCRIPT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CLUSTER_ID="$(echo "$SCRIPT_ABS" | sed -n 's|.*/.sky_clusters/\([^/]*\)/.*|\1|p')"
  if [ -n "$CLUSTER_ID" ]; then
    DATA_ROOT="/workspace/clusters/${CLUSTER_ID}"
  else
    DATA_ROOT="/workspace"
  fi
else
  DATA_ROOT="$HOME"
fi
DATA_DIR="${DATA_ROOT}/data/fleet/math"

if [ "${SKYPILOT_NODE_RANK:-0}" = "0" ]; then
  source .venv/bin/activate
  if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then
    echo "=== Preparing MATH (levels ${LEVELS}) dataset -> ${DATA_DIR} ==="
    mkdir -p "$DATA_DIR"
    python -m examples.train.math.math_dataset --output-dir "$DATA_DIR" --levels ${LEVELS}
  else
    echo "=== MATH dataset already present at ${DATA_DIR} ==="
  fi
  echo "=== Pre-downloading model: $MODEL_PATH ==="
  HF_HOME=/workspace/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1 hf download "$MODEL_PATH"
fi

# `sky exec` (unlike `sky launch`) does not populate SKYPILOT_NUM_GPUS_PER_NODE, which
# fleet-common-run.sh multiplies by SKYPILOT_NUM_NODES to size placement. Left empty it
# yields 0 GPUs -> policy_dp_size=0 -> ZeroDivisionError in validate_batch_sizes. Derive
# it from the visible GPUs so this works under both `sky exec` and `sky launch`.
# Catch empty, non-numeric, AND the literal "0" that sky exec sets (not just unset).
case "${SKYPILOT_NUM_GPUS_PER_NODE:-0}" in
  ''|*[!0-9]*|0)
    export SKYPILOT_NUM_GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
    echo "SKYPILOT_NUM_GPUS_PER_NODE was unset/0; derived $SKYPILOT_NUM_GPUS_PER_NODE from nvidia-smi"
    ;;
esac

# ---- Route through the shared Fleet infra ----
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint skyrl.train.entrypoints.main_base \
  --env-class aime \
  --data-dir-name math -- \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.model.path="${MODEL_PATH}" \
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
  generator.inference_engine_tensor_parallel_size="${INFERENCE_TP}" \
  generator.backend="${INFERENCE_BACKEND}" \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.enforce_eager=true \
  generator.batched=true \
  generator.n_samples_per_prompt="${N_SAMPLES_PER_PROMPT}" \
  generator.eval_n_samples_per_prompt="${EVAL_N_SAMPLES_PER_PROMPT}" \
  generator.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  generator.sampling_params.temperature="${TEMPERATURE}" \
  generator.sampling_params.top_p="${TOP_P}" \
  generator.eval_sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  generator.eval_sampling_params.temperature=0.0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.run_name="${RUN_NAME}" \
  trainer.resume_mode=latest \
  "$@"
