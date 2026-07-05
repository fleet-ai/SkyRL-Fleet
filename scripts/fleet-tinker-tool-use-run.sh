#!/usr/bin/env bash
# Launch Fleet tool-use training via Tinker hosted service.
# Mirrors fleet-35b-run.sh config but uses the Tinker backend.
#
# Required env vars: TINKER_API_KEY, FLEET_API_KEY, WANDB_API_KEY,
#                    TASKS_FILE, DATASET_FILE
# Optional env vars (with defaults to preserve original interactive behavior):
#   TINKER_API_URL       (SDK uses default if not set)
#   MODEL_NAME           (default: Qwen/Qwen3.5-35B-A3B)
#   MAX_STEPS            (default: 200)
#   LORA_RANK            (default: 16)
#   BATCH_SIZE           (default: 16)
#   LEARNING_RATE        (default: 5.0e-7)
#   N_SAMPLES_PER_PROMPT (default: 8)
#   EVAL_EVERY           (default: 20)
#   EVAL_DATASET_FILE    (default: unset → no held-out eval)
#   RESULTS_OUT          (default: unset → no eval-results JSON)
#   EVAL_BEFORE_TRAIN    (default: unset → no pre-train eval)
#   LOSS_FN              (default: ppo)
#   WANDB_PROJECT        (default: fleet-tinker-grpo)
#   WANDB_NAME           (default: auto from model + timestamp)
#   SAVE_STATE_EVERY     (default: 10; 0 disables periodic save_state)
#   MAX_CONCURRENT       (default: 32; max concurrent Fleet env rollouts)
#   MAX_TURNS            (default: 50)
#   MAX_GENERATE_LENGTH  (default: 3000; max tokens per assistant turn)
#   MAX_INPUT_LENGTH     (default: 128000; rollout context cap)
#   MAX_SEQUENCE_LENGTH  (default: 131072; trainer truncation cap. Must
#                          be >= MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH or
#                          response tokens get dropped from the gradient.)
#
# Pass-through behavior: any extra positional args are appended verbatim to
# the python invocation so a caller can still override anything ad-hoc.
set -euo pipefail

export TINKER_API_KEY="${TINKER_API_KEY:?Set TINKER_API_KEY}"
export TINKER_API_URL="${TINKER_API_URL:-}"
export FLEET_API_KEY="${FLEET_API_KEY:?Set FLEET_API_KEY}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

cd "$(dirname "$0")/.."  # cd to SkyRL-v2 root

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
MAX_STEPS="${MAX_STEPS:-200}"
LORA_RANK="${LORA_RANK:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-7}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_EVERY="${EVAL_EVERY:-20}"
LOSS_FN="${LOSS_FN:-ppo}"
WANDB_PROJECT="${WANDB_PROJECT:-fleet-tinker-grpo}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-3000}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-128000}"
# Training-side sequence cap. Must be >= MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH
# or trainer truncates response tokens (and you pay to generate tokens that
# get dropped from the gradient). Default 131072 = 128K context + headroom.
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-131072}"
# Screenshot compression (browser_use / computer_use). 0 = disabled.
SCREENSHOT_MAX_DIM="${SCREENSHOT_MAX_DIM:-0}"

EXTRA_ARGS=()
if [ -n "${EVAL_DATASET_FILE:-}" ]; then
    EXTRA_ARGS+=(--eval-dataset-file "$EVAL_DATASET_FILE")
fi
if [ -n "${RESULTS_OUT:-}" ]; then
    EXTRA_ARGS+=(--results-out "$RESULTS_OUT")
fi
if [ "${EVAL_BEFORE_TRAIN:-}" = "1" ] || [ "${EVAL_BEFORE_TRAIN:-}" = "true" ]; then
    EXTRA_ARGS+=(--eval-before-train)
fi
if [ -n "${EVAL_N_SAMPLES_PER_PROMPT:-}" ]; then
    EXTRA_ARGS+=(--eval-n-samples-per-prompt "$EVAL_N_SAMPLES_PER_PROMPT")
fi
if [ -n "${WANDB_NAME:-}" ]; then
    EXTRA_ARGS+=(--wandb-name "$WANDB_NAME")
fi
# Resume from a saved training state (Cloud Run executions cap at 24h; runs
# projected past that continue in a fresh execution via LOAD_STATE+START_STEP).
if [ -n "${LOAD_STATE:-}" ]; then
    EXTRA_ARGS+=(--load-state "$LOAD_STATE")
fi
if [ -n "${START_STEP:-}" ] && [ "${START_STEP:-0}" != "0" ]; then
    EXTRA_ARGS+=(--start-step "$START_STEP")
fi

python -m integrations.fleet.entrypoints.main_fleet_tinker \
    --model-name "$MODEL_NAME" \
    --tasks-file "${TASKS_FILE:?Set TASKS_FILE}" \
    --dataset-file "${DATASET_FILE:?Set DATASET_FILE}" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    --max-steps "$MAX_STEPS" \
    --max-turns "$MAX_TURNS" \
    --max-generate-length "$MAX_GENERATE_LENGTH" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --eval-every "$EVAL_EVERY" \
    --temperature 0.9 \
    --top-p 0.95 \
    --stop-sequences '["</tool_call>"]' \
    --loss-fn "$LOSS_FN" \
    --save-state-every "$SAVE_STATE_EVERY" \
    --max-concurrent "$MAX_CONCURRENT" \
    --screenshot-max-dim "$SCREENSHOT_MAX_DIM" \
    --wandb-project "$WANDB_PROJECT" \
    "${EXTRA_ARGS[@]}" \
    "$@"
