#!/usr/bin/env bash
# Run held-out eval against a Tinker checkpoint, no training.
#
# Same code path as `fleet-tinker-tool-use-run.sh`'s periodic eval (the
# trainer calls _run_eval every N steps); this wrapper just forces the
# entrypoint into --eval-only mode so no training_client is created and no
# forward_backward / optim_step ever runs.
#
# Required env vars:
#   FROM_CHECKPOINT      Tinker URI: tinker://<run-id>:train:<replica>/sampler_weights/<step>
#   MODEL_NAME           Base HF model name (e.g. moonshotai/Kimi-K2.6:peft:131072)
#   TASKS_FILE           Tasks JSON (same shape the trainer consumes)
#   EVAL_DATASET_FILE    Held-out parquet (use auto_train/splitter.split_90_10)
#   TINKER_API_KEY
#   FLEET_API_KEY        Required by FleetTaskEnv to talk to envs
#
# Optional env vars:
#   TINKER_API_URL              (SDK default if unset)
#   WANDB_API_KEY               (eval logs to wandb if set; otherwise WANDB_MODE
#                                handles fall-through; pass WANDB_MODE=disabled
#                                to skip wandb entirely)
#   WANDB_PROJECT               (default: fleet-tinker-eval)
#   WANDB_NAME                  (default: auto from model + timestamp)
#   RESULTS_OUT                 (default: /tmp/fleet-tinker-eval-results.json)
#   EVAL_BATCH_SIZE             (default: 16)
#   EVAL_N_SAMPLES_PER_PROMPT   (default: 4 → pass@4)
#   MAX_TURNS                   (default: 50)
#   MAX_GENERATE_LENGTH         (default: 3000)
#   MAX_INPUT_LENGTH            (default: 128000)
#   MAX_SEQUENCE_LENGTH         (default: 131072)
#   MAX_CONCURRENT              (default: 8 — held-out sets are small; lower
#                                concurrency keeps Fleet env contention down
#                                during ad-hoc eval runs)
#   TEMPERATURE                 (default: 0.6  — Kimi K2.6 recommended)
#   TOP_P                       (default: 0.95 — Kimi K2.6 recommended)
#   STOP_SEQUENCES              (default: [])
#
# Pass-through: extra positional args go to the python entrypoint verbatim.
set -euo pipefail

export TINKER_API_KEY="${TINKER_API_KEY:?Set TINKER_API_KEY}"
export TINKER_API_URL="${TINKER_API_URL:-}"
export FLEET_API_KEY="${FLEET_API_KEY:?Set FLEET_API_KEY}"
# WANDB_API_KEY is optional for eval-only; if absent and WANDB_MODE not set,
# wandb will prompt and crash. Default to disabled when key is missing.
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ]; then
    export WANDB_MODE=disabled
fi

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME}"
FROM_CHECKPOINT="${FROM_CHECKPOINT:?Set FROM_CHECKPOINT (tinker://... URI)}"
TASKS_FILE="${TASKS_FILE:?Set TASKS_FILE}"
EVAL_DATASET_FILE="${EVAL_DATASET_FILE:?Set EVAL_DATASET_FILE (held-out parquet)}"
RESULTS_OUT="${RESULTS_OUT:-/tmp/fleet-tinker-eval-results.json}"
WANDB_PROJECT="${WANDB_PROJECT:-fleet-tinker-eval}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-3000}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-128000}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-131072}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
STOP_SEQUENCES="${STOP_SEQUENCES:-[]}"

EXTRA_ARGS=()
if [ -n "${WANDB_NAME:-}" ]; then
    EXTRA_ARGS+=(--wandb-name "$WANDB_NAME")
fi

# --max-steps is unused in eval-only but the parser still accepts it; set to 0
# so a stray downstream check (e.g. wandb config logging) sees a sane value.
python -m integrations.fleet.entrypoints.main_fleet_tinker \
    --eval-only \
    --from-checkpoint "$FROM_CHECKPOINT" \
    --model-name "$MODEL_NAME" \
    --tasks-file "$TASKS_FILE" \
    --eval-dataset-file "$EVAL_DATASET_FILE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-n-samples-per-prompt "$EVAL_N_SAMPLES_PER_PROMPT" \
    --max-steps 0 \
    --max-turns "$MAX_TURNS" \
    --max-generate-length "$MAX_GENERATE_LENGTH" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
    --max-concurrent "$MAX_CONCURRENT" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --stop-sequences "$STOP_SEQUENCES" \
    --results-out "$RESULTS_OUT" \
    --wandb-project "$WANDB_PROJECT" \
    "${EXTRA_ARGS[@]}" \
    "$@"
