#!/bin/bash
set -x

# Launch GRPO training on arc-witness-envs via SkyPilot (GCP).
#
# Prerequisites:
#   1. sky check (verify GCP credentials)
#   2. Push witness integration code to guanghan/arc-agi-training branch
#   3. Set environment variables below or pass via --env
#
# Usage:
#   bash examples/train_integrations/witness/run_witness.sh
#
# Override defaults:
#   GAME_IDS="tw10 tw04" MODEL="Qwen/Qwen2.5-VL-7B-Instruct" \
#     bash examples/train_integrations/witness/run_witness.sh

: "${FLEET_API_KEY:?Set FLEET_API_KEY}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

: "${GAME_IDS:=tw10}"
: "${REWARD_MODE:=shaped}"
: "${OBS_MODE:=grid}"
: "${MODEL:=Qwen/Qwen3.5-9B}"
: "${MAX_TURNS:=50}"
: "${NUM_EPOCHS:=20}"

sky launch tasks/witness-grpo.yaml \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env GAME_IDS="$GAME_IDS" \
  --env REWARD_MODE="$REWARD_MODE" \
  --env OBS_MODE="$OBS_MODE" \
  --env MODEL="$MODEL" \
  --env MAX_TURNS="$MAX_TURNS" \
  --env NUM_EPOCHS="$NUM_EPOCHS" \
  -y --down
