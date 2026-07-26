#!/usr/bin/env bash
# Qwen3.5-9B GRPO for the leaky-lies self-play poker env (leaky_poker). Text-only.
#
# The policy plays the hero seat of heads-up Hold'em; reward is the DENSE per-bluff leakage feature
# (reward_scale*(bluff_gain - leak_lambda*leakage)), leakage served by the reader service
# (READER_BASE_URL). Two opponent arms, selected by OPPONENT_MODE:
#   exploiter (default) — a fixed in-env BEHAVIORAL bluff-catcher (frozen-exploiter arm). No serving.
#   llm                 — a served policy snapshot (self-play arm); set OPPONENT_BASE_URL + rotate it
#                         with fleet-negotiation-opponent-serve.sh style serving.
# Reward arms by REWARD_MODE: dense (headline) | sparse (realized-chips baseline; needs no reader).
# Eval is pinned to the frozen exploiter by the dataset (val rows carry extra_info.opponent_mode),
# so win-rate is measured against a STATIONARY opponent regardless of the training arm.
#
# Required: WANDB_API_KEY; for REWARD_MODE=dense also a running reader_service.py at READER_BASE_URL.
#   Start it first:
#     CUDA_VISIBLE_DEVICES=0 /workspace/allie/performative/.venv/bin/python \
#       /workspace/allie/TextArena/deception_poc/reader_service.py --port 8137 &
#     export READER_BASE_URL=http://127.0.0.1:8137
set -euo pipefail
cd "$(dirname "$0")/.."

export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-leaky_poker}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export MODEL_TAG="${MODEL_TAG:-qwen35}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"   # 1 node x 8 H200, TP=1

# --- arm config ---
export REWARD_MODE="${REWARD_MODE:-dense}"          # dense | sparse
export OPPONENT_MODE="${OPPONENT_MODE:-exploiter}"  # exploiter | llm | scripted
export LEAK_LAMBDA="${LEAK_LAMBDA:-100.0}"
export READER_MODE="${READER_MODE:-endpoint}"       # endpoint | stub | local
export READER_BASE_URL="${READER_BASE_URL:-http://127.0.0.1:8137}"
export OPPONENT_MODEL="${OPPONENT_MODEL:-openrouter/Qwen/Qwen3.5-9B}"
export OPPONENT_BASE_URL="${OPPONENT_BASE_URL:-}"
export HOLD_LIE_RATE="${HOLD_LIE_RATE:-true}"
export NUM_ROUNDS="${NUM_ROUNDS:-4}"
export MAX_TURNS="${MAX_TURNS:-48}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-64}"   # a bracket action is tiny

# --- dataset ---
DATA_ROOT="${DATA_ROOT:-$HOME}"
DATA_DIR="${DATA_ROOT}/data/fleet/leaky_poker"
echo "[leaky_poker] preparing dataset -> $DATA_DIR"
uv run --isolated python skyrl-gym/skyrl_gym/envs/leaky_poker/prepare_dataset.py \
  --output_dir "$DATA_DIR" --n_train "${N_TRAIN:-2048}" --n_val "${N_VAL:-128}" \
  --num_rounds "$NUM_ROUNDS" || \
  /workspace/allie/performative/.venv/bin/python skyrl-gym/skyrl_gym/envs/leaky_poker/prepare_dataset.py \
    --output_dir "$DATA_DIR" --n_train "${N_TRAIN:-2048}" --n_val "${N_VAL:-128}" --num_rounds "$NUM_ROUNDS"

TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-$HOME/leaky_poker_transcripts}"
RUN_NAME="fleet_${MODEL_TAG}_9b_leakypoker_${OPPONENT_MODE}_${REWARD_MODE}_lam${LEAK_LAMBDA}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"

# New hydra section for a brand-new env: prefix every key with '+' so the leaky_poker config block
# is CREATED (it is not in the base config schema).
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --env-class leaky_poker \
  --data-dir-name leaky_poker -- \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=['${DATA_DIR}/validation.parquet']" \
  +environment.skyrl_gym.leaky_poker.reward_mode=$REWARD_MODE \
  +environment.skyrl_gym.leaky_poker.opponent_mode=$OPPONENT_MODE \
  +environment.skyrl_gym.leaky_poker.leak_lambda=$LEAK_LAMBDA \
  +environment.skyrl_gym.leaky_poker.reader_mode=$READER_MODE \
  +environment.skyrl_gym.leaky_poker.reader_base_url=$READER_BASE_URL \
  +environment.skyrl_gym.leaky_poker.opponent_model=$OPPONENT_MODEL \
  +environment.skyrl_gym.leaky_poker.opponent_base_url=$OPPONENT_BASE_URL \
  +environment.skyrl_gym.leaky_poker.hold_lie_rate=$HOLD_LIE_RATE \
  +environment.skyrl_gym.leaky_poker.num_rounds=$NUM_ROUNDS \
  "+environment.skyrl_gym.leaky_poker.transcript_dir=$TRANSCRIPT_DIR/$RUN_NAME" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  +generator.chat_template_kwargs.enable_thinking=false \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=${HF_SAVE_INTERVAL:-30} \
  trainer.max_ckpts_to_keep=3 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=3 \
  generator.enforce_eager=false \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-leaky-poker-grpo" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_${MODEL_TAG}_9b_leakypoker_${OPPONENT_MODE}_${REWARD_MODE}" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
