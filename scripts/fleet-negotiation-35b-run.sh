#!/usr/bin/env bash
# Qwen3.5-35B-A3B GRPO training config for the negotiation RLVR environment.
# Text-only (NOT vision) — no mm_processor, MODALITY=negotiation.
#
# Environment: 2-player item-division negotiation (Deal or No Deal / CaSiNo).
#   - "you" side: the trained policy
#   - "them" side: an opponent LLM served via OpenRouter (litellm)
#   Reward is the policy's normalized self-score (verifiable from the game state).
#
# Reward ablation arms:
#   outcome       (default) — pure self-score reward
#   outcome_pareto          — self-score + weighted Pareto bonus (PARETO_COEF)
# Switch arms by setting REWARD_MODE=outcome_pareto before launch.
#
# THINKING IS OFF. Same reasoning as the 9B config: 6-turn budget is too short for
# <think> blocks — they consume the entire turn without emitting a <propose> tag,
# yielding ~80% no_deal. The 35B model is more capable but the hard turn cap is
# architectural, not a capability gap. Enable if you have a longer turn budget.
#
# Model: Qwen/Qwen3.5-35B-A3B (MoE, 35B total / ~3B active, text-only)
# Topology: 1 node x 8x H200 = 8 GPUs; TP=2 -> 4 inference engines.
#   Training uses FSDP2 across all 8 GPUs (shard=full, no TP in trainer).
#   NOTE: node count is set by the SkyPilot task YAML (num_nodes: 1), not here.
#
# Required env vars: WANDB_API_KEY, OPENROUTER_API_KEY
#   OPENROUTER_API_KEY powers the opponent LLM (env "them" side via litellm)
# Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root (scripts/ is directly under repo root)

# Defaults for vars normally set by SkyPilot YAML envs block
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-negotiation}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
export MODEL_TAG="${MODEL_TAG:-qwen35}"
export NEGOTIATION_DATASET="${NEGOTIATION_DATASET:-dnd}"
export NEGOTIATION_PROTOCOL="${NEGOTIATION_PROTOCOL:-single}"
export REWARD_MODE="${REWARD_MODE:-outcome}"  # set to outcome_pareto for ablation arm
export PARETO_COEF="${PARETO_COEF:-0.5}"
# Penalty per policy message whose prose promises an item while the <propose> JSON
# keeps all of it. Set 0 to disable.
export DECEPTION_PENALTY="${DECEPTION_PENALTY:--0.1}"
# Sublinear length penalty on total response tokens. See fleet-negotiation-9b-run.sh
# for the full rationale (counters length runaway observed in the outcome baseline).
# penalty = COEF * (tokens / REF) ** ALPHA  (fn=power; sqrt at ALPHA=0.5)
#   REF=0 -> auto = MAX_TURNS * MAX_GENERATE_LENGTH = 6144 (full budget).
export LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.2}"
export LENGTH_PENALTY_ALPHA="${LENGTH_PENALTY_ALPHA:-0.5}"
export LENGTH_PENALTY_FN="${LENGTH_PENALTY_FN:-power}"  # power (sqrt at alpha=0.5) | log
export LENGTH_PENALTY_REF="${LENGTH_PENALTY_REF:-0}"    # 0 -> MAX_TURNS * MAX_GENERATE_LENGTH
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export OPPONENT_MODEL="${OPPONENT_MODEL:-openrouter/openai/gpt-4o-mini}"
export MAX_TURNS="${MAX_TURNS:-6}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-1024}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
# Cap on validation scenarios (0 = use all 293 deduped dnd/val). Subsampled with a
# fixed seed so eval cost (n_prompts * eval_n_samples_per_prompt full games vs the
# OpenRouter opponent) stays bounded. ~128 is plenty for a stable eval signal.
export MAX_VAL="${MAX_VAL:-64}"
# 1 node x 8 H200; TP=2 -> 4 inference engines.
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"
# Read IB HCA from /etc/nccl.conf (correct IB-only list; intersection script was
# picking up Ethernet adapters on nodes 8/9 — see env-fixes doc Fix 2).
export SKIP_IB_INTERSECTION="${SKIP_IB_INTERSECTION:-1}"
export RUN_ID="${RUN_ID:-}"
# Where to persist full episode transcripts (policy "you" turns keep their <think>
# reasoning) for inspection. The env appends one JSON line per finished episode to a
# per-process file under this dir. Set empty to disable.
export TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-$HOME/exports/negotiation_transcripts}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running (powers the opponent LLM via litellm/OpenRouter)}"

# Qwen3.5 GDN models can hang silently in the FlashInfer GDN JIT on GCP/RunPod
# (see fleet-35b-run.sh); force the triton GDN prefill backend.
export VLLM_GDN_PREFILL_BACKEND=triton

source .venv/bin/activate

DATA_DIR="${HOME}/data/fleet/negotiation"
python3 skyrl-gym/skyrl_gym/envs/negotiation/prepare_dataset.py \
  --output_dir "$DATA_DIR" \
  --dataset "$NEGOTIATION_DATASET" \
  --protocol "$NEGOTIATION_PROTOCOL" \
  --max_turns "$MAX_TURNS" \
  --max_val "$MAX_VAL"

# Pareto arm: stronger regularization to prevent mode collapse.
PARETO_ARGS=()
if [ "$REWARD_MODE" = "outcome_pareto" ]; then
  PARETO_ARGS=(
    trainer.algorithm.kl_loss_coef=0.05
    trainer.policy.optimizer_config.max_grad_norm=0.5
    "environment.skyrl_gym.negotiation.invalid_penalty=-0.05"
  )
fi

# Thinking arm: when ENABLE_THINKING=true the policy emits <think>...</think> before
# its action. Two things must change vs the no-think default:
#   1. Use the qwen3_without_thinking custom chat template. This retokenizes the chat
#      history each turn and strips <think> from every NON-last assistant turn, so the
#      policy's own multi-turn training context never carries prior-turn reasoning
#      (matches Qwen3 inference behaviour). The full reasoning is still saved for
#      inspection via transcript_dir below.
#   2. Drop "</think>" from the stop strings. With thinking on it must NOT stop at the
#      end of the reasoning block — the model needs to continue and emit <propose>/
#      <accept>/<deal> in the same turn.
# The no-think default keeps token-in-token-out (the tuned recipe) and the original
# stop set including "</think>".
if [ "$ENABLE_THINKING" = "true" ]; then
  THINK_ARGS=(
    generator.chat_template.source=name
    generator.chat_template.name_or_path=qwen3_without_thinking
    'generator.sampling_params.stop=["</propose>","</deal>","<accept>"]'
    'generator.eval_sampling_params.stop=["</propose>","</deal>","<accept>"]'
  )
else
  THINK_ARGS=(
    +generator.chat_template_kwargs.enable_thinking=false
    'generator.sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]'
    'generator.eval_sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]'
  )
fi

RUN_NAME="fleet_${MODEL_TAG}_35b_negotiation_${NEGOTIATION_DATASET}_${REWARD_MODE}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --env-class negotiation \
  --data-dir-name negotiation -- \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=['${DATA_DIR}/validation.parquet']" \
  environment.skyrl_gym.negotiation.reward_mode=$REWARD_MODE \
  environment.skyrl_gym.negotiation.pareto_coef=$PARETO_COEF \
  environment.skyrl_gym.negotiation.deception_penalty=$DECEPTION_PENALTY \
  environment.skyrl_gym.negotiation.protocol=$NEGOTIATION_PROTOCOL \
  environment.skyrl_gym.negotiation.opponent_model=$OPPONENT_MODEL \
  "environment.skyrl_gym.negotiation.transcript_dir=${TRANSCRIPT_DIR:+$TRANSCRIPT_DIR/$RUN_NAME}" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  generator.inference_engine_tensor_parallel_size=2 \
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
  trainer.max_ckpts_to_keep=2 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.length_penalty_coef=$LENGTH_PENALTY_COEF \
  generator.length_penalty_alpha=$LENGTH_PENALTY_ALPHA \
  generator.length_penalty_fn=$LENGTH_PENALTY_FN \
  generator.length_penalty_ref=$LENGTH_PENALTY_REF \
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
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=3 \
  generator.enforce_eager=false \
  generator.gpu_memory_utilization=0.75 \
  generator.inject_context_status=true \
  generator.context_warning_threshold=0.90 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-negotiation-grpo" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_${MODEL_TAG}_35b_negotiation" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  ${THINK_ARGS[@]+"${THINK_ARGS[@]}"} \
  ${PARETO_ARGS[@]+"${PARETO_ARGS[@]}"} \
  "$@"
