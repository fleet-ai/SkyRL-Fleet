#!/usr/bin/env bash
# Qwen3.5-9B GRPO training config for the negotiation RLVR environment.
# Text-only (NOT vision) — no mm_processor, MODALITY=negotiation.
#
# Environment: 2-player item-division negotiation (Deal or No Deal / CaSiNo).
#   - "you" side: the trained policy
#   - "them" side: an opponent LLM served via OpenRouter (litellm)
#   Reward is the policy's normalized self-score (verifiable from the game state).
#
# Reward ablation arms:
#   outcome       (default) — pure self-score reward
#   outcome_pareto          — self-score + PARETO_COEF * joint_efficiency (continuous
#                             joint-efficiency shaping, not the sparse binary Pareto flag)
# Switch arms by setting REWARD_MODE=outcome_pareto before launch.
#
# Orthogonal prompt ablation:
#   PROACTIVE_ELICITATION=1  — instructs both sides to probe/share private values before
#                              proposing (see the PROACTIVE_ELICITATION export below). Combine
#                              freely with either reward arm.
#
# THINKING IS OFF. Qwen3.5-9B is hybrid-reasoning, and for this short,
# turn-budgeted task thinking mode is the *worst* config — it burns the whole
# message budget on a <think> block and rarely commits a tag in time
# (~80% no_deal, see eval/REPORT.md "turn thinking OFF"). We disable it via
# enable_thinking:false for the policy, and opponent_no_think on the env side.
#
# Model: Qwen/Qwen3.5-9B (GDN MoE, ~1B active params, text-only here)
# Topology: 2 nodes x 8x H200 = 16 GPUs, TP=1 -> 16 inference engines.
#   NOTE: the node count itself is set by the SkyPilot task YAML (num_nodes: 2),
#   not this script. This script only sizes NUM_INFERENCE_ENGINES to match.
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
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export MODEL_TAG="${MODEL_TAG:-qwen35}"
export NEGOTIATION_DATASET="${NEGOTIATION_DATASET:-dnd}"  # dnd carries training signal; casino saturates
export NEGOTIATION_PROTOCOL="${NEGOTIATION_PROTOCOL:-single}"
export REWARD_MODE="${REWARD_MODE:-outcome}"  # set to outcome_pareto for ablation arm
export PARETO_COEF="${PARETO_COEF:-0.5}"
# Penalty per policy message whose prose promises an item to the opponent while
# the <propose> JSON keeps all of it (observed reward hack: gpt-4o-mini accepts
# based on the prose; the verifiable reward reads only the JSON — rate grew
# 10%->25% over steps 1-17 of the outcome baseline). Set 0 to disable.
export DECEPTION_PENALTY="${DECEPTION_PENALTY:--0.1}"
# Sublinear length penalty on total response tokens (policy/response_length), applied to the
# final reward in the generator. Counters the observed length runaway: in the outcome baseline
# (wandb sjwub9f2) response_length climbed ~1.5k->7k tokens over steps 9-63, saturating the
# 6*1024=6144 generation budget; no_deal then exploded 0.01->0.85 and you_norm collapsed
# 0.80->0.12 as the policy wrote essays and never committed a <propose>/<accept> in time.
# penalty = COEF * (tokens / REF) ** ALPHA  (fn=power; sqrt at ALPHA=0.5)
#   REF=0 -> auto = MAX_TURNS * MAX_GENERATE_LENGTH = 6144 (full budget).
# With COEF=0.2, ALPHA=0.5: a concise ~1.5k-token episode loses ~0.10, a budget-saturating
# ~6k-token one loses ~0.20 — a ~0.10 reward gap (on the scale of you_norm and DECEPTION_PENALTY)
# that nudges toward brevity without punishing legitimate multi-turn bargaining. Set 0 to disable.
export LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.2}"
export LENGTH_PENALTY_ALPHA="${LENGTH_PENALTY_ALPHA:-0.5}"
export LENGTH_PENALTY_FN="${LENGTH_PENALTY_FN:-power}"  # power (sqrt at alpha=0.5) | log
export LENGTH_PENALTY_REF="${LENGTH_PENALTY_REF:-0}"    # 0 -> MAX_TURNS * MAX_GENERATE_LENGTH
# Proactive preference-elicitation ablation (prompt-level). When 1, both the policy and the
# opponent system prompts get an instruction to probe/share private item values before proposing,
# rather than defaulting to a value-blind "fair" even split. This targets the dominant Pareto-gap
# driver diagnosed in research_logs/proactive.md: the proposer notes the partner's values are
# unknown and splits evenly without ever opening the information channel an integrative trade needs.
# Baked into the dataset prompts at prep time (passes --proactive to prepare_dataset.py). Set 0 to disable.
export PROACTIVE_ELICITATION="${PROACTIVE_ELICITATION:-0}"
export OPPONENT_MODEL="${OPPONENT_MODEL:-openrouter/openai/gpt-4o-mini}"
export MAX_TURNS="${MAX_TURNS:-6}"  # per-agent message budget; must match dataset prep --max_turns
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"  # negotiation transcripts are short
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-1024}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
# 2 nodes x 8 H200 = 16 GPUs; TP=1 (dense MoE fits on one GPU) -> 16 engines.
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-16}"
# Skip the per-job IB-HCA intersection on the RunPod SLURM cluster: /etc/nccl.conf
# already has the correct per-node NCCL_IB_HCA values; the intersection script was
# overriding them with wrong device names on nodes 8/9 (see agent/launch-run.md).
export SKIP_IB_INTERSECTION="${SKIP_IB_INTERSECTION:-1}"
export RUN_ID="${RUN_ID:-}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running (powers the opponent LLM via litellm/OpenRouter)}"

# Qwen3.5 GDN models can hang silently in the FlashInfer GDN JIT on GCP/RunPod
# (memory: commit 31098293); force the triton GDN prefill backend as a safe
# fallback. Harmless on hosts where FlashInfer would have worked.
export VLLM_GDN_PREFILL_BACKEND=triton

# Prepare the negotiation dataset into $DATA_DIR before launching training.
# fleet-common-run.sh will auto-point data.train_data / data.val_data at
# ${DATA_ROOT}/data/fleet/negotiation/, but we also pass explicit hydra overrides
# below (after --) so $HOME-rooted paths win regardless of DATA_ROOT resolution.
source .venv/bin/activate

DATA_DIR="${HOME}/data/fleet/negotiation"
PREP_ARGS=()
if [ "$PROACTIVE_ELICITATION" = "1" ]; then
  PREP_ARGS+=(--proactive)
fi
python3 skyrl-gym/skyrl_gym/envs/negotiation/prepare_dataset.py \
  --output_dir "$DATA_DIR" \
  --dataset "$NEGOTIATION_DATASET" \
  --protocol "$NEGOTIATION_PROTOCOL" \
  --max_turns "$MAX_TURNS" \
  ${PREP_ARGS[@]+"${PREP_ARGS[@]}"}

# Pareto arm: stronger regularization to prevent mode collapse (KL was 0.001 — far
# too weak to hold against the reward gradient; grad_norm spiked to 17 at collapse).
PARETO_ARGS=()
if [ "$REWARD_MODE" = "outcome_pareto" ]; then
  PARETO_ARGS=(
    trainer.algorithm.kl_loss_coef=0.05
    trainer.policy.optimizer_config.max_grad_norm=0.5
    "environment.skyrl_gym.negotiation.invalid_penalty=-0.05"
  )
fi

# Tag the run name when the proactive-elicitation ablation arm is active.
PROACTIVE_TAG=""
if [ "$PROACTIVE_ELICITATION" = "1" ]; then
  PROACTIVE_TAG="_proactive"
fi

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
  trainer.max_ckpts_to_keep=3 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.length_penalty_coef=$LENGTH_PENALTY_COEF \
  generator.length_penalty_alpha=$LENGTH_PENALTY_ALPHA \
  generator.length_penalty_fn=$LENGTH_PENALTY_FN \
  generator.length_penalty_ref=$LENGTH_PENALTY_REF \
  'generator.sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]' \
  'generator.eval_sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]' \
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
  generator.gpu_memory_utilization=0.8 \
  generator.inject_context_status=true \
  generator.context_warning_threshold=0.90 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-negotiation-grpo" \
  trainer.run_name="fleet_${MODEL_TAG}_9b_negotiation_${NEGOTIATION_DATASET}_${REWARD_MODE}${PROACTIVE_TAG}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_${MODEL_TAG}_9b_negotiation" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  ${PARETO_ARGS[@]+"${PARETO_ARGS[@]}"} \
  "$@"
