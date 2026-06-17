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
# THINKING IS ON (default). The policy emits <think>...</think> before its action;
# the reasoning is kept in the saved traces but STRIPPED from the multi-turn transcript
# and the opponent's view via the qwen3_without_thinking template, so the policy's own
# context never carries prior-turn reasoning. Train and eval match. MAX_GENERATE_LENGTH
# is sized to give the think channel room. Set ENABLE_THINKING=false only for a
# token-in-token-out ablation.
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
export LENGTH_PENALTY_REF="${LENGTH_PENALTY_REF:-1500}"  # calibrated to operating length (~healthy episode tokens)
# Log a think vs visible (non-think) token breakdown of the response_ids the LP measures, plus how
# much of the penalty is attributable to thinking tokens. Logged to wandb as generate/thinking_tokens_mean,
# generate/visible_tokens_mean, generate/thinking_token_frac, and (LP on) generate/length_penalty_{visible,thinking}_mean.
# NOTE: with the qwen3_without_thinking template only the LAST turn's <think> survives in response_ids,
# so this reflects exactly the thinking the LP penalizes (not total thinking across all turns).
export LOG_THINKING_TOKEN_METRICS="${LOG_THINKING_TOKEN_METRICS:-false}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export OPPONENT_MODEL="${OPPONENT_MODEL:-openrouter/openai/gpt-4o-mini}"
# Adversary cost tracking (logged to wandb as environment/opponent_*tokens[_sum] and
# environment/opponent_cost_usd[_sum]). USD per 1M tokens for the active opponent.
# Defaults to gpt-5.5 pricing; set to 0 to log tokens only, or to gpt-4o-mini
# (0.15 / 0.60) when reverting the adversary.
export OPPONENT_PRICE_IN="${OPPONENT_PRICE_IN:-5.0}"
export OPPONENT_PRICE_OUT="${OPPONENT_PRICE_OUT:-30.0}"
# Self-hosted opponent endpoint (cost lever vs OpenRouter GPT-4o-mini). When
# OPPONENT_BASE_URL is set, the env drives the "them" side against an OpenAI-
# compatible vLLM server (see scripts/fleet-negotiation-opponent-serve.sh +
# tasks/negotiation-opponent-serve-35b-1node.yaml) instead of OpenRouter. Pair it
# with a litellm "openai/<served-name>" model string, e.g.:
#   OPPONENT_MODEL=openai/qwen35-opponent \
#   OPPONENT_BASE_URL=http://<host-node-ip>:6479/v1
export OPPONENT_BASE_URL="${OPPONENT_BASE_URL:-}"
# Aggressive-adversary arm (research_logs/negotiation-35b-thinking-leakage-06-16.md
# item 4): append ADVERSARY_AGGRESSIVE_BLOCK to the opponent's system prompt so it
# negotiates harder and actively exploits any preference/value the policy leaks
# (punishes the over-disclosure pathology). Pair with a capable opponent (gpt-5.5).
export OPPONENT_AGGRESSIVE="${OPPONENT_AGGRESSIVE:-false}"
# Self-play arm (item 3): point the env-played opponent at the LIVE training policy's
# own HTTP endpoint, so the policy negotiates against an up-to-date copy of ITSELF
# (true self-play — the opponent weights advance every step). No external API/cost.
# The trainer serves the policy at generator.inference_engine.http_endpoint_{host,port}
# (default 127.0.0.1:8000, served_model_name=policy); we route the opponent there via
# litellm's openai provider. Overrides OPPONENT_MODEL/BASE_URL/pricing when true.
export SELF_PLAY="${SELF_PLAY:-false}"
export SELF_PLAY_BASE_URL="${SELF_PLAY_BASE_URL:-http://127.0.0.1:8000/v1}"
if [ "$SELF_PLAY" = "true" ]; then
  OPPONENT_MODEL="openai/policy"
  OPPONENT_BASE_URL="$SELF_PLAY_BASE_URL"
  OPPONENT_PRICE_IN=0.0
  OPPONENT_PRICE_OUT=0.0
  echo "=== SELF-PLAY: opponent = live policy endpoint ($OPPONENT_BASE_URL, model=policy); no external opponent API ==="
fi
# In-loop exploitation probe (run_probe.py): plays the live policy vs a scripted
# Python conceder ($0, no external API) every eval cycle. Logged as eval/probe/*.
export PROBE_EVAL="${PROBE_EVAL:-true}"
export PROBE_N="${PROBE_N:-16}"
export PROBE_DATASET="${PROBE_DATASET:-dnd}"
# In-loop LLM-as-judge deception probe: scores the policy messages produced inside
# the probe games above for deception (false_promise / omission). Measurement only
# (never a reward). Logged as eval/deception_judge/*. Paid but cheap -- a cheap
# judge (JUDGE_MODEL) does a few dozen single-shot classifications per eval, via
# OPENROUTER_API_KEY. gpt-4.1-mini was calibrated on past traces (gpt-4o-mini
# over-flags honest bargaining; see .overnight/judge_modelcmp_out.json).
export JUDGE_EVAL="${JUDGE_EVAL:-true}"
export JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1-mini}"
export MAX_TURNS="${MAX_TURNS:-6}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"  # thinking arm needs room (>=4096); see grad-explosion log
# Opponent thinking. By default the opponent runs /no_think with a small (512-tok)
# cap. For SELF-PLAY the opponent IS the policy, so let it THINK like the policy:
# its <think> is stripped before it enters the opponent's history or the policy's
# observation (see env._step_single), so reasoning never leaks/carries forward — it
# only shapes that turn's reply. The cap then matches the policy's own per-turn
# budget (MAX_GENERATE_LENGTH) so the opponent is never truncated mid-<think> — true
# symmetry with "self". This costs nothing in the common case (vLLM stops at EOS;
# healthy turns are ~0.3-1.5k tok per the 06-15/06-12 logs), it's only headroom.
# Both still env-overridable (an explicit OPPONENT_NO_THINK/OPPONENT_MAX_TOKENS wins).
if [ "$SELF_PLAY" = "true" ]; then
  export OPPONENT_NO_THINK="${OPPONENT_NO_THINK:-false}"
  export OPPONENT_MAX_TOKENS="${OPPONENT_MAX_TOKENS:-$MAX_GENERATE_LENGTH}"
else
  export OPPONENT_NO_THINK="${OPPONENT_NO_THINK:-true}"
  export OPPONENT_MAX_TOKENS="${OPPONENT_MAX_TOKENS:-512}"
fi
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
# Cap on validation scenarios (0 = use all 293 deduped dnd/val). Subsampled with a
# fixed seed so eval cost (n_prompts * eval_n_samples_per_prompt full games vs the
# OpenRouter opponent) stays bounded. ~128 is plenty for a stable eval signal.
export MAX_VAL="${MAX_VAL:-64}"
# In-loop held-out eval set appended as a SECOND eval parquet (checks the policy
# isn't just memorizing dnd / measures transfer). Logged separately in wandb as
# eval/negotiation_<EXTRA_VAL_DATASET>/* vs eval/negotiation_dnd/*. Skipped when it
# equals NEGOTIATION_DATASET (not held out then) or EXTRA_VAL_N=0.
#   synthetic (default) — procedurally generated: 4-6 items, asymmetric totals,
#     zero-value/conflict items, controllable integrative headroom. HIGH discrimination
#     for joint_eff/pareto, which is what we're optimizing.
#   casino — real CaSiNo corpus (food/water/firewood). NOTE: it saturates at ~92%
#     joint-eff with ~no spread (see eval/REPORT.md), so it's a weak in-loop signal for
#     the integrative hypothesis — prefer it in the OFFLINE harness. Set
#     EXTRA_VAL_DATASET=casino EXTRA_VAL_N=36 to restore the old behavior.
export EXTRA_VAL_DATASET="${EXTRA_VAL_DATASET:-synthetic}"
export EXTRA_VAL_N="${EXTRA_VAL_N:-${CASINO_EVAL:-64}}"
# In-loop eval composition. EVAL_DND=false drops the in-distribution dnd validation
# set (redundant with training reward) and runs ONLY the held-out transfer set
# (EXTRA_VAL_DATASET). Default true keeps both for backward compatibility.
export EVAL_DND="${EVAL_DND:-true}"
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
if [ -n "$OPPONENT_BASE_URL" ]; then
  # Self-hosted opponent endpoint: no OpenRouter needed. litellm's openai provider
  # still requires *some* api key in the request header; the vLLM server ignores it
  # unless launched with --api-key (OPPONENT_API_KEY), so a placeholder is fine.
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  echo "=== Opponent: self-hosted vLLM at $OPPONENT_BASE_URL (model=$OPPONENT_MODEL); OPENROUTER_API_KEY not required ==="
else
  : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running (powers the opponent LLM via litellm/OpenRouter), or set OPPONENT_BASE_URL to use a self-hosted endpoint}"
fi

# Qwen3.5 GDN models can hang silently in the FlashInfer GDN JIT on GCP/RunPod
# (see fleet-35b-run.sh); force the triton GDN prefill backend.
export VLLM_GDN_PREFILL_BACKEND=triton

source .venv/bin/activate

DATA_DIR="${HOME}/data/fleet/negotiation"
EXTRA_VAL_ARGS=()
_dnd_val="'${DATA_DIR}/validation.parquet'"
_extra_val=""
if [ "${EXTRA_VAL_N}" != "0" ] && [ "$NEGOTIATION_DATASET" != "$EXTRA_VAL_DATASET" ]; then
  # casino uses corpus split 'all'; synthetic is procedurally generated (split ignored).
  _xv_split=all; [ "$EXTRA_VAL_DATASET" = "synthetic" ] && _xv_split=val
  EXTRA_VAL_ARGS=(--extra_val_dataset "$EXTRA_VAL_DATASET" --extra_val_split "$_xv_split" --max_extra_val "$EXTRA_VAL_N")
  _extra_val="'${DATA_DIR}/validation_${EXTRA_VAL_DATASET}.parquet'"
fi
# Assemble the in-loop eval sets. EVAL_DND=false drops the in-distribution dnd
# validation set (it largely tracks reward/avg_raw_reward) and keeps ONLY the
# held-out transfer set (the unique signal). Falls back to dnd val if no extra
# set is configured, so eval is never empty.
if [ "${EVAL_DND:-true}" = "false" ] && [ -n "$_extra_val" ]; then
  VAL_DATA="[${_extra_val}]"
elif [ -n "$_extra_val" ]; then
  VAL_DATA="[${_dnd_val},${_extra_val}]"
else
  VAL_DATA="[${_dnd_val}]"
fi
# Preference-elicitation arm (the 2x2 Elicitation factor). NEGOTIATION_ELICIT=two_sided
# injects the prepared proactive mutual-disclosure instruction (ask their priorities +
# state yours, route by value) into BOTH system prompts via prepare_dataset --proactive.
# 'none' (default) = no elicitation. ('one_sided' is not yet wired — needs an ask-only
# prompt variant.) Cells: C3=outcome+two_sided, C4=outcome_jointeff+two_sided.
ELICIT_ARGS=()
case "${NEGOTIATION_ELICIT:-none}" in
  two_sided) ELICIT_ARGS=(--elicit two_sided) ;;
  one_sided) ELICIT_ARGS=(--elicit one_sided) ;;
  none) ;;
  *) echo "WARN: NEGOTIATION_ELICIT='${NEGOTIATION_ELICIT}' unsupported (use none|two_sided|one_sided); treating as none" ;;
esac
python3 skyrl-gym/skyrl_gym/envs/negotiation/prepare_dataset.py \
  --output_dir "$DATA_DIR" \
  --dataset "$NEGOTIATION_DATASET" \
  --protocol "$NEGOTIATION_PROTOCOL" \
  --max_turns "$MAX_TURNS" \
  --max_val "$MAX_VAL" \
  ${ELICIT_ARGS[@]+"${ELICIT_ARGS[@]}"} \
  ${EXTRA_VAL_ARGS[@]+"${EXTRA_VAL_ARGS[@]}"}

# Pareto arm: stronger regularization to prevent mode collapse.
# (outcome_jointeff is an exact alias of outcome_pareto — the continuous joint-eff
#  reward used by the 2x2 elicitation experiment — so it gets the same regularization.)
PARETO_ARGS=()
if [ "$REWARD_MODE" = "outcome_pareto" ] || [ "$REWARD_MODE" = "outcome_jointeff" ]; then
  # Stabilized jointeff defaults (the 2026-06-13 c2 run blew up: grad 4->54, KL ->1.8,
  # entropy decay, reward regression at ~step 55 — the jointeff reward + strong leak
  # penalty drove a KL/grad runaway that kl=0.05/grad=0.5/lr=5e-7 couldn't contain).
  # Stronger KL + entropy anchor, tighter clip, lower lr. Still overridable via the
  # TUNE_ARGS env vars (LR / KL_LOSS_COEF_FINAL / ENTROPY_COEF_FINAL / MAX_GRAD_NORM_FINAL).
  PARETO_ARGS=(
    trainer.algorithm.kl_loss_coef=${KL_LOSS_COEF_FINAL:-0.1}
    trainer.algorithm.entropy_loss_coef=${ENTROPY_COEF_FINAL:-0.01}
    trainer.policy.optimizer_config.max_grad_norm=${MAX_GRAD_NORM_FINAL:-0.3}
    trainer.policy.optimizer_config.lr=${LR:-3e-7}
    "environment.skyrl_gym.negotiation.invalid_penalty=-0.05"
  )
fi

# Training-stability hardening (research_logs/negotiation-35b-grad-explosion-reward-
# regression-2026-06-12.md): the outcome/thinking baseline peaked ~step 70 then
# entropy-collapsed -> grad_norm 4->500 -> token-repetition output collapse -> reward
# regressed 0.83->0.55. Counter the root cause (entropy collapse) with an entropy floor
# + stronger KL anchor (default kl_loss_coef was a negligible 0.001), cap the destructive
# update magnitude, drop runaway/malformed rollouts from the batch, and penalise
# non-parsing actions. lr stays 5e-7: the post-mortem's "5e-6" misread the unused
# optimizer block; the trained policy_lr logged 5e-7 every step, so we do NOT raise it.
# Each knob is env-overridable for sweeps. Pareto arm overrides kl/grad/invalid below.
STABILITY_ARGS=(
  trainer.algorithm.use_entropy_loss=${USE_ENTROPY_LOSS:-true}
  trainer.algorithm.entropy_loss_coef=${ENTROPY_LOSS_COEF:-0.005}
  trainer.algorithm.kl_loss_coef=${KL_LOSS_COEF:-0.02}
  trainer.policy.optimizer_config.max_grad_norm=${MAX_GRAD_NORM:-0.5}
  generator.apply_overlong_filtering=${APPLY_OVERLONG_FILTERING:-true}
  environment.skyrl_gym.negotiation.invalid_penalty=${INVALID_PENALTY:--0.05}
)

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
    # The qwen3_without_thinking custom template retokenizes the chat history each
    # turn (to strip <think> from non-last turns), which breaks the per-token
    # alignment of rollout logprobs. The generator hard-raises "Response Logprobs
    # bookkeeping is not supported with custom chat template" if the engine returns
    # them, so we MUST disable rollout logprobs here (-> 100% trajectory_error
    # otherwise). Safe: off_policy_correction.tis_ratio_type is null (no importance
    # sampling), so the loss uses forward-pass logprobs, not rollout logprobs.
    generator.sampling_params.logprobs=null
    generator.eval_sampling_params.logprobs=null
    # Thinking-channel integrity (research_logs/negotiation-35b-grad-explosion-reward-
    # regression-2026-06-12.md "Thinking-channel abandonment + private-value leak"):
    # by ~step 70 the policy emitted empty <think></think> and moved its reasoning into
    # the visible message, leaking its own item valuations to the opponent. Penalise
    # abandoning the think channel and disclosing values; the env logs think_nonempty_rate
    # and value_leak_msgs so this is visible during training, not only in post-hoc eval.
    environment.skyrl_gym.negotiation.empty_think_penalty=${EMPTY_THINK_PENALTY:--0.02}
    environment.skyrl_gym.negotiation.value_leak_penalty=${VALUE_LEAK_PENALTY:--0.05}
  )
else
  THINK_ARGS=(
    +generator.chat_template_kwargs.enable_thinking=false
    'generator.sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]'
    'generator.eval_sampling_params.stop=["</propose>","</deal>","<accept>","</think>"]'
  )
fi

# Late stabilization overrides (applied LAST -> win over PARETO_ARGS and the main args).
# Used to tame the KL/grad runaway seen on the jointeff arm (grad 4->54, KL ->1.8,
# entropy decay, reward regression at ~step 55): lower lr, stronger KL/entropy anchor,
# tighter clip. Each is opt-in via env; unset leaves the upstream value unchanged.
TUNE_ARGS=()
[ -n "${LR:-}" ]                  && TUNE_ARGS+=("trainer.policy.optimizer_config.lr=$LR")
[ -n "${KL_LOSS_COEF_FINAL:-}" ]  && TUNE_ARGS+=("trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF_FINAL")
[ -n "${ENTROPY_COEF_FINAL:-}" ]  && TUNE_ARGS+=("trainer.algorithm.entropy_loss_coef=$ENTROPY_COEF_FINAL")
[ -n "${MAX_GRAD_NORM_FINAL:-}" ] && TUNE_ARGS+=("trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM_FINAL")

# Opponent endpoint override: when OPPONENT_BASE_URL is set, route the env-played
# opponent at a self-hosted OpenAI-compatible vLLM server (the hosted 35B endpoint)
# instead of OpenRouter. Unset -> the env default (OpenRouter) is used unchanged.
OPPONENT_ARGS=()
if [ -n "$OPPONENT_BASE_URL" ]; then
  OPPONENT_ARGS=("environment.skyrl_gym.negotiation.opponent_base_url=$OPPONENT_BASE_URL")
fi

RUN_NAME="fleet_${MODEL_TAG}_35b_negotiation_${NEGOTIATION_DATASET}_${REWARD_MODE}_${RUN_ID:-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')}"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint integrations.fleet.entrypoints.main_negotiation \
  --env-class negotiation \
  --data-dir-name negotiation -- \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=${VAL_DATA}" \
  environment.skyrl_gym.negotiation.reward_mode=$REWARD_MODE \
  environment.skyrl_gym.negotiation.pareto_coef=$PARETO_COEF \
  environment.skyrl_gym.negotiation.deception_penalty=$DECEPTION_PENALTY \
  environment.skyrl_gym.negotiation.protocol=$NEGOTIATION_PROTOCOL \
  environment.skyrl_gym.negotiation.opponent_model=$OPPONENT_MODEL \
  environment.skyrl_gym.negotiation.opponent_no_think=$OPPONENT_NO_THINK \
  environment.skyrl_gym.negotiation.opponent_max_tokens=$OPPONENT_MAX_TOKENS \
  environment.skyrl_gym.negotiation.opponent_aggressive=$OPPONENT_AGGRESSIVE \
  environment.skyrl_gym.negotiation.opponent_price_per_mtok_in=$OPPONENT_PRICE_IN \
  environment.skyrl_gym.negotiation.opponent_price_per_mtok_out=$OPPONENT_PRICE_OUT \
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
  trainer.train_batch_size=${TRAIN_BATCH_SIZE:-16} \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE:-16} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=${HF_SAVE_INTERVAL:-30} \
  trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP:-2} \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  generator.length_penalty_coef=$LENGTH_PENALTY_COEF \
  generator.length_penalty_alpha=$LENGTH_PENALTY_ALPHA \
  generator.length_penalty_fn=$LENGTH_PENALTY_FN \
  generator.length_penalty_ref=$LENGTH_PENALTY_REF \
  generator.log_thinking_token_metrics=$LOG_THINKING_TOKEN_METRICS \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.zero_variance_filter=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.served_model_name=policy \
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
  ${STABILITY_ARGS[@]+"${STABILITY_ARGS[@]}"} \
  ${THINK_ARGS[@]+"${THINK_ARGS[@]}"} \
  ${PARETO_ARGS[@]+"${PARETO_ARGS[@]}"} \
  ${TUNE_ARGS[@]+"${TUNE_ARGS[@]}"} \
  ${OPPONENT_ARGS[@]+"${OPPONENT_ARGS[@]}"} \
  "$@"
