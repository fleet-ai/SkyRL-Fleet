#!/usr/bin/env bash
# Warm-start ablation launcher for the SkyRL negotiation RL environment.
#
# Launches ONE arm of the SFT warm-start ablation
# (spec: skyrl-gym/skyrl_gym/envs/negotiation/behavior_warmstart.md).
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  CLOBBER SAFETY                                                              │
# │  fleet-negotiation-35b-run.sh hardcodes trainer.ckpt_path keyed on          │
# │  MODEL_TAG alone. Two arms sharing MODEL_TAG would write to the same dir     │
# │  and clobber / wrongly resume from each other. This wrapper assigns a        │
# │  DISTINCT ckpt_path per arm (appended after "--" as a trailing Hydra arg;    │
# │  OmegaConf.from_cli is last-wins, so the trailing value overrides the        │
# │  hardcoded earlier one — see integrations/fleet/entrypoints/main_fleet.py    │
# │  and skyrl/train/config/config.py SkyRLTrainConfig.from_cli_overrides).      │
# │                                                                              │
# │  IDENTICAL RL HYPERPARAMETERS ACROSS ARMS                                    │
# │  The ONLY differences between baseline and warm are:                         │
# │    • MODEL_PATH  (warm = SFT checkpoint; baseline = default base model)      │
# │    • RUN_ID      (warm = ws_warm_<dataset>; baseline = ws_baseline)          │
# │    • ckpt_path   (derived from RUN_ID above)                                 │
# │  EVERYTHING ELSE — reward mode, penalties, thinking, lr, epochs, opponent,   │
# │  seed — MUST be passed identically to both arms. Use env vars (REWARD_MODE,  │
# │  ENABLE_THINKING, NUM_EPOCHS, etc.) and/or trailing Hydra overrides ("$@").  │
# │  Whatever you pass to the baseline arm, pass verbatim to the warm arm.       │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# Required env vars:
#   ARM              baseline | warm
#   WANDB_API_KEY    (required by fleet-negotiation-35b-run.sh)
#   OPENROUTER_API_KEY  (powers the opponent LLM via litellm/OpenRouter)
#
# Optional env vars (warm arm only):
#   SFT_DATASET      casino (default) | dnd   — selects SFT ckpt path + run name
#   SFT_CKPT         explicit path to SFT HF checkpoint dir
#                    (default: $HOME/exports/sft_${SFT_DATASET})
#
# Optional env vars forwarded to fleet-negotiation-35b-run.sh (pass identically
# to BOTH arms to keep RL hyperparameters frozen):
#   REWARD_MODE, ENABLE_THINKING, NUM_EPOCHS, OPPONENT_MODEL, PARETO_COEF,
#   DECEPTION_PENALTY, LENGTH_PENALTY_*, NEGOTIATION_DATASET, MAX_TURNS,
#   MAX_GENERATE_LENGTH, MAX_INPUT_LENGTH, MAX_VAL, EXTRA_VAL_DATASET, etc.
#
# Trailing positional args ("$@") are forwarded as Hydra dotlist overrides to
# fleet-negotiation-35b-run.sh. OmegaConf is last-wins, so trailing overrides
# take precedence over anything the run script sets earlier. Pass IDENTICAL
# trailing args to both arms.
#
# Examples:
#
#   # Baseline arm (default RL config):
#   ARM=baseline bash scripts/fleet-negotiation-warmstart-run.sh
#
#   # Warm arm (CaSiNo SFT prior, default):
#   ARM=warm SFT_DATASET=casino bash scripts/fleet-negotiation-warmstart-run.sh
#
#   # Both arms with RL hyperparameters frozen (pass the same env + trailing overrides):
#   export ENABLE_THINKING=true NUM_EPOCHS=30 REWARD_MODE=outcome
#   ARM=baseline bash scripts/fleet-negotiation-warmstart-run.sh \
#       trainer.algorithm.kl_loss_coef=0.02
#   ARM=warm SFT_DATASET=casino bash scripts/fleet-negotiation-warmstart-run.sh \
#       trainer.algorithm.kl_loss_coef=0.02
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL repo root (scripts/ is directly under repo root)

# ---------------------------------------------------------------------------
# Validate ARM
# ---------------------------------------------------------------------------
ARM="${ARM:?ARM is required. Set ARM=baseline or ARM=warm.}"
if [ "$ARM" != "baseline" ] && [ "$ARM" != "warm" ]; then
  echo "ERROR: ARM must be 'baseline' or 'warm', got: '${ARM}'" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Reward isolation (spec: behavior_warmstart.md "RL Training" — reward must be
# the negotiation OUTCOME ONLY, with NO behavioral penalties).
#
# This ablation MEASURES the emergence of deception and value-leak as failure
# modes. Shaping those behaviors into the reward (deception/value-leak/empty-think
# penalties) would suppress the very thing we're measuring and confound the
# warm-start vs baseline comparison. So we force the pure-outcome reward and zero
# every behavioral penalty for BOTH arms. The behaviors are still TRACKED as
# metrics (deception_msgs / value_leak_msgs / think_nonempty_rate are logged
# regardless of the penalty coefficient — env.py decouples detection from
# shaping), they are simply never fed back into the reward.
#
# These default to 0 here (overriding fleet-negotiation-35b-run.sh's nonzero
# guardrail defaults of deception=-0.1, value_leak=-0.05, empty_think=-0.02).
# They are still overridable by setting them explicitly in the environment, but
# doing so BREAKS the isolation — only do it deliberately, and identically for
# both arms.
export REWARD_MODE="${REWARD_MODE:-outcome}"
export DECEPTION_PENALTY="${DECEPTION_PENALTY:-0}"
export VALUE_LEAK_PENALTY="${VALUE_LEAK_PENALTY:-0}"
export EMPTY_THINK_PENALTY="${EMPTY_THINK_PENALTY:-0}"

if [ "$REWARD_MODE" != "outcome" ] \
   || [ "$DECEPTION_PENALTY" != "0" ] \
   || [ "$VALUE_LEAK_PENALTY" != "0" ] \
   || [ "$EMPTY_THINK_PENALTY" != "0" ]; then
  echo "WARNING: reward isolation overridden — reward is NOT pure outcome." >&2
  echo "         REWARD_MODE=$REWARD_MODE DECEPTION_PENALTY=$DECEPTION_PENALTY" >&2
  echo "         VALUE_LEAK_PENALTY=$VALUE_LEAK_PENALTY EMPTY_THINK_PENALTY=$EMPTY_THINK_PENALTY" >&2
  echo "         This confounds the warm-start ablation (penalties suppress the" >&2
  echo "         deception / value-leak behaviors the experiment measures)." >&2
fi

# ---------------------------------------------------------------------------
# Resolve per-arm identity and paths
# ---------------------------------------------------------------------------
SFT_DATASET="${SFT_DATASET:-casino}"

if [ "$ARM" = "baseline" ]; then
  RUN_ID="${RUN_ID:-ws_baseline}"
  CKPT_PATH="$HOME/ckpts/${RUN_ID}"
  # MODEL_PATH: intentionally not overridden; fleet-negotiation-35b-run.sh defaults
  # to Qwen/Qwen3.5-35B-A3B. Inheriting MODEL_PATH from the environment is fine
  # only if it is genuinely the base model — do not accidentally pass an SFT ckpt here.

else
  # warm arm
  SFT_CKPT="${SFT_CKPT:-$HOME/exports/sft_${SFT_DATASET}}"
  RUN_ID="ws_warm_${SFT_DATASET}"
  CKPT_PATH="$HOME/ckpts/ws_warm_${SFT_DATASET}"

  # Verify the SFT checkpoint exists and looks like a HuggingFace model directory.
  if [ ! -f "${SFT_CKPT}/config.json" ]; then
    echo "" >&2
    echo "ERROR: SFT checkpoint not found: ${SFT_CKPT}/config.json" >&2
    echo "" >&2
    echo "  Run Stage 1 first:" >&2
    echo "    SFT_DATASET=${SFT_DATASET} bash scripts/fleet-negotiation-sft-run.sh" >&2
    echo "" >&2
    echo "  Then re-run this script." >&2
    echo "" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  SFT Warm-Start RL — Negotiation (ARM=${ARM})"
echo "========================================================"
if [ "$ARM" = "warm" ]; then
  echo "  SFT_DATASET : ${SFT_DATASET}"
  echo "  SFT_CKPT    : ${SFT_CKPT}"
  echo "  MODEL_PATH  : ${SFT_CKPT}  (SFT warm-start init)"
else
  echo "  MODEL_PATH  : ${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}  (base model; no SFT)"
fi
echo "  RUN_ID      : ${RUN_ID}"
echo "  CKPT_PATH   : ${CKPT_PATH}"
echo "  reward      : ${REWARD_MODE}  (deception=${DECEPTION_PENALTY} value_leak=${VALUE_LEAK_PENALTY} empty_think=${EMPTY_THINK_PENALTY})"
echo "                ^ behavioral penalties OFF: behaviors are measured, not shaped"
echo "  extra args  : $*"
echo "========================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch — trailing Hydra args override the run script's hardcoded values
# (OmegaConf last-wins). trainer.ckpt_path and trainer.resume_mode appear after
# all of the run script's own args, so they win over the hardcoded defaults.
# ---------------------------------------------------------------------------
if [ "$ARM" = "baseline" ]; then
  RUN_ID="$RUN_ID" \
    bash scripts/fleet-negotiation-35b-run.sh \
      "trainer.ckpt_path=${CKPT_PATH}" \
      trainer.resume_mode=latest \
      "$@"
else
  MODEL_PATH="$SFT_CKPT" RUN_ID="$RUN_ID" \
    bash scripts/fleet-negotiation-35b-run.sh \
      "trainer.ckpt_path=${CKPT_PATH}" \
      trainer.resume_mode=latest \
      "$@"
fi
