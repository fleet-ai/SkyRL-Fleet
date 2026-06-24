#!/usr/bin/env bash
# Row-only (policy-only) frontier cross-play matrix.
#
# Plays ONLY the policy's row + column of the cross-play matrix (policy vs each
# frontier model, both seats), skipping the frontier x frontier block. Single-tag
# protocol + per-model training-matched sampling, matching the selfplay-canask
# training rollout distribution. See CROSSPLAY_REPRO.md for the full rationale.
#
# Think gate is intentionally OFF (eval is unconstrained decoding) so behavior
# transfer is a real signal; do not re-enable forced <think> here.
#
# Usage:
#   OPENROUTER_API_KEY=...  ./crossplay_rowonly.sh base
#   OPENROUTER_API_KEY=...  S30_BASE_URL=http://10.66.0.6:6479/v1  ./crossplay_rowonly.sh s30
set -euo pipefail

WHICH="${1:-s30}"                       # base | s30
PY="${PY:-python}"
N="${N:-16}"                            # scenarios per cell
MAX_TURNS="${MAX_TURNS:-6}"
SEED="${SEED:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
SPLIT="${SPLIT:-val}"                   # held-out DnD split
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

common=(
  --policy-only
  --protocol single
  --elicit can_ask
  --match-train-sampling
  --dataset dnd --split "$SPLIT"
  --n "$N" --max-turns "$MAX_TURNS" --seed "$SEED"
  --concurrency "$CONCURRENCY"
)

case "$WHICH" in
  base)
    # Pre-RL base checkpoint served via OpenRouter.
    BASE_SLUG="${BASE_SLUG:-qwen/qwen3.5-35b-a3b}"
    "$PY" "$HERE/run_crossplay.py" \
      --policy-model "$BASE_SLUG" --policy-label Base-qwen35-35b \
      --policy-base-url https://openrouter.ai/api/v1 \
      --out-prefix crossplay_matrix_base_canask \
      "${common[@]}"
    ;;
  s30)
    # Self-Play canask checkpoint (global_step_30) served locally via vLLM.
    S30_BASE_URL="${S30_BASE_URL:-http://10.66.0.6:6479/v1}"
    S30_MODEL="${S30_MODEL:-qwen35-policy}"
    "$PY" "$HERE/run_crossplay.py" \
      --policy-model "$S30_MODEL" --policy-label SelfPlay-canask-s30 \
      --policy-base-url "$S30_BASE_URL" \
      --out-prefix crossplay_matrix_s30_canask \
      "${common[@]}"
    ;;
  *)
    echo "usage: $0 [base|s30]" >&2; exit 2 ;;
esac
