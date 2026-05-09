#!/usr/bin/env bash
# Preflight-gated wrapper around `sky launch`.
#
# Use this in place of `sky launch` for any SkyRL-Fleet job. The wrapper
# runs scripts/fleet-preflight.sh first; if every check passes it then
# invokes `sky launch` with the original arguments. If any check fails,
# `sky launch` is never called — so we don't burn provisioning time on a
# job that's guaranteed to die in setup.
#
# Usage:
#   bash scripts/fleet-launch.sh tasks/openenv-fleet-grpo-vl.yaml \
#     --env FLEET_API_KEY="$FLEET_API_KEY" \
#     --env WANDB_API_KEY="$WANDB_API_KEY" \
#     --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
#     --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
#
# Forward preflight-only flags before a literal `--`:
#   bash scripts/fleet-launch.sh \
#     --preflight-arg --skip-gcloud \
#     --preflight-arg --require OPENROUTER_API_KEY \
#     -- tasks/task-gen-grpo-qwen3_5-9b.yaml --env ...
#
# Environment:
#   SKIP_PREFLIGHT=1   Bypass the preflight entirely (escape hatch; avoid
#                      using this in scripts — it defeats the safety net).
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

PREFLIGHT_ARGS=()
SKY_ARGS=()
SAW_DASHDASH=false

# Args before `--` that begin with `--preflight-arg <X>` are forwarded to
# fleet-preflight.sh; everything else (and everything after `--`) is
# forwarded to `sky launch`. The common case — no preflight tweaks — is
# `bash fleet-launch.sh <yaml> --env ...` with no `--`.
while [[ $# -gt 0 ]]; do
  if [ "$SAW_DASHDASH" = true ]; then
    SKY_ARGS+=("$1"); shift; continue
  fi
  case "$1" in
    --preflight-arg) PREFLIGHT_ARGS+=("$2"); shift 2 ;;
    --) SAW_DASHDASH=true; shift ;;
    *)  SKY_ARGS+=("$1"); shift ;;
  esac
done

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "WARNING: SKIP_PREFLIGHT=1 — bypassing fleet-preflight.sh" >&2
else
  # Empty-array safe expansion (works under `set -u`).
  bash scripts/fleet-preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}
fi

if [ ${#SKY_ARGS[@]} -eq 0 ]; then
  echo "ERROR: no arguments to forward to 'sky launch'" >&2
  echo "Usage: bash scripts/fleet-launch.sh <yaml> [sky launch args...]" >&2
  exit 1
fi

if ! command -v sky >/dev/null 2>&1; then
  echo "ERROR: sky CLI not found on PATH" >&2
  exit 1
fi

echo "=== Preflight passed; running: sky launch ${SKY_ARGS[*]} ==="
exec sky launch "${SKY_ARGS[@]}"
