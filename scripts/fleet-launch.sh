#!/usr/bin/env bash
# Preflight-gated wrapper around `sky launch`.
#
# Use this in place of `sky launch` for any SkyRL-Fleet job. The wrapper
# runs scripts/fleet-preflight.sh first; if every check passes it then
# invokes `sky launch` with the supplied arguments. If any check fails,
# `sky launch` is never called — so we don't burn provisioning time on a
# job that's guaranteed to die in setup.
#
# Default behavior: appends `--down` to the sky launch invocation so the
# cluster is torn down automatically when the job finishes (regardless of
# whether it SUCCEEDED or FAILED). This prevents zombie clusters from
# holding GPU allocations after training exits. `--down` only triggers on
# job exit, never mid-training, so it cannot prematurely kill a live run.
#
# To preserve the cluster after job exit (e.g. for `sky exec` follow-up
# jobs or interactive debugging), pass `--keep-cluster` as a preflight-side
# flag (i.e. before the literal `--`). Passing `--down` yourself in the
# sky-launch args is a no-op — the wrapper detects it and won't double up.
#
# Argument convention:
#
#   bash scripts/fleet-launch.sh [sky-launch args...]
#       — no `--`: every token is forwarded to `sky launch`. Preflight runs
#         with default settings (FLEET / WANDB / AWS required, gcloud + sky
#         + liveness all enabled).
#
#   bash scripts/fleet-launch.sh [preflight args...] -- [sky-launch args...]
#       — with `--`: tokens before `--` are forwarded verbatim to
#         scripts/fleet-preflight.sh (e.g. `--require OPENROUTER_API_KEY`,
#         `--skip-gcloud`, `--skip-liveness`). Tokens after `--` go to
#         `sky launch`. The `--keep-cluster` flag is consumed by this
#         wrapper and is NOT forwarded to preflight.
#
# Examples:
#
#   # Standard launch — auto-teardown on exit.
#   bash scripts/fleet-launch.sh tasks/openenv-fleet-grpo-vl.yaml \
#     --env FLEET_API_KEY="$FLEET_API_KEY" \
#     --env WANDB_API_KEY="$WANDB_API_KEY"
#
#   # Keep cluster up after job exit (for sky exec follow-ups).
#   bash scripts/fleet-launch.sh --keep-cluster -- tasks/foo.yaml --env ...
#
#   # Task-gen launch — also require OPENROUTER_API_KEY.
#   bash scripts/fleet-launch.sh --require OPENROUTER_API_KEY -- \
#     tasks/task-gen-grpo-qwen3_5-9b.yaml --env FLEET_API_KEY=... ...
#
#   # Non-GCP launch — skip the gcloud check.
#   bash scripts/fleet-launch.sh --skip-gcloud -- tasks/foo.yaml --env ...
#
# Environment:
#   SKIP_PREFLIGHT=1   Bypass the preflight entirely (escape hatch; avoid
#                      using this in scripts — it defeats the safety net).
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

PREFLIGHT_ARGS=()
SKY_ARGS=()
SAW_DASHDASH=false
KEEP_CLUSTER=false

# Split argv on a literal `--`. Without `--`, every token goes to sky.
# `--keep-cluster` is a wrapper-only flag stripped before forwarding to either
# preflight or sky launch.
for arg in "$@"; do
  if [ "$arg" = "--keep-cluster" ]; then
    KEEP_CLUSTER=true
    continue
  fi
  if [ "$SAW_DASHDASH" = true ]; then
    SKY_ARGS+=("$arg")
  elif [ "$arg" = "--" ]; then
    SAW_DASHDASH=true
  else
    PREFLIGHT_ARGS+=("$arg")
  fi
done
if [ "$SAW_DASHDASH" = false ]; then
  SKY_ARGS=(${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"})
  PREFLIGHT_ARGS=()
fi

# Auto-teardown by default: append `--down` so the cluster is released when
# the job exits (SUCCEEDED or FAILED). Skip if the user already passed
# `--down` themselves (sky errors on duplicate flags) or opted out via
# `--keep-cluster`.
if [ "$KEEP_CLUSTER" = false ]; then
  ALREADY_HAS_DOWN=false
  for arg in ${SKY_ARGS[@]+"${SKY_ARGS[@]}"}; do
    if [ "$arg" = "--down" ]; then
      ALREADY_HAS_DOWN=true
      break
    fi
  done
  if [ "$ALREADY_HAS_DOWN" = false ]; then
    SKY_ARGS+=("--down")
  fi
fi

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "WARNING: SKIP_PREFLIGHT=1 — bypassing fleet-preflight.sh" >&2
else
  # Empty-array safe expansion (works under `set -u`).
  bash scripts/fleet-preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}
fi

if [ ${#SKY_ARGS[@]} -eq 0 ]; then
  echo "ERROR: no arguments to forward to 'sky launch'" >&2
  echo "Usage: bash scripts/fleet-launch.sh [preflight args...] -- <yaml> [sky launch args...]" >&2
  echo "   or: bash scripts/fleet-launch.sh <yaml> [sky launch args...]" >&2
  exit 1
fi

if ! command -v sky >/dev/null 2>&1; then
  echo "ERROR: sky CLI not found on PATH" >&2
  exit 1
fi

echo "=== Preflight passed; running: sky launch ${SKY_ARGS[*]} ==="
exec sky launch "${SKY_ARGS[@]}"
