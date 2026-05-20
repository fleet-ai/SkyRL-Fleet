#!/usr/bin/env bash
# Preflight-gated wrapper around `sky launch`.
#
# Use this in place of `sky launch` for any SkyRL-Fleet job. The wrapper
# runs scripts/fleet-preflight.sh first; if every check passes it then
# invokes `sky launch` with the supplied arguments. If any check fails,
# `sky launch` is never called — so we don't burn provisioning time on a
# job that's guaranteed to die in setup.
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
#         `sky launch`.
#
# Examples:
#
#   # Standard launch — uses default preflight settings.
#   bash scripts/fleet-launch.sh tasks/openenv-fleet-grpo-vl.yaml \
#     --env FLEET_API_KEY="$FLEET_API_KEY" \
#     --env WANDB_API_KEY="$WANDB_API_KEY" \
#     --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
#     --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
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
#   FLEET_RAY_PORT                 Ray head port; defaults to 6379.
#   FLEET_RAY_DASHBOARD_PORT       Ray dashboard port; defaults to 8265.
#
# Task YAMLs may use __FLEET_RAY_PORT__ and __FLEET_RAY_DASHBOARD_PORT__
# placeholders in resources.ports; this wrapper renders them before launch.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

PREFLIGHT_ARGS=()
SKY_ARGS=()
SAW_DASHDASH=false

# Split argv on a literal `--`. Without `--`, every token goes to sky.
for arg in "$@"; do
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

is_port() {
  [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

sky_env_is_set() {
  local key=$1
  local i arg next
  for i in "${!SKY_ARGS[@]}"; do
    arg="${SKY_ARGS[$i]}"
    if [ "$arg" = "--env" ] && [ $((i + 1)) -lt ${#SKY_ARGS[@]} ]; then
      next="${SKY_ARGS[$((i + 1))]}"
      if [[ "$next" == "$key="* ]]; then
        return 0
      fi
    elif [[ "$arg" == "--env=$key="* ]]; then
      return 0
    fi
  done
  return 1
}

get_sky_env() {
  local key=$1
  local default=$2
  local value="$default"
  local i arg next

  if [ -n "${!key+x}" ]; then
    value="${!key}"
  fi
  for i in "${!SKY_ARGS[@]}"; do
    arg="${SKY_ARGS[$i]}"
    if [ "$arg" = "--env" ] && [ $((i + 1)) -lt ${#SKY_ARGS[@]} ]; then
      next="${SKY_ARGS[$((i + 1))]}"
      if [[ "$next" == "$key="* ]]; then
        value="${next#*=}"
      fi
    elif [[ "$arg" == "--env=$key="* ]]; then
      value="${arg#--env=$key=}"
    fi
  done
  printf '%s\n' "$value"
}

find_task_yaml_index() {
  local i arg
  for i in "${!SKY_ARGS[@]}"; do
    arg="${SKY_ARGS[$i]}"
    if [[ "$arg" == *.yaml || "$arg" == *.yml ]] && [ -f "$arg" ]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

task_needs_render() {
  local task_yaml=$1
  grep -q '__FLEET_RAY_PORT__\|__FLEET_RAY_DASHBOARD_PORT__' "$task_yaml"
}

render_task_template() {
  local task_yaml=$1
  local output_yaml=$2
  sed \
    -e "s/__FLEET_RAY_PORT__/$RAY_PORT/g" \
    -e "s/__FLEET_RAY_DASHBOARD_PORT__/$RAY_DASHBOARD_PORT/g" \
    "$task_yaml" > "$output_yaml"
}

TMP_SKY_YAML=""
cleanup_tmp_yaml() {
  if [ -n "$TMP_SKY_YAML" ]; then
    rm -f "$TMP_SKY_YAML"
  fi
}
trap cleanup_tmp_yaml EXIT

RAY_PORT="$(get_sky_env FLEET_RAY_PORT 6379)"
RAY_DASHBOARD_PORT="$(get_sky_env FLEET_RAY_DASHBOARD_PORT 8265)"
if ! is_port "$RAY_PORT"; then
  echo "ERROR: FLEET_RAY_PORT must be a valid TCP port (got: $RAY_PORT)" >&2
  exit 1
fi
if ! is_port "$RAY_DASHBOARD_PORT"; then
  echo "ERROR: FLEET_RAY_DASHBOARD_PORT must be a valid TCP port (got: $RAY_DASHBOARD_PORT)" >&2
  exit 1
fi
if [ "$RAY_PORT" = "$RAY_DASHBOARD_PORT" ]; then
  echo "ERROR: FLEET_RAY_PORT and FLEET_RAY_DASHBOARD_PORT must be different ports" >&2
  exit 1
fi

if [ -n "${FLEET_RAY_PORT+x}" ] && ! sky_env_is_set FLEET_RAY_PORT; then
  SKY_ARGS+=(--env "FLEET_RAY_PORT=$RAY_PORT")
fi
if [ -n "${FLEET_RAY_DASHBOARD_PORT+x}" ] && ! sky_env_is_set FLEET_RAY_DASHBOARD_PORT; then
  SKY_ARGS+=(--env "FLEET_RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT")
fi

if TASK_YAML_INDEX="$(find_task_yaml_index)"; then
  TASK_YAML="${SKY_ARGS[$TASK_YAML_INDEX]}"
  if task_needs_render "$TASK_YAML"; then
    TMP_SKY_YAML="$(mktemp "./.fleet-launch.XXXXXX")"
    render_task_template "$TASK_YAML" "$TMP_SKY_YAML"
    SKY_ARGS[$TASK_YAML_INDEX]="$TMP_SKY_YAML"
    echo "=== Rendered Ray ports: [$RAY_PORT, $RAY_DASHBOARD_PORT] ==="
  fi
fi

echo "=== Preflight passed; running: sky launch ${SKY_ARGS[*]} ==="
sky launch "${SKY_ARGS[@]}"
