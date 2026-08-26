#!/usr/bin/env bash
# Runtime entrypoint for SkyRL-Fleet jobs on rl1.
#
# The image (integrations/fleet/rl1/Dockerfile) bakes every install from
# scripts/fleet-common-setup.sh; this script does only what needs runtime
# secrets or the pod's identity, then execs the job's command:
#   1. mirror stdout/stderr to $RUN_DIR/driver.log on the shared filesystem
#   2. single-node SKYPILOT_* shims (fleet-common-run.sh reads them)
#   3. nemo-relay (ATOF trace emission) wheel install — fail-open
#   4. dataset download from S3 + parquet prep (fleet-common-setup.sh tail)
#   5. exec the command
#
# Usage: bash integrations/fleet/rl1/entrypoint.sh '<command string>'
#   e.g. bash integrations/fleet/rl1/entrypoint.sh 'bash scripts/fleet-9b-run.sh'
#
# Required env: FLEET_API_KEY, WANDB_API_KEY, AWS_ACCESS_KEY_ID,
#   AWS_SECRET_ACCESS_KEY, MODALITY
# Optional env: RUN_DIR, DATA_VERSION (v7), S3_DATASET_BUCKET, ENV_KEYS,
#   DIFFICULTY, MAX_TASKS, SKYPILOT_NUM_GPUS_PER_NODE (default: probe)
set -euo pipefail

cd /opt/skyrl
source .venv/bin/activate

if [ -n "${RUN_DIR:-}" ]; then
  mkdir -p "$RUN_DIR"
  exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
fi

: "${FLEET_API_KEY:?}" ; : "${WANDB_API_KEY:?}"
: "${AWS_ACCESS_KEY_ID:?}" ; : "${AWS_SECRET_ACCESS_KEY:?}"
if [ "${MODALITY:-}" != "tool_use" ] && [ "${MODALITY:-}" != "computer_use" ] && [ "${MODALITY:-}" != "browser_use" ]; then
  echo "ERROR: MODALITY must be tool_use|computer_use|browser_use, got: ${MODALITY:-unset}"; exit 1
fi

# --- SKYPILOT shims (fleet-common-run.sh reads these) ---
# On rl1 this script runs as the RayJob driver on the head; KubeRay owns the
# Ray cluster (FLEET_EXTERNAL_RAY=1 in the manifest) and WORKERS is the GPU
# pod count. On a bare single pod, defaults probe the local GPUs.
export SKYPILOT_NODE_IPS="${SKYPILOT_NODE_IPS:-$(hostname -i | awk '{print $1}')}"
export SKYPILOT_NUM_GPUS_PER_NODE="${SKYPILOT_NUM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
export SKYPILOT_NUM_NODES="${SKYPILOT_NUM_NODES:-${WORKERS:-1}}"
export SKYPILOT_NODE_RANK="${SKYPILOT_NODE_RANK:-0}"
echo "driver ip=$SKYPILOT_NODE_IPS gpus_per_worker=$SKYPILOT_NUM_GPUS_PER_NODE workers=$SKYPILOT_NUM_NODES external_ray=${FLEET_EXTERNAL_RAY:-0}"

# --- ATOF: nemo-relay wheel (rollout observability; fail-open, same as
# fleet-common-setup.sh — the wheels live on S3 so this needs AWS creds) ---
NEMO_WHEEL_DIR="$(mktemp -d)"
if aws s3 cp --recursive s3://fleet-nemo-relay-artifacts/wheels/latest/ "$NEMO_WHEEL_DIR/" \
  && uv pip install \
    "$NEMO_WHEEL_DIR"/nemo_relay-*"$(uname -m)".whl \
    "$NEMO_WHEEL_DIR"/nemo_relay_runtime-*.whl; then
  echo "nemo-relay installed (ATOF on)"
else
  echo "WARNING: nemo-relay install failed; ATOF disabled (fail-open)" >&2
fi
rm -rf "$NEMO_WHEEL_DIR"

# --- dataset download + parquet prep (fleet-common-setup.sh tail) ---
export DATA_VERSION="${DATA_VERSION:-v7}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
DATA_ROOT="$HOME"
TASKS_FILE="${DATA_ROOT}/data/fleet/tasks_${MODALITY}.json"
DATA_DIR="${DATA_ROOT}/data/fleet/${MODALITY}"
mkdir -p "${DATA_ROOT}/data/fleet"
S3_PATH="s3://${S3_DATASET_BUCKET}/${DATA_VERSION}/openenv/all_${MODALITY}.json"
echo "Downloading dataset from $S3_PATH..."
aws s3 cp "$S3_PATH" "$TASKS_FILE"
python3 -c "import json; print('tasks:', len(json.load(open('$TASKS_FILE'))['tasks']))"

PREPARE_CMD=(python -m integrations.fleet.prepare_dataset
  --tasks-json "$TASKS_FILE" --output-dir "$DATA_DIR"
  --modality "$MODALITY" --env-class fleet_task)
[ -n "${ENV_KEYS:-}" ] && PREPARE_CMD+=(--env-filter "$ENV_KEYS")
[ -n "${DIFFICULTY:-}" ] && PREPARE_CMD+=(--difficulty-filter "$DIFFICULTY")
[ -n "${MAX_TASKS:-}" ] && PREPARE_CMD+=(--max-tasks "$MAX_TASKS")
"${PREPARE_CMD[@]}"

echo "=== exec: $1 ==="
exec bash -c "$1"
