#!/usr/bin/env bash
# Fleet shared setup: env validation, venv, dependencies, OpenEnv, dataset download
#
# Usage (from SkyPilot YAML setup block):
#   bash skyrl-train/scripts/fleet-common-setup.sh \
#     --openenv-branch deniz/fleet_client \
#     --extra-setup skyrl-train/scripts/fleet-qwen35-extra-setup.sh
#
# Required env vars: FLEET_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#   MODALITY, DATA_VERSION, S3_DATASET_BUCKET
# Optional env vars: ENV_KEYS, DIFFICULTY
set -euo pipefail

# Defaults
OPENENV_BRANCH="deniz/fleet_client"
EXTRA_SETUP=""
DATA_ROOT=""
SKIP_UV_ISOLATED=false
EXTRA_PIP=""
SKIP_PREPARE=false
ENV_CLASS="fleet_task"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --openenv-branch) OPENENV_BRANCH="$2"; shift 2 ;;
    --extra-setup) EXTRA_SETUP="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --skip-uv-isolated) SKIP_UV_ISOLATED=true; shift ;;
    --extra-pip) EXTRA_PIP="$2"; shift 2 ;;
    --skip-prepare) SKIP_PREPARE=true; shift ;;
    --env-class) ENV_CLASS="$2"; shift 2 ;;
    *) echo "ERROR: Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-detect data root: /workspace if writable (RunPod), else $HOME.
# On shared /workspace (Slurm), scope by sky cluster name so concurrent runs
# with the same MODALITY don't stomp on each other's prepared parquet files.
if [ -z "$DATA_ROOT" ]; then
  if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    SCRIPT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CLUSTER_ID="$(echo "$SCRIPT_ABS" | sed -n 's|.*/.sky_clusters/\([^/]*\)/.*|\1|p')"
    if [ -n "$CLUSTER_ID" ]; then
      DATA_ROOT="/workspace/clusters/${CLUSTER_ID}"
    else
      DATA_ROOT="/workspace"
    fi
  else
    DATA_ROOT="$HOME"
  fi
fi

# Resolve extra-setup path to absolute before cd (it's relative to repo root)
if [ -n "$EXTRA_SETUP" ]; then
  EXTRA_SETUP="$(cd "$(dirname "$EXTRA_SETUP")" && pwd)/$(basename "$EXTRA_SETUP")"
fi

# In upstream SkyRL, training packages live at repo root (skyrl/, skyrl-gym/, integrations/)
# No need to cd into skyrl-train/ — the venv and dependencies are at root level

echo "=== Fleet Common Setup ==="
echo "OpenEnv branch: $OPENENV_BRANCH"
echo "Data root: $DATA_ROOT"
echo "Extra setup: ${EXTRA_SETUP:-none}"

# --- Environment validation ---
echo "Validating environment variables..."
if [ -z "${FLEET_API_KEY:-}" ]; then
  echo "ERROR: FLEET_API_KEY is required"; exit 1
fi
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for S3 dataset download"; exit 1
fi
if [ "${MODALITY:-}" != "tool_use" ] && [ "${MODALITY:-}" != "computer_use" ] && [ "${MODALITY:-}" != "browser_use" ]; then
  echo "ERROR: MODALITY must be 'tool_use', 'computer_use', or 'browser_use', got: ${MODALITY:-unset}"; exit 1
fi
echo "Environment validation passed"

# --- Shared-filesystem guard ---
# On Slurm with shared /workspace, only one node should install packages.
# SLURM_PROCID=0 is the first task (head node); others wait for a sentinel file.
SETUP_SENTINEL=".venv/.setup_complete"
IS_INSTALLER=true
if [ -n "${SLURM_PROCID:-}" ] && [ "$SLURM_PROCID" != "0" ]; then
  IS_INSTALLER=false
  echo "Slurm worker (SLURM_PROCID=$SLURM_PROCID): waiting for node-0 to finish install..."
  while [ ! -f "$SETUP_SENTINEL" ]; do sleep 5; done
  echo "Sentinel found, activating shared venv"
  source .venv/bin/activate
  echo "=== Fleet Common Setup Complete (worker) ==="
  return 0 2>/dev/null || exit 0
fi

echo "Slurm installer (SLURM_PROCID=${SLURM_PROCID:-unset}): installing packages..."
# Remove stale sentinel from previous runs
rm -f "$SETUP_SENTINEL"

# --- Fix Ray binary permissions (some cloud images strip +x) ---
for f in .venv/bin/ray .venv/lib/python*/site-packages/ray/core/src/ray/raylet/raylet; do
  [ -f "$f" ] && chmod +x "$f" 2>/dev/null || true
done

# --- System dependencies (GCP images may lack build tools) ---
if ! command -v c++ &>/dev/null; then
  echo "Installing build-essential (c++ compiler required for causal-conv1d)..."
  sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends build-essential
fi

# --- Python environment ---
if [ -d ".venv" ]; then
  echo "Virtual environment already exists, reusing"
else
  uv venv --python 3.12 --seed
fi
source .venv/bin/activate
# vLLM 0.17.0 has native Qwen3.5 support (GDN via torch.ops.vllm.gdn_attention_core),
# FlashAttention 4, and PyTorch 2.10.0
uv sync --extra fsdp
uv pip install wandb boto3 awscli
# Pin fleet-python<=0.2.119: 0.2.120+ has async BaseWrapper bug (missing jwt/team_id params)
uv pip install "litellm>=1.75.5" "fleet-python<=0.2.119" logfire "mcp>=1.0.0"

# --- ATOF: nemo-relay wheel (rollout observability, enabled by default) ---
echo "Installing nemo-relay wheel for ATOF event emission..."
NEMO_WHEEL_DIR="$(mktemp -d)"
if aws s3 cp --recursive s3://fleet-nemo-relay-artifacts/wheels/latest/ "$NEMO_WHEEL_DIR/" \
  && uv pip install \
    "$NEMO_WHEEL_DIR"/nemo_relay-*.whl \
    "$NEMO_WHEEL_DIR"/nemo_relay_runtime-*.whl; then
  echo "nemo-relay and nemo-relay-runtime installed."
else
  echo "WARNING: NeMo wheel install failed; ATOF will be disabled (fail-open)." >&2
fi
rm -rf "$NEMO_WHEEL_DIR"

# --- Extra pip packages (installed before extra-setup to avoid dependency downgrades) ---
if [ -n "$EXTRA_PIP" ]; then
  echo "Installing extra pip packages: $EXTRA_PIP"
  uv pip install $EXTRA_PIP
fi

# --- Extra setup hook (model-specific dependencies) ---
if [ -n "$EXTRA_SETUP" ]; then
  echo "Running extra setup: $EXTRA_SETUP"
  source "$EXTRA_SETUP"
fi

# --- OpenEnv (force reinstall for latest changes) ---
uv pip install --force-reinstall --no-cache-dir --no-deps "git+https://github.com/fleet-ai/OpenEnv.git@${OPENENV_BRANCH}"

# --- Dataset download ---
mkdir -p "${DATA_ROOT}/data/fleet"
TASKS_FILE="${DATA_ROOT}/data/fleet/tasks_${MODALITY}.json"
S3_PATH="s3://${S3_DATASET_BUCKET}/${DATA_VERSION}/openenv/all_${MODALITY}.json"
echo "Downloading dataset from $S3_PATH..."
aws s3 cp "$S3_PATH" "$TASKS_FILE"
TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('$TASKS_FILE'))['tasks']))")
echo "Downloaded $TASK_COUNT tasks for modality: $MODALITY"

# --- Prepare dataset (parquet files) ---
if [ "$SKIP_PREPARE" = true ]; then
  echo "Skipping prepare_dataset (--skip-prepare). Caller handles preparation."
else
  DATA_DIR="${DATA_ROOT}/data/fleet/${MODALITY}"
  PREPARE_CMD="python -m integrations.fleet.prepare_dataset --tasks-json $TASKS_FILE --output-dir $DATA_DIR --modality $MODALITY --env-class $ENV_CLASS"
  [ -n "${ENV_KEYS:-}" ] && PREPARE_CMD="$PREPARE_CMD --env-filter $ENV_KEYS"
  [ -n "${DIFFICULTY:-}" ] && PREPARE_CMD="$PREPARE_CMD --difficulty-filter $DIFFICULTY"
  [ -n "${MAX_TASKS:-}" ] && PREPARE_CMD="$PREPARE_CMD --max-tasks $MAX_TASKS"
  eval "$PREPARE_CMD"
fi

# --- Pre-download model weights (head node only, shared NFS) ---
# Download once to /workspace/hf_cache so all nodes read from NFS at runtime
# instead of each node downloading independently (460 GB+ for large models).
if [ -n "${MODEL_PATH:-}" ]; then
  echo "Pre-downloading model: $MODEL_PATH"
  # huggingface_hub >= 1.0 removed `huggingface-cli` — it now prints
  # "deprecated and no longer works" and exits nonzero, which under `set -e`
  # killed setup outright (hit on all six exp1 arms, 2026-08-02). Prefer the new
  # `hf` CLI, fall back to the old one for older images.
  #
  # This is only a cache warm-up so all nodes read weights from shared NFS
  # instead of each pulling its own copy; training re-downloads on demand if it
  # is missing. So a failure here must never abort setup.
  if command -v hf >/dev/null 2>&1; then
    HF_HOME=/workspace/hf_cache hf download "$MODEL_PATH" >/dev/null \
      || echo "WARNING: hf download failed; nodes will fetch weights at runtime." >&2
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME=/workspace/hf_cache huggingface-cli download "$MODEL_PATH" --quiet \
      || echo "WARNING: huggingface-cli download failed; nodes will fetch weights at runtime." >&2
  else
    echo "WARNING: no hf CLI found; nodes will fetch weights at runtime." >&2
  fi
  # Make cache readable/writable by all users. Use || true because other
  # users' files in the shared cache may not be chown-able.
  chmod -R a+rwX /workspace/hf_cache 2>/dev/null || true
  echo "Model cached at /workspace/hf_cache"
fi

# --- Signal workers that install is done ---
touch "$SETUP_SENTINEL"
echo "=== Fleet Common Setup Complete ==="
