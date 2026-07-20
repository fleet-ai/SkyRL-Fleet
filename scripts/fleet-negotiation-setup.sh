#!/usr/bin/env bash
# Setup for the negotiation RLVR environment (text-only, multi-node aware).
#
# Unlike scripts/fleet-common-setup.sh, the negotiation job needs NO Fleet API
# key, NO AWS credentials, and NO S3 dataset download:
#   - the opponent ("them") side is an OpenRouter LLM (litellm), and
#   - the training dataset is generated locally by the run script
#     (skyrl-gym/skyrl_gym/envs/negotiation/prepare_dataset.py).
# So this installs only the venv + deps and pre-downloads the model weights.
#
# Required env vars: WANDB_API_KEY, OPENROUTER_API_KEY
# Optional: MODEL_PATH (pre-download), AWS_* (only if you want S3 checkpoints)
set -euo pipefail

OPENENV_BRANCH=""   # accepted for parity with fleet-common-setup; unused (no Fleet env)
EXTRA_SETUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --openenv-branch) OPENENV_BRANCH="$2"; shift 2 ;;
    --extra-setup) EXTRA_SETUP="$2"; shift 2 ;;
    *) echo "ERROR: Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Environment validation (no FLEET/AWS required for negotiation) ---
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required (powers the opponent LLM)}"

# Resolve extra-setup path to absolute before any cd (it's relative to repo root)
if [ -n "$EXTRA_SETUP" ]; then
  EXTRA_SETUP="$(cd "$(dirname "$EXTRA_SETUP")" && pwd)/$(basename "$EXTRA_SETUP")"
fi

echo "=== Negotiation Setup ==="
echo "Extra setup: ${EXTRA_SETUP:-none}"

# --- Shared-filesystem guard (Slurm multi-node) ---
# On Slurm with shared /workspace, only node-0 installs packages; other nodes
# wait for a sentinel file, then activate the shared venv. Mirrors
# fleet-common-setup.sh so the 2-node launch behaves identically.
SETUP_SENTINEL=".venv/.setup_complete"
if [ -n "${SLURM_PROCID:-}" ] && [ "$SLURM_PROCID" != "0" ]; then
  echo "Slurm worker (SLURM_PROCID=$SLURM_PROCID): waiting for node-0 to finish install..."
  while [ ! -f "$SETUP_SENTINEL" ]; do sleep 5; done
  echo "Sentinel found, activating shared venv"
  source .venv/bin/activate
  echo "=== Negotiation Setup Complete (worker) ==="
  return 0 2>/dev/null || exit 0
fi

echo "Installer (SLURM_PROCID=${SLURM_PROCID:-unset}): installing packages..."
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
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
  echo "Stale/broken .venv found (no bin/activate) — removing and recreating"
  rm -rf .venv
fi
if [ -d ".venv" ]; then
  echo "Virtual environment already exists, reusing"
else
  uv venv --python 3.12 --seed
fi
source .venv/bin/activate
# vLLM 0.17.0 (native Qwen3.5/GDN support) + FlashAttention + PyTorch 2.10 come
# from the fsdp extra, matching the 35B/9B Fleet jobs.
# Megatron: transformer-engine builds from source during uv sync and needs CUDA dev headers.
if [ "${TRAIN_EXTRA:-fsdp}" = "megatron" ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="$CUDA_HOME/bin:$PATH"
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include:/usr/include:${CPATH:-}"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export NVTE_CUDA_ARCHS="90"
  export MAX_JOBS="${MAX_JOBS:-64}"
  echo "[megatron] CUDA build env: CUDA_HOME=$CUDA_HOME NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
fi
uv sync --extra "${TRAIN_EXTRA:-fsdp}"
# wandb: logging. boto3/awscli: optional S3 checkpoint upload. litellm: opponent LLM.
uv pip install wandb boto3 awscli
uv pip install "litellm>=1.75.5"
# datasets: required by skyrl-gym negotiation prepare_dataset.py (head-node dataset gen).
uv pip install datasets

# --- ATOF: nemo-relay wheel (rollout observability, enabled by default) ---
export SKYRL_ATOF_ENABLED=1
echo "Installing nemo-relay wheel for ATOF event emission..."
NEMO_WHEEL_DIR="$(mktemp -d)"
if aws s3 cp --recursive s3://fleet-nemo-relay-artifacts/wheels/latest/ "$NEMO_WHEEL_DIR/" \
  && uv pip install "$NEMO_WHEEL_DIR"/nemo_relay-*.whl; then
  echo "nemo-relay installed."
else
  echo "WARNING: nemo-relay wheel install failed; ATOF will be disabled (fail-open)." >&2
fi
rm -rf "$NEMO_WHEEL_DIR"

# --- Extra setup hook (Qwen3.5-specific deps: transformers 5.3.0, causal-conv1d,
# CUDA toolkit, and ~/.cuda_env which the run script sources via --cuda-env) ---
if [ -n "$EXTRA_SETUP" ]; then
  echo "Running extra setup: $EXTRA_SETUP"
  source "$EXTRA_SETUP"
fi

# --- Pre-download model weights (head node only, shared NFS) ---
if [ -n "${MODEL_PATH:-}" ]; then
  echo "Pre-downloading model: $MODEL_PATH"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME=/workspace/hf_cache hf download "$MODEL_PATH" --quiet \
      || echo "WARN: model pre-download failed; nodes will fetch at runtime"
  else
    HF_HOME=/workspace/hf_cache huggingface-cli download "$MODEL_PATH" --quiet \
      || echo "WARN: model pre-download failed; nodes will fetch at runtime"
  fi
  chmod -R a+rwX /workspace/hf_cache 2>/dev/null || true
fi

# --- Signal workers that install is done ---
touch "$SETUP_SENTINEL"
echo "=== Negotiation Setup Complete ==="
