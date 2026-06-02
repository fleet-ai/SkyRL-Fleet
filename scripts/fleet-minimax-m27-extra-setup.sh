#!/usr/bin/env bash
# MiniMax-M2.7-specific dependencies (sourced by fleet-common-setup.sh via --extra-setup)
#
# MiniMax M2.7: 230B total / 10B active MoE, 256 experts, 8 active per token.
# Different architecture from Qwen3.5: no GatedDeltaNet, no causal-conv1d needed.
#
# Installs: transformers (MiniMax-compatible), flash-attn, CUDA toolkit
# Writes: $HOME/.cuda_env (sourced at run time)

# --- transformers ---
# MiniMax M2.7 uses trust_remote_code=True with custom modeling code.
# The base SkyRL install provides transformers; upgrade if needed for MiniMax compat.
# MiniMax M2 series requires transformers >= 4.45.0.
TRANSFORMERS_VER=$(python -c "import transformers; print(transformers.__version__)")
echo "transformers version: $TRANSFORMERS_VER"
# M2.7 HF card lists transformers as supported; verify the model loads.
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('MiniMaxAI/MiniMax-M2.7', trust_remote_code=True)
print(f'MiniMax M2.7 config loaded: model_type={cfg.model_type}, '
      f'num_experts={getattr(cfg, \"num_local_experts\", \"?\")}, '
      f'vocab_size={cfg.vocab_size}')
" || {
  echo "ERROR: MiniMax M2.7 config failed to load. May need transformers upgrade."
  exit 1
}

# --- flash-attn ---
# MiniMax uses standard multi-head attention (not GatedDeltaNet).
# flash-attn 2.8.3 prebuilt wheel for torch 2.10 + CUDA 12 (same as Qwen setup).
uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

python -c "import torch; print(f'torch={torch.__version__}')"

# --- CUDA toolkit (for any JIT kernels) ---
CUDA_HOME=""
for d in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
  if [ -x "$d/bin/nvcc" ]; then
    CUDA_HOME="$d"
    break
  fi
done
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
  NVCC_PATH=$(command -v nvcc)
  CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
fi
if [ -z "$CUDA_HOME" ]; then
  echo "nvcc not found on system. Installing CUDA toolkit from NVIDIA apt repo..."
  sudo apt-get update -qq
  UBUNTU_VER=$(lsb_release -rs 2>/dev/null | tr -d '.' || echo "2204")
  KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "Installing CUDA keyring from $KEYRING_URL"
  wget -qO /tmp/cuda-keyring.deb "$KEYRING_URL" 2>&1 || curl -sLo /tmp/cuda-keyring.deb "$KEYRING_URL"
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends cuda-nvcc-12-8 libcublas-dev-12-8 cuda-nvrtc-dev-12-8
  CUDA_HOME="/usr/local/cuda-12.8"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

# Write cuda_env for run phase (fleet-common-run.sh sources this via --cuda-env)
echo "export CUDA_HOME=$CUDA_HOME" > "$HOME/.cuda_env"
echo "export PATH=$CUDA_HOME/bin:\$PATH" >> "$HOME/.cuda_env"

# --- vLLM version check ---
# MiniMax M2.7 requires vLLM with commit cf3eacfe (day-0 M2 support).
# Check if current vLLM works; if not, log a warning.
python -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
# Try importing MiniMax model support
try:
    from vllm.model_executor.models import _MODELS
    has_minimax = any('minimax' in k.lower() or 'MiniMax' in k for k in _MODELS)
    if has_minimax:
        print('vLLM has MiniMax model support')
    else:
        print('WARNING: vLLM may not have MiniMax support. Consider upgrading to nightly.')
except Exception as e:
    print(f'Could not check vLLM model registry: {e}')
"

# --- Verify pinned packages survived ---
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "torch version after setup: $TORCH_VER"
if [[ "$TORCH_VER" != 2.10.0* ]]; then
  echo "WARNING: torch was downgraded to $TORCH_VER, reinstalling 2.10.0+cu128"
  pip install --force-reinstall --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; assert torch.__version__.startswith('2.10.0'), f'Expected 2.10.0 got {torch.__version__}'"
python -c "import torch; import flash_attn_2_cuda; print('flash_attn CUDA extension OK')"
