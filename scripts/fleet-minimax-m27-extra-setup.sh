#!/usr/bin/env bash
# MiniMax-M2.7-specific dependencies (sourced by fleet-common-setup.sh via --extra-setup)
#
# MiniMax M2.7: 230B total / 10B active MoE, 256 experts, 8 active per token.
# Requires vLLM >= 0.22 (day-0 M2.7 support, NVIDIA-optimized kernels).
# vLLM 0.22 requires torch 2.11 (upgraded from SkyRL default 2.10).
#
# Installs: vLLM 0.22, torch 2.11, flash-attn, CUDA toolkit
# Writes: $HOME/.cuda_env (sourced at run time)

set -euo pipefail

# --- Upgrade torch to 2.11 (required by vLLM 0.22) ---
echo "=== Upgrading torch to 2.11 for vLLM 0.22 ==="
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# torch 2.11 needs NCCL 2.28.9+ (ncclDevCommCreate API). --no-deps leaves
# the old nvidia-nccl-cu12==2.27.5 from torch 2.10 in place.
pip install "nvidia-nccl-cu12>=2.28.9"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')"
python -c "import torchvision; print(f'torchvision={torchvision.__version__}')"

# --- Upgrade vLLM to 0.22 (MiniMax M2.7 support) ---
echo "=== Upgrading vLLM to 0.22.0 ==="
# vLLM default PyPI wheel is cu130 (libcudart.so.13); RunPod has CUDA 12.8.
# cu129 wheel works on CUDA 12.8 (minor version backward compat).
VLLM_VERSION=0.22.0
CPU_ARCH=$(uname -m)
pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
# vLLM 0.22 upgrades flashinfer-python to 0.6.11 but the base venv's
# flashinfer-jit-cache stays at 0.6.6. Upgrade it from flashinfer's index.
pip install --force-reinstall flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129

# --- flash-attn (rebuild for torch 2.11) ---
# The prebuilt wheel is for torch 2.10; need to check compat or rebuild.
# vLLM 0.22 bundles FlashAttention 3 support, so explicit flash-attn may not be needed.
# Install if available, skip gracefully if wheel doesn't match.
echo "=== Installing flash-attn ==="
pip install flash-attn --no-build-isolation 2>&1 || echo "WARNING: flash-attn install failed, vLLM will use its bundled FA3"

# --- Verify MiniMax config loads ---
echo "=== Verifying MiniMax M2.7 config ==="
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('MiniMaxAI/MiniMax-M2.7', trust_remote_code=True)
print(f'MiniMax M2.7 config loaded: model_type={cfg.model_type}, '
      f'num_experts={getattr(cfg, \"num_local_experts\", \"?\")}, '
      f'vocab_size={cfg.vocab_size}')
" || {
  echo "ERROR: MiniMax M2.7 config failed to load."
  exit 1
}

# --- CUDA toolkit for JIT kernels (DeepGemm, FlashInfer) ---
# vLLM 0.22 cu129 installs nvidia-cuda-nvcc-cu12 (12.9) as a pip package.
# Use system CUDA 12.8 if available (driver 580 supports it). The pip nvcc
# provides the compiler; system CUDA provides nvrtc and other runtime libs.
CUDA_HOME=""
for d in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
  if [ -x "$d/bin/nvcc" ]; then
    CUDA_HOME="$d"
    break
  fi
done
if [ -z "$CUDA_HOME" ]; then
  echo "WARNING: No system CUDA toolkit found. DeepGemm JIT may fail."
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

# Write cuda_env for run phase
echo "export CUDA_HOME=$CUDA_HOME" > "$HOME/.cuda_env"
echo "export PATH=$CUDA_HOME/bin:\$PATH" >> "$HOME/.cuda_env"

# --- Verify versions ---
echo "=== Final version check ==="
python -c "
import torch, vllm
print(f'torch={torch.__version__}')
print(f'vLLM={vllm.__version__}')
assert torch.__version__.startswith('2.11'), f'Expected torch 2.11, got {torch.__version__}'
"

# --- vLLM MiniMax support check ---
python -c "
import vllm
print(f'vLLM {vllm.__version__} installed')
" || echo "WARNING: vLLM import failed"

echo "=== MiniMax M2.7 extra setup complete ==="
