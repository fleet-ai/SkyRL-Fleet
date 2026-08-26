# Trainer image for SkyRL-Fleet jobs on rl1.
#
# Bakes what scripts/fleet-common-setup.sh (+ fleet-qwen35-extra-setup.sh)
# installs on a fresh SkyPilot VM, so a job is just: this image + a command
# (e.g. `bash scripts/fleet-9b-run.sh`). Runtime-only steps that need
# secrets (S3 dataset download, parquet prep) live in
# integrations/fleet/rl1/entrypoint.sh.
#
# Build from the REPO ROOT: docker build -f integrations/fleet/rl1/Dockerfile .

# Base matches docker/Dockerfile and uv.lock (ray 2.51.1, py312, cu128).
FROM anyscale/ray:2.51.1-slim-py312-cu128

RUN sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
        git wget curl build-essential ninja-build \
        kmod libxml2 libnuma-dev \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Full CUDA 12.8 toolkit: nvcc + libcu++ headers for FlashInfer/TileLang JIT
# (Qwen3.5 GDN kernels compile at train time) and the causal-conv1d source
# build below. CUDA 12.8 is the first toolkit with Blackwell (sm_100) support,
# which the B200 pool needs. Early layer on purpose: ~5GB that must never
# rebuild on code edits.
RUN cd /tmp \
    && curl -sSfLo cuda.run https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run \
    && sudo sh cuda.run --silent --toolkit --override \
    && rm cuda.run
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Same uv the repo's docker/Dockerfile pins.
RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh
ENV PATH="/home/ray/.local/bin:${PATH}"

RUN sudo mkdir -p /opt/skyrl && sudo chown ray:users /opt/skyrl
WORKDIR /opt/skyrl

# Layer split: resolve dependencies against the manifests alone so code-only
# changes reuse the multi-GB dependency layer. skyrl-gym rides along because
# it is an editable path source in [tool.uv.sources] — the sync needs it.
COPY --chown=ray:users pyproject.toml uv.lock /opt/skyrl/
COPY --chown=ray:users skyrl-gym /opt/skyrl/skyrl-gym
# UV_NO_CACHE: the wheel cache would otherwise live inside this layer and
# roughly double it.
ENV UV_NO_CACHE=1
RUN uv venv --python 3.12 --seed \
    && uv sync --frozen --extra fsdp --no-install-project

# The rest of fleet-common-setup.sh's installs, in its order.
RUN uv pip install wandb boto3 awscli \
    && uv pip install "litellm>=1.75.5" "fleet-python<=0.2.119" logfire "mcp>=1.0.0"

# fleet-qwen35-extra-setup.sh, baked:
# transformers 5.3.0 (first stable release with qwen3_5_moe; the project pin
# <5 predates Qwen3.5 — the setup script force-upgrades post-sync, so do we).
RUN uv pip install -U "transformers==5.3.0"

# causal-conv1d: GatedDeltaNet fast CUDA kernels. Without it fla-core falls
# back to a naive PyTorch path that crashes with cudaErrorIllegalAddress
# under FSDP2. Source build (needs nvcc + torch): the builder node has no
# GPU, so pin the arch list to the cluster's pools — 9.0 (H200) and 10.0
# (B200). MAX_JOBS caps ninja parallelism to fit the builder's memory.
RUN TORCH_CUDA_ARCH_LIST="9.0;10.0" MAX_JOBS=4 \
    .venv/bin/pip install --no-cache-dir --no-build-isolation "causal-conv1d>=1.6.0"

# OpenEnv (Fleet client branch), same force-reinstall as the setup script.
# Pinned to a commit, not the branch: a branch ref both drifts and gets
# frozen by Docker layer cache (the RUN line doesn't change when the branch
# moves). Bump via --build-arg OPENENV_REF=<sha>.
ARG OPENENV_REF=df97268a34dd2b9b56d4b3b394ecb1eb0059f2a5
RUN uv pip install --force-reinstall --no-cache-dir --no-deps \
    "git+https://github.com/fleet-ai/OpenEnv.git@${OPENENV_REF}"

# fleet-common-run.sh sources $HOME/.cuda_env at run time (FlashInfer JIT).
RUN printf 'export CUDA_HOME=%s\nexport PATH=%s/bin:$PATH\n' "$CUDA_HOME" "$CUDA_HOME" > /home/ray/.cuda_env

# Full source lands after the dependency layers; then install the project
# itself (deps already satisfied — --no-deps so nothing re-resolves and
# clobbers the pins above).
COPY --chown=ray:users . /opt/skyrl
RUN uv pip install --no-deps -e .

# Venv ahead of the base interpreter so `python`/`aws` resolve to the venv.
ENV PATH="/opt/skyrl/.venv/bin:${PATH}"

# Import smoke: `pip install completed` != `image works`.
RUN python -c "import integrations.fleet.entrypoints.main_fleet" \
    && python -c "import causal_conv1d; print('causal-conv1d', causal_conv1d.__version__)" \
    && python -c "import transformers; assert transformers.__version__=='5.3.0', transformers.__version__" \
    && python -c "import torch; assert torch.__version__.startswith('2.10.0'), torch.__version__" \
    && python -c "import torch; import flash_attn_2_cuda; print('flash_attn CUDA extension OK')" \
    && python -c "import vllm; print('vllm', vllm.__version__)" \
    && python -c "import envs.fleet_env" \
    && python -c "from skyrl_gym.envs.fleet_task.env import FleetTaskEnv" \
    && nvcc --version \
    && test -e /usr/local/cuda/include/cuda/atomic \
    && aws --version
