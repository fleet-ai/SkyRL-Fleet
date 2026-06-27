#!/usr/bin/env bash
# Direct-slurm sbatch wrapper for the GSM8k GRPO baseline, pinned to node-9.
#
# Bypasses SkyPilot entirely: node-9 is a bare-metal node OUTSIDE the SkyPilot
# managed pool (SkyPilot's generated sbatch scripts --exclude node-8,9,10), so
# we submit directly with --nodelist=node-9. We must supply, by hand, the env
# shims that SkyPilot's YAML setup/run blocks normally inject, and build the
# uv venv ourselves (no enroot container here).
#
# See fleet-common-run.sh: it references MODALITY unconditionally and reads the
# Ray head IP from SKYPILOT_NODE_IPS (first whitespace-separated token). That IP
# MUST be node-9's REAL routable IP, NOT the docker-internal 172.19.x.x that
# `hostname -I` lists first, or Ray's head registers as a ghost node and the
# GPU placement group can't schedule.

#SBATCH --job-name=neeraj-rewind-gsm8k-baseline
#SBATCH --nodelist=node-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --no-requeue
#SBATCH --output=/workspace/neeraj/logs/neeraj-rewind-gsm8k-baseline-%j.out

set -euo pipefail

mkdir -p /workspace/neeraj/logs

cd /workspace/neeraj/skyrl-fleet

# --- Credentials (sourced; cluster is single-owner/all-root, 600 on NFS) ---
source /workspace/neeraj/.secrets/wandb.env
source /workspace/neeraj/.secrets/research-jobs.env

# --- Standalone (non-SkyPilot) env shims required by fleet-common-run.sh ---
export MODALITY=gsm8k                 # dummy: referenced unconditionally; unused for gsm8k env-class
export SKYPILOT_NUM_GPUS_PER_NODE=8
export SKYPILOT_NUM_NODES=1
export SKYPILOT_NODE_RANK=0

# Ray head IP: pick node-9's routable 10.66.x interface, NOT docker-internal
# 172.19.x.x (which `hostname -I` lists first) and NOT loopback. Fall back to
# the known-good ens1 address if detection turns up nothing.
NODE_IP="$(hostname -I | tr ' ' '\n' | grep -E '^10\.66\.' | head -n1 || true)"
if [ -z "${NODE_IP}" ]; then
  NODE_IP="10.66.0.11"
fi
export SKYPILOT_NODE_IPS="${NODE_IP}"
echo "=== sbatch wrapper: SKYPILOT_NODE_IPS=${SKYPILOT_NODE_IPS} (Ray head IP) ==="

# --- Build the venv (normally done in SkyPilot YAML setup block) ---
if [ ! -d .venv ]; then
  uv venv --python 3.12 --seed
fi
source .venv/bin/activate
uv sync --extra fsdp
uv pip install wandb
# Standalone (non-SkyPilot) needs Ray's dashboard/metrics deps: pyproject pins
# bare `ray==2.51.1` (no [default] extra), so `uv sync --extra fsdp` omits
# aiohttp_cors / opencensus. Without them the Ray metrics agent never becomes
# ready; the vLLM engine actors then block 30s in WaitForServerReadyWithRetry,
# get disconnected (IOError: Broken pipe), and all 8 die mid-init with
# "Engine core initialization failed". Install ray[default] at the SAME pinned
# version so only the missing dashboard sub-deps are added.
uv pip install "ray[default]==2.51.1"

# --- Prepare data + launch GRPO training ---
bash scripts/fleet-gsm8k-baseline-run.sh
