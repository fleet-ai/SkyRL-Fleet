#!/usr/bin/env bash
# Self-hosted opponent ("them") endpoint for the negotiation RLVR environment.
#
# Serves the SAME Qwen3.5-35B-A3B model we train against as an OpenAI-compatible
# vLLM endpoint, so training drives the opponent side against our own GPUs instead
# of paying OpenRouter (GPT-4o-mini) per-call costs. This is the cost lever: a long
# GRPO run issues one opponent call per turn per rollout (MAX_TURNS * n_samples *
# batch * steps) — at GPT-4o-mini rates that adds up fast; on a reserved node it's free.
#
# Topology (the requested dp=8 / tp=1 / fp8 config): ONE dedicated node, 8x H200.
#   --data-parallel-size 8  -> 8 independent full-model replicas (one per GPU),
#                              maximizing concurrent-game throughput.
#   --tensor-parallel-size 1 -> no intra-replica sharding (dp*tp = 8 = whole node).
#   --quantization fp8       -> online FP8 weight quant (~halves weight memory and
#                              lifts decode throughput vs bf16). 35B-A3B is a 35B-total
#                              MoE; in FP8 the weights (~35 GB) fit on one H200 (141 GB),
#                              so TP=1 is sufficient and DP=8 buys 8x request concurrency.
#
# Run this on the node you dedicate to hosting (rank-0 / "node 0"); it binds all 8
# local GPUs. Then point training at it by setting, in the trainer env
# (see scripts/fleet-negotiation-35b-run.sh):
#   OPPONENT_MODEL=openai/qwen35-opponent
#   OPPONENT_BASE_URL=http://<this-node-ip>:6479/v1
#
# Required env vars: none (model defaults to Qwen/Qwen3.5-35B-A3B).
# To serve a trained export instead of the base model, set OPPONENT_SERVE_MODEL
# to the checkpoint dir (e.g. $HOME/exports/<run>/hf/global_step_xxx).
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root (scripts/ is directly under repo root)

# Model to host as the opponent. Defaults to the base 35B; override with a local
# export dir to host a trained checkpoint. Falls back to MODEL_PATH if exported by
# the launching task env.
export OPPONENT_SERVE_MODEL="${OPPONENT_SERVE_MODEL:-${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}}"
# Stable name litellm references as "openai/<SERVED_NAME>" — decoupled from the path
# so swapping the underlying checkpoint doesn't change the trainer's OPPONENT_MODEL.
export OPPONENT_SERVED_NAME="${OPPONENT_SERVED_NAME:-qwen35-opponent}"

export OPPONENT_SERVE_HOST="${OPPONENT_SERVE_HOST:-0.0.0.0}"
# 6479 is the port already exposed by the SkyPilot task YAMLs (RunPod/GKE/GCP).
export OPPONENT_SERVE_PORT="${OPPONENT_SERVE_PORT:-6479}"

export OPPONENT_DP_SIZE="${OPPONENT_DP_SIZE:-8}"
export OPPONENT_TP_SIZE="${OPPONENT_TP_SIZE:-1}"
export OPPONENT_QUANTIZATION="${OPPONENT_QUANTIZATION:-fp8}"
# Whole node by default (dp*tp = 8 GPUs).
export OPPONENT_GPUS="${OPPONENT_GPUS:-0,1,2,3,4,5,6,7}"

# Must cover the opponent's context: system prompt + full negotiation history +
# its own reply. Keep >= MAX_INPUT_LENGTH (8192) + opponent_max_tokens headroom.
export OPPONENT_MAX_MODEL_LEN="${OPPONENT_MAX_MODEL_LEN:-12288}"
export OPPONENT_GPU_MEM_UTIL="${OPPONENT_GPU_MEM_UTIL:-0.85}"
# Optional bearer token. Empty = no auth (vLLM accepts any/no key); the trainer
# sends a placeholder key in that case (see fleet-negotiation-35b-run.sh).
export OPPONENT_API_KEY="${OPPONENT_API_KEY:-}"

# Qwen3.5 GDN models can hang silently in the FlashInfer GDN JIT on GCP/RunPod
# (see fleet-negotiation-35b-run.sh); force the triton GDN prefill backend.
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"

source .venv/bin/activate

API_KEY_ARGS=()
[ -n "$OPPONENT_API_KEY" ] && API_KEY_ARGS=(--api-key "$OPPONENT_API_KEY")

set -x
CUDA_VISIBLE_DEVICES="$OPPONENT_GPUS" vllm serve "$OPPONENT_SERVE_MODEL" \
  --served-model-name "$OPPONENT_SERVED_NAME" \
  --host "$OPPONENT_SERVE_HOST" \
  --port "$OPPONENT_SERVE_PORT" \
  --data-parallel-size "$OPPONENT_DP_SIZE" \
  --tensor-parallel-size "$OPPONENT_TP_SIZE" \
  --quantization "$OPPONENT_QUANTIZATION" \
  --max-model-len "$OPPONENT_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$OPPONENT_GPU_MEM_UTIL" \
  --dtype auto \
  --trust-remote-code \
  --enable-prefix-caching \
  ${API_KEY_ARGS[@]+"${API_KEY_ARGS[@]}"} \
  "$@"
