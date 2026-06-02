#!/usr/bin/env bash
# Witness RUN — runs from the SkyPilot `run:` block (one per node).
#
# Delegates ALL generic cluster bring-up to scripts/fleet-common-run.sh (the team's
# shared runner): per-job Ray temp-dir + ports keyed on SLURM_JOB_ID, --node-ip-address
# from the overlay IP, the cross-node NCCL_IB_HCA intersection, per-UID HF/Triton caches,
# diagnostics, multi-node head/worker dispatch, and the trainer launch. This RETIRES the
# old inline hand-patches (Ray start, TMPDIR/AF_UNIX-107, df|head SIGPIPE, ghost docker-IP,
# NCCL/gIB) and inherits the team's current + future fixes for free.
#
# Witness keeps only what's witness-specific: reward/harness env, OpenRouter preflight,
# the witness Hydra overrides, and the rank-0 S3 ckpt mirror + 16-shard corruption assert
# + final HF-export→S3 (added after p2_r5_35b corrupted to 8/16 ranks; common-run has no
# S3 mirroring). Witness Hydra args are passed after `--` and WIN over common-run's baselines.
#
# Required env (from the YAML envs: block): WANDB_API_KEY, RUN_LABEL, MODEL knobs, reward flags.
set -euo pipefail
source .venv/bin/activate
[ -f "$HOME/.cuda_env" ] && source "$HOME/.cuda_env"

export ARC_WITNESS_AGENT_DIR=$HOME/arc-witness-agent
export WITNESS_ENVS_DIR=$HOME/arc-witness-envs
export HYDRA_FULL_ERROR=1
# fleet-common-run.sh references $MODALITY under `set -u` (only for an unread fleet
# TASKS_FILE path that witness never uses) — give it a harmless value.
export MODALITY="${MODALITY:-witness}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
# 35B GatedDeltaNet: FlashInfer GDN-prefill JIT hangs on RunPod → use triton (team convention).
export VLLM_GDN_PREFILL_BACKEND=triton

# --- Cross-node NCCL: force TCP sockets (2026-06-02) ---
# Inter-node IB is broken for our node pair: a 3.5 MB BROADCAST (SeqNum=1) hung 10 min on
# rails that are "active" on BOTH nodes (verified /workspace/.sky_ib_hca: node-8 & node-9
# both expose mlx5_0,1,2,3,6,7,8,9, yet the collective never connected → the IB fabric
# doesn't route between these nodes on those rails). FM/NVLink are healthy (Fabric:
# Completed/Success, NVLink up), so it's specifically inter-node IB. Disable IB → NCCL uses
# TCP over NCCL_SOCKET_IFNAME (ens1, from /etc/environment). NCCL_DEBUG=INFO prints the
# chosen transport + per-interface speeds, so if TCP is too slow we can switch to RoCE on
# the 200G Ethernet NICs (mlx5_10/11) instead.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# Reward / harness env consumed by the witness env + agent harness.
export ENABLE_PLAN_DIVERSITY_PENALTY PLAN_DIVERSITY_SCHEME PLAN_DIVERSITY_PENALTY
export ENABLE_RULE_JUDGE_REWARD RULE_JUDGE_MODEL RULE_JUDGE_SAMPLE_RATE RULE_JUDGE_MAX_REWARD
export ENABLE_RUBRIC_REWARD RUBRIC_MODEL RUBRIC_SAMPLE_RATE RUBRIC_PER_ITEM_REWARD
export OPENROUTER_API_KEY OPENROUTER_BASE_URL USE_ENV_GROUND_TRUTH AGENT_MLLM_ENABLED
export ENABLE_SECONDARY_PROVIDER_FALLBACK SECONDARY_MODEL SECONDARY_SKIP_PRIMARY

# OpenRouter preflight (skipped under SMOKE_ONLY)
if [ "${SMOKE_ONLY:-0}" != "1" ] && { [ "${ENABLE_RULE_JUDGE_REWARD:-0}" = "1" ] || [ "${ENABLE_RUBRIC_REWARD:-0}" = "1" ] || [ "${ENABLE_SECONDARY_PROVIDER_FALLBACK:-0}" = "1" ]; }; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: judge/fallback enabled but OPENROUTER_API_KEY is empty" >&2; exit 1; }
  echo "[preflight] OpenRouter-backed path enabled; key present (last 4: ${OPENROUTER_API_KEY: -4})"
fi

# End-to-end import smoke (catches WitnessAgentEnv path issues before the 35B load)
cd ~/sky_workdir
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["ARC_WITNESS_AGENT_DIR"])
from agent.runtime.process_reward import compute_plan_diversity_penalty, compute_rule_judge_reward, compute_rubric_reward
from examples.train_integrations.witness.env_agent import WitnessAgentEnv
print("[ok] WitnessAgentEnv + mounted latest agent harness import")
PY

if [ "${SMOKE_ONLY:-0}" = "1" ]; then
  echo "[smoke] SMOKE_ONLY=1 complete: mounts, venv, env imports, reward hooks, config gates validated."
  exit 0
fi

DATA_DIR="$HOME/data/witness_v5b7"
POLICY_PATH=$(cat "$HOME/policy_path.txt")
CKPT_LOCAL_DIR="/workspace/guanghan/ckpts/witness_grpo_v5b7_${RUN_LABEL}"
EXPORT_LOCAL_DIR="/workspace/guanghan/exports/witness_grpo_v5b7_${RUN_LABEL}"
CKPT_S3_DIR="s3://fleet-guanghan/witness_grpo_v5b7_${RUN_LABEL}"
EXPORT_S3_DIR="s3://fleet-guanghan/witness_grpo_v5b7_${RUN_LABEL}/hf_export"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-100000}"   # > total steps ⇒ only end-of-run HF export

# === rank-0 only: S3 ckpt mirror + 16-shard corruption assert + final HF-export→S3 ===
# common-run launches the trainer in the FOREGROUND on rank 0; this background loop runs
# alongside it, watching the witness ckpt dir on shared /workspace. Workers (sleep infinity
# inside common-run) and SMOKE never reach here.
if [ "${SKYPILOT_NODE_RANK:-0}" = "0" ]; then
  EXPECTED_SHARDS=$(( ${SKYPILOT_NUM_GPUS_PER_NODE:-8} * ${SKYPILOT_NUM_NODES:-1} ))
  (
    set +e +o pipefail   # best-effort: a benign SIGPIPE/non-zero must never kill the mirror
    LAST_SYNCED_DIR=""
    while true; do
      sleep 300  # 5 min
      [ -d "$CKPT_LOCAL_DIR" ] || continue
      NEWEST=$(ls -td "$CKPT_LOCAL_DIR"/global_step_* 2>/dev/null | head -1)
      if [ -n "$NEWEST" ] && [ "$NEWEST" != "$LAST_SYNCED_DIR" ]; then
        BN=$(basename "$NEWEST")
        # SkyRL never asserts shard-count == world_size; that gap let the corrupt 8/16-rank
        # p2_r5_35b ckpt through silently. /workspace is shared, so rank0 must see ALL ranks.
        SHARD_CNT=$(ls -1 "$NEWEST"/policy/model_world_size_*_rank_*.pt 2>/dev/null | wc -l | tr -d ' ')
        if [ "$SHARD_CNT" -ne "$EXPECTED_SHARDS" ]; then
          echo "[ckpt-assert $(date +%H:%M:%S)] WARNING: $BN has $SHARD_CNT/$EXPECTED_SHARDS model shards -- PARTIAL ckpt, resume would be CORRUPT (a node's shards never reached shared /workspace)"
        else
          echo "[ckpt-assert $(date +%H:%M:%S)] OK: $BN has all $SHARD_CNT/$EXPECTED_SHARDS model shards"
        fi
        echo "[ckpt-mirror $(date +%H:%M:%S)] syncing $BN to S3"
        aws s3 sync "$NEWEST" "$CKPT_S3_DIR/$BN" --quiet 2>/dev/null && LAST_SYNCED_DIR="$NEWEST" \
          || echo "[ckpt-mirror] sync failed (will retry next cycle)"
      fi
      [ -f "$CKPT_LOCAL_DIR/latest_ckpt_global_step.txt" ] && \
        aws s3 cp "$CKPT_LOCAL_DIR/latest_ckpt_global_step.txt" \
          "$CKPT_S3_DIR/latest_ckpt_global_step.txt" --quiet 2>/dev/null || true
    done
  ) &
  CKPT_MIRROR_PID=$!
  # On exit (trainer done OR crash): stop the mirror, then a final full sync of the sharded
  # ckpts AND the consolidated HF export (end-of-run save_models fires before this trap).
  trap '
    kill $CKPT_MIRROR_PID 2>/dev/null || true
    [ -d "$CKPT_LOCAL_DIR" ] && aws s3 sync "$CKPT_LOCAL_DIR" "$CKPT_S3_DIR/" --quiet 2>/dev/null || true
    [ -d "$EXPORT_LOCAL_DIR" ] && aws s3 sync "$EXPORT_LOCAL_DIR" "$EXPORT_S3_DIR/" --quiet 2>/dev/null || true
  ' EXIT
fi

# Delegate cluster bring-up + trainer launch to the shared runner. Witness Hydra overrides
# come after `--` and win over common-run's baselines (placement/strategy/env_class/
# num_inference_engines are injected by common-run from SKYPILOT_* — not repeated here).
# NOTE: not `exec` — we must keep this shell alive for the rank-0 mirror + EXIT trap.
set +e
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint examples.train_integrations.witness.entrypoints.main_witness \
  --env-class witness_agent -- \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.policy.model.path="$POLICY_PATH" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.flash_attn=false \
  generator.chat_template_kwargs='{enable_thinking:false}' \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  trainer.algorithm.loss_reduction="sequence_mean" \
  generator.inference_engine_tensor_parallel_size="${INFERENCE_TENSOR_PARALLEL_SIZE}" \
  trainer.epochs="${NUM_EPOCHS}" \
  trainer.eval_batch_size="${EVAL_BATCH_SIZE}" \
  trainer.eval_before_train="${EVAL_BEFORE_TRAIN}" \
  trainer.eval_interval="${EVAL_INTERVAL}" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="${TRAIN_BATCH_SIZE}" \
  trainer.policy_mini_batch_size="${POLICY_MINI_BATCH_SIZE}" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval="${CKPT_INTERVAL}" \
  trainer.max_ckpts_to_keep="${MAX_CKPTS_TO_KEEP}" \
  trainer.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  generator.max_input_length="${MAX_INPUT_LENGTH}" \
  generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.95 \
  generator.eval_sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
  trainer.policy.optimizer_config.lr="${LR}" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef="${KL_LOSS_COEF}" \
  trainer.algorithm.use_entropy_loss=true \
  trainer.algorithm.entropy_loss_coef="${ENTROPY_LOSS_COEF}" \
  generator.max_turns="${MAX_TURNS}" \
  generator.backend="${INFERENCE_BACKEND}" \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.enforce_eager=false \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt="${N_SAMPLES_PER_PROMPT}" \
  generator.eval_n_samples_per_prompt="${EVAL_N_SAMPLES_PER_PROMPT}" \
  generator.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  trainer.logger="${LOGGER}" \
  trainer.project_name="arc-agi-3" \
  trainer.run_name="witness_grpo_v5b7_${RUN_LABEL}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_LOCAL_DIR" \
  trainer.hf_save_interval="${HF_SAVE_INTERVAL}" \
  trainer.export_path="$EXPORT_LOCAL_DIR" \
  trainer.dump_data_batch=true \
  "$@"
RC=$?
set -e
echo "[witness-run] fleet-common-run.sh exited code=$RC"
exit "$RC"
