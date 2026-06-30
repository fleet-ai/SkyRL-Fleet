#!/usr/bin/env bash
# Witness SETUP — runs from the SkyPilot `setup:` block (one per node).
#
# Why witness has its OWN setup script (vs the team's fleet-common-setup.sh):
# common-setup is hardwired to the fleet S3 modality-dataset flow (hard-requires
# FLEET_API_KEY + MODALITY and unconditionally `aws s3 cp`s s3://.../all_$MODALITY.json
# BEFORE --skip-prepare). Witness builds its data locally via prepare_witness_dataset.py
# and uses no fleet dataset. Rather than hack common-setup with dummy envs, we keep a
# witness-owned setup and reuse only the team's qwen35 dep hook (fleet-qwen35-extra-setup.sh).
#
# We DO adopt common-setup's key reliability trick: the .venv lives on shared NFS, so only
# rank 0 installs (single-node install) and workers wait on a sentinel. This eliminates the
# two-nodes-installing-into-one-NFS-venv race that produced transient
# `ModuleNotFoundError: transformers.model_debugging_utils` / `flash_attn_2_cuda` on re-runs.
set -euo pipefail

# Key the sentinel on SLURM_JOB_ID (same across all nodes in the allocation) so a stale
# sentinel left on the shared-NFS .venv by a PRIOR job can't be mistaken for this job's →
# workers correctly wait for THIS job's rank-0 install, not an old one.
SENTINEL=".venv/.witness_setup_complete_${SLURM_JOB_ID:-${SLURM_JOBID:-nojob}}"

# --- Worker nodes: the shared-NFS .venv is installed once by rank 0. Just wait + activate. ---
if [ "${SLURM_PROCID:-${SKYPILOT_NODE_RANK:-0}}" != "0" ]; then
  echo "[witness-setup] worker (procid=${SLURM_PROCID:-?}) waiting for rank-0 install sentinel..."
  for _ in $(seq 1 360); do [ -f "$SENTINEL" ] && break; sleep 5; done   # up to 30 min
  [ -f "$SENTINEL" ] || { echo "ERROR: rank-0 setup sentinel never appeared after 30 min" >&2; exit 1; }
  source .venv/bin/activate
  echo "[witness-setup] worker: shared venv ready (installed by rank 0)."
  exit 0
fi

# ============================ rank 0 only from here ============================
rm -f "$SENTINEL"

if ! command -v c++ &>/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends build-essential
fi
# ---------------------------------------------------------------------------------------------
# Clean venv rebuild every launch — reproduces the team's effective "fresh disk" behavior.
#
# Witness is pinned to a persistent /workspace NFS volume, so a venv from a prior job survives.
# Because we pip-install extras beyond the lockfile (arc-agi/openai/pyyaml), a REUSED venv has
# DIVERGED from it — and `uv sync` then tries to prune those extras, dying while removing their
# __pycache__ on NFS (`os error 39 / ENOTEMPTY`), leaving a half-pruned, gutted venv. The team
# never hits this: their clouds hand out a fresh disk per launch, so `uv sync` only INSTALLS,
# never prunes. We reproduce that deterministically — clear any prior venv, then build clean
# (install-only on an empty venv can't hit the NFS-removal ENOTEMPTY). No skip-sync probe, no
# --inexact: those were workarounds that drifted from the team and could still trust a bad venv.
# Trade-off: one cold build (~10-15 min) per launch, accepted for determinism.
#
# NFS-safe clear: `mv` (rename) a populated venv aside, never `rm -rf`. If orphan processes from a
# prior job still hold its .so files open, NFS silly-renames deleted-but-open files to .nfsXXXX and
# rmdir reports ENOTEMPTY. `mv` works regardless of open handles. The .venv.broken.* dirs are
# harmless — delete them later once no process holds them.
#
# The aside target MUST be unique per LAUNCH, not per allocation: SLURM_JOB_ID is identical across
# repeated `sky launch` on a REUSED allocation (each launch re-runs setup), so a bare-job-id target
# collides — 2nd launch moves .venv INTO the existing .venv.broken.<id>/, 3rd launch hits
# "cannot overwrite .venv.broken.<id>/.venv: File exists", falls to `rm -rf .venv`, hits the
# orphan-held .nfsXXXX (Device or resource busy) → FAILED_SETUP (debugged 2026-06-14). Append $$
# (pid, unique per launch) so the rename always succeeds.
for _bd in .venv.broken.*; do [ -e "$_bd" ] && rm -rf "$_bd" 2>/dev/null || true; done  # declutter old asides (best-effort; skips busy .nfs)
if [ -d .venv ]; then mv .venv ".venv.broken.${SLURM_JOB_ID:-nojob}.$$" || rm -rf .venv; fi
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra fsdp
for f in .venv/bin/ray .venv/lib/python*/site-packages/ray/core/src/ray/raylet/raylet; do
  [ -f "$f" ] && chmod +x "$f" 2>/dev/null || true
done
uv pip install wandb boto3 awscli pyyaml openai
# 35B-specific deps (shared with the team's task-gen): transformers 5.3.0, flash-attn 2.8.3,
# CUDA toolkit (writes $HOME/.cuda_env), causal-conv1d built from source.
source scripts/fleet-qwen35-extra-setup.sh
uv pip install arc-agi

export WITNESS_ENVS_DIR=$HOME/arc-witness-envs
export ARC_WITNESS_AGENT_DIR=$HOME/arc-witness-agent

test -f "$ARC_WITNESS_AGENT_DIR/agent/core.py" || { echo "ERROR: arc-witness-agent file_mount missing"; exit 1; }
test -f "$WITNESS_ENVS_DIR/witness_grid.py" || { echo "ERROR: arc-witness-envs file_mount missing"; exit 1; }
echo "[ok] file_mounts present | RUN_LABEL=$RUN_LABEL | MODEL=$MODEL"

# Latest-harness import check: catches stale file_mounts before burning $30/hr loading 35B.
python - <<'PY'
import os, sys
agent_dir = os.environ["ARC_WITNESS_AGENT_DIR"]
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)
from agent.runtime.process_reward import (
    compute_plan_diversity_penalty,
    compute_rule_judge_reward,
    compute_rubric_reward,
)
from agent.llm.client import LLMClient
from agent.core import AgentCore
print("[ok] latest agent reward + LLM fallback symbols import")
PY

# Phase 2 expects example plan '1,4,2,5,3' removed from meta_reasoning prompt
if grep -q '<plan>1,4,2,5,3</plan>' "$ARC_WITNESS_AGENT_DIR/agent/decision/meta_reasoning.py"; then
  echo "ERROR: prompt example '1,4,2,5,3' still in meta_reasoning.py — Phase 2 expects it removed"
  exit 1
fi
echo "[ok] prompt example correctly dropped"

# Determine POLICY_PATH once in setup; run section reads from file (no drift).
POLICY_PATH="$MODEL"
if [ -n "${POLICY_CHECKPOINT_S3:-}" ]; then
  mkdir -p "$HOME/policy_ckpt"
  echo "[policy] downloading 35B policy checkpoint from $POLICY_CHECKPOINT_S3"
  aws s3 sync "$POLICY_CHECKPOINT_S3" "$HOME/policy_ckpt/" 2>&1 | tail -10

  # Multimodal config patch (Finding K: SkyRL save_hf_model omits preprocessor_config.json)
  echo "[patch] downloading multimodal config from $MODEL → ~/policy_ckpt"
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL', allow_patterns=['preprocessor_config.json','processor_config.json','chat_template.json','chat_template.jinja','video_preprocessor_config.json'], local_dir='$HOME/policy_ckpt', local_dir_use_symlinks=False)"
  POLICY_PATH="$HOME/policy_ckpt"
else
  echo "[policy] POLICY_CHECKPOINT_S3 empty — using $MODEL (instruct, no witness task SFT) via vLLM HF cache"
fi
echo "$POLICY_PATH" > "$HOME/policy_path.txt"
echo "[policy] POLICY_PATH=$POLICY_PATH"

# Spot-preemption recovery: download prior ckpts for this RUN_LABEL if any (shared /workspace).
CKPT_S3_BASE="s3://fleet-guanghan/witness_grpo_v5b7_${RUN_LABEL}"
CKPT_LOCAL="${WITNESS_CKPT_BASE:-/workspace/guanghan/rl_ckpts}/witness_grpo_v5b7_${RUN_LABEL}"  # match fleet-witness-run.sh (legacy ckpts owned by orphaned uid 1004 → unwritable as squashed-root)
echo "[resume] checking for prior ckpts at $CKPT_S3_BASE"
if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && aws s3 ls "$CKPT_S3_BASE/global_step_" 2>/dev/null | head -1 | grep -q .; then
  echo "[resume] prior ckpts found, downloading to $CKPT_LOCAL"
  mkdir -p "$CKPT_LOCAL"
  aws s3 sync "$CKPT_S3_BASE/" "$CKPT_LOCAL/" --no-progress --exclude "logs/*" 2>&1 | tail -10
  echo "[resume] downloaded; will use trainer.resume_mode=latest"
  ls -d "$CKPT_LOCAL"/global_step_* 2>/dev/null
else
  echo "[resume] no prior ckpts, or AWS creds unavailable — fresh training start"
fi

# Witness dataset prep (this is why we can't reuse common-setup's fleet dataset flow).
cd ~/sky_workdir
python examples/train_integrations/witness/prepare_witness_dataset.py \
  --game_ids $GAME_IDS \
  --val_game_ids $VAL_GAME_IDS \
  --reward_mode $REWARD_MODE \
  --obs_mode $OBS_MODE \
  --rules_mode $RULES_MODE \
  --max_levels $MAX_LEVELS \
  --env_class witness_agent \
  --max_orai_steps $MAX_ORAI_STEPS \
  --output_dir $HOME/data/witness_v5b7

touch "$SENTINEL"
echo "[witness-setup] rank 0 complete; sentinel written — workers released."
