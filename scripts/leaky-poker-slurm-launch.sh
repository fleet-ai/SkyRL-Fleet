#!/usr/bin/env bash
# Launch a leaky_poker GRPO arm on the RunPod Slurm cluster via a direct sbatch (bypasses SkyPilot;
# /workspace is NFS-shared across nodes so paths + venvs + the probe all resolve). Excludes the
# down node-8. The DENSE arm co-locates the reader service on GPU 7 (localhost); the trainer then
# uses GPUs 0-6. SPARSE arm needs no reader and uses all 8 GPUs.
#
#   bash scripts/leaky-poker-slurm-launch.sh <arm-name> <dense|sparse> <exploiter|llm|scripted>
# e.g.
#   bash scripts/leaky-poker-slurm-launch.sh sparseExpl sparse exploiter
#   bash scripts/leaky-poker-slurm-launch.sh denseExpl  dense  exploiter
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
ARM="${1:?arm name}"; REWARD="${2:?dense|sparse}"; OPP="${3:-exploiter}"
LOGDIR="$HOME/leaky_poker_logs"; mkdir -p "$LOGDIR"

if [ "$REWARD" = "dense" ]; then GPUS_TRAINER="0,1,2,3,4,5,6"; NENG=7; NEED_READER=1
else GPUS_TRAINER="0,1,2,3,4,5,6,7"; NENG=8; NEED_READER=0; fi

JOBSH="$LOGDIR/${ARM}_job.sh"
cat > "$JOBSH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lpk_$ARM
#SBATCH --exclude=node-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=$LOGDIR/${ARM}_%j.out
set -euo pipefail
cd "$REPO"
set -a; . /workspace/allie/.env; set +a          # WANDB / OPENROUTER / AWS / HF creds
export SKYPILOT_NUM_NODES=1 SKYPILOT_NODE_RANK=0 SKYPILOT_NUM_GPUS_PER_NODE=$NENG
export LEAKY_POKER_TEXTARENA=/workspace/allie/TextArena
export LEAKY_POKER_DECEPTION_DIR=/workspace/allie/TextArena/deception_poc
if [ "$NEED_READER" = "1" ]; then
  echo "[node] starting reader service on GPU 7"
  CUDA_VISIBLE_DEVICES=7 /workspace/allie/performative/.venv/bin/python \
    /workspace/allie/TextArena/deception_poc/reader_service.py \
    --probe /workspace/allie/TextArena/deception_poc/probes/probe_leakreader_qwen3_8b.npz \
    --host 0.0.0.0 --port 8137 > "$LOGDIR/${ARM}_reader.log" 2>&1 &
  for i in \$(seq 1 60); do curl -s http://127.0.0.1:8137/health 2>/dev/null | grep -q '"ok": true' && break; sleep 3; done
  export READER_BASE_URL=http://127.0.0.1:8137 READER_MODE=endpoint
fi
export CUDA_VISIBLE_DEVICES=$GPUS_TRAINER
export MODALITY=leaky_poker MODEL_PATH=Qwen/Qwen3.5-9B MODEL_TAG=qwen35
export REWARD_MODE=$REWARD OPPONENT_MODE=$OPP NUM_INFERENCE_ENGINES=$NENG
export NUM_EPOCHS=\${NUM_EPOCHS:-3}
exec bash scripts/fleet-leaky-poker-9b-run.sh
EOF

echo "[launch] arm=$ARM reward=$REWARD opp=$OPP engines=$NENG reader=$NEED_READER"
echo "[launch] job script: $JOBSH"
sbatch "$JOBSH"
