#!/usr/bin/env bash
# Launch the SELF-PLAY leaky_poker arm: one node serves the opponent snapshot (vLLM OpenAI server),
# a second node trains the hero against it (dense reward + co-located reader). Round 0 opponent = the
# base policy Qwen3.5-9B; rotate by re-serving a newer checkpoint and relaunching (armsrace_loop.py).
#
#   bash scripts/leaky-poker-selfplay-launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
LOGDIR="$HOME/leaky_poker_logs"; mkdir -p "$LOGDIR"
PORT="${OPPONENT_SERVE_PORT:-6479}"

# 1) opponent server (serves Qwen3.5-9B as "qwen35-opponent" on 0.0.0.0:$PORT, 8 GPUs, fp8)
SRV="$LOGDIR/selfplay_server_job.sh"
cat > "$SRV" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lpk_sp_server
#SBATCH --exclude=node-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=$LOGDIR/selfplay_server_%j.out
set -euo pipefail
cd "$REPO"
set -a; . /workspace/allie/.env; set +a
export OPPONENT_SERVE_MODEL=Qwen/Qwen3.5-9B OPPONENT_SERVED_NAME=qwen35-opponent
export OPPONENT_SERVE_PORT=$PORT OPPONENT_GPUS=0,1,2,3,4,5,6,7
exec bash scripts/fleet-negotiation-opponent-serve.sh
EOF
SRV_JOB=$(sbatch --parsable "$SRV")
echo "[selfplay] opponent server job=$SRV_JOB"

# 2) wait for the server node, then its /health
for i in $(seq 1 60); do
  NODE=$(squeue -j "$SRV_JOB" -h -o "%N" 2>/dev/null || true)
  [ -n "$NODE" ] && [ "$NODE" != "(null)" ] && break
  sleep 5
done
echo "[selfplay] server node=$NODE ; waiting for it to answer /v1/models"
for i in $(seq 1 120); do
  curl -s "http://${NODE}:${PORT}/v1/models" 2>/dev/null | grep -q "qwen35-opponent" && { echo "[selfplay] opponent ready"; break; }
  sleep 10
done

# 3) training node: dense reward, opponent_mode=llm pointed at the server, reader on GPU 7
TRAIN="$LOGDIR/selfplay_train_job.sh"
cat > "$TRAIN" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lpk_sp_train
#SBATCH --exclude=node-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=$LOGDIR/selfplay_train_%j.out
set -euo pipefail
cd "$REPO"
set -a; . /workspace/allie/.env; set +a
export SKYPILOT_NUM_NODES=1 SKYPILOT_NODE_RANK=0 SKYPILOT_NUM_GPUS_PER_NODE=7
export SKYPILOT_NODE_IPS="\$(hostname -I | awk '{print \$1}')"
export LEAKY_POKER_TEXTARENA=/workspace/allie/TextArena
export LEAKY_POKER_DECEPTION_DIR=/workspace/allie/TextArena/deception_poc
CUDA_VISIBLE_DEVICES=7 /workspace/allie/performative/.venv/bin/python \
  /workspace/allie/TextArena/deception_poc/reader_service.py \
  --probe /workspace/allie/TextArena/deception_poc/probes/probe_leakreader_qwen3_8b.npz \
  --host 0.0.0.0 --port 8137 > "$LOGDIR/selfplay_reader.log" 2>&1 &
for i in \$(seq 1 60); do curl -s http://127.0.0.1:8137/health 2>/dev/null | grep -q '"ok": true' && break; sleep 3; done
export READER_BASE_URL=http://127.0.0.1:8137 READER_MODE=endpoint
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export MODALITY=leaky_poker MODEL_PATH=Qwen/Qwen3.5-9B MODEL_TAG=qwen35
export REWARD_MODE=dense OPPONENT_MODE=llm NUM_INFERENCE_ENGINES=7 NUM_EPOCHS=\${NUM_EPOCHS:-3}
export OPPONENT_MODEL=openai/qwen35-opponent OPPONENT_BASE_URL=http://${NODE}:${PORT}/v1
exec bash scripts/fleet-leaky-poker-9b-run.sh
EOF
TRAIN_JOB=$(sbatch --parsable "$TRAIN")
echo "[selfplay] training job=$TRAIN_JOB (opponent=http://${NODE}:${PORT}/v1)"
