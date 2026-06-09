#!/usr/bin/env bash
# Fleet shared run: Ray cluster setup (multi-node aware) + training launch
#
# Usage (from SkyPilot YAML run block):
#   bash skyrl-train/scripts/fleet-common-run.sh \
#     --use-python-direct --cuda-env "$HOME/.cuda_env" \
#     --set-ulimit --no-pytorch-alloc-conf -- \
#     trainer.policy.model.path="Qwen/Qwen3.5-9B" \
#     trainer.epochs=20 ...
#
# Multi-node:
#   Rank 0 (head): starts Ray head, launches training
#   Rank >0 (workers): joins Ray cluster, sleeps
#
# Required env vars: WANDB_API_KEY, MODALITY, INFERENCE_BACKEND,
#   SKYPILOT_NUM_GPUS_PER_NODE, SKYPILOT_NODE_IPS
# Optional env vars: SKYPILOT_NUM_NODES, SKYPILOT_NODE_RANK
set -euo pipefail

# Defaults
DATA_ROOT=""
CKPT_ROOT=""
USE_PYTHON_DIRECT=false
CUDA_ENV=""
SET_ULIMIT=false
NO_PYTORCH_ALLOC_CONF=false
NCCL_HEARTBEAT=""
ENTRYPOINT="integrations.fleet.entrypoints.main_fleet"
ENV_CLASS="fleet_task"
DATA_DIR_NAME=""
HYDRA_OVERRIDES=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --use-python-direct) USE_PYTHON_DIRECT=true; shift ;;
    --cuda-env) CUDA_ENV="$2"; shift 2 ;;
    --set-ulimit) SET_ULIMIT=true; shift ;;
    --no-pytorch-alloc-conf) NO_PYTORCH_ALLOC_CONF=true; shift ;;
    --nccl-heartbeat) NCCL_HEARTBEAT="$2"; shift 2 ;;
    --entrypoint) ENTRYPOINT="$2"; shift 2 ;;
    --env-class) ENV_CLASS="$2"; shift 2 ;;
    --data-dir-name) DATA_DIR_NAME="$2"; shift 2 ;;
    --) shift; HYDRA_OVERRIDES=("$@"); break ;;
    *) echo "ERROR: Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-detect data/ckpt root: /workspace if writable (RunPod), else $HOME.
# On shared /workspace (Slurm), scope by sky cluster name so concurrent runs
# with the same MODALITY don't read each other's prepared parquet files.
# Must mirror the resolution in fleet-common-setup.sh.
if [ -z "$DATA_ROOT" ]; then
  if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    SCRIPT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CLUSTER_ID="$(echo "$SCRIPT_ABS" | sed -n 's|.*/.sky_clusters/\([^/]*\)/.*|\1|p')"
    if [ -n "$CLUSTER_ID" ]; then
      DATA_ROOT="/workspace/clusters/${CLUSTER_ID}"
    else
      DATA_ROOT="/workspace"
    fi
  else
    DATA_ROOT="$HOME"
  fi
fi
if [ -z "$CKPT_ROOT" ]; then
  CKPT_ROOT="$DATA_ROOT"
fi
DATA_DIR_NAME="${DATA_DIR_NAME:-$MODALITY}"

echo "=== Fleet Common Run ==="
echo "Entrypoint: $ENTRYPOINT"
echo "Env class: $ENV_CLASS"
echo "Data root: $DATA_ROOT"
echo "Data dir name: $DATA_DIR_NAME"
echo "Ckpt root: $CKPT_ROOT"

# Activate venv from repo root (upstream SkyRL layout)
source .venv/bin/activate

# --- Optional settings ---
if [ "$SET_ULIMIT" = true ]; then
  # Set open files limit. Try 1M first, fall back to hard limit if too high.
  ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
fi

# vLLM TP>1 uses pidfd_getfd for CUDA IPC weight sync between Ray workers.
# This requires ptrace access, which is blocked by default (ptrace_scope=1).
sudo sysctl -w kernel.yama.ptrace_scope=0 2>/dev/null || true

if [ -n "$CUDA_ENV" ]; then
  source "$CUDA_ENV" 2>/dev/null || true
fi

if [ "$NO_PYTORCH_ALLOC_CONF" = false ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

if [ -n "$NCCL_HEARTBEAT" ]; then
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$NCCL_HEARTBEAT"
fi

TMP_DIR="${CKPT_ROOT}/skyrl-tmp"
mkdir -p "$TMP_DIR"
# TMPDIR and HF_HOME must be LOCAL, not NFS. On SLURM (RunPod), /workspace is NFS
# and SkyPilot sets HOME to /workspace/.sky_clusters/... so ~/.cache is also NFS.
# - filelock on NFS causes ESTALE (errno 116) when concurrent vLLM engines race
# - Ray needs local temp for Unix domain sockets (UDS don't work over NFS)
# Ray's temp dir AND ports must be unique per job. These SLURM nodes are
# multi-tenant: other users' SkyPilot jobs share the physical node, and Ray's
# temp dir + ports are global per node. A bare /tmp/skyrl-ray + fixed port 6479
# collide with a co-tenant's root-owned Ray (which we can't kill) — our head then
# connects to their GCS and dies with a session-name mismatch. Key everything on
# SLURM_JOB_ID: it's identical across all nodes in the allocation, so head and
# workers independently derive the same temp dir and ports and still agree.
JOB_KEY="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
[ -n "$JOB_KEY" ] || JOB_KEY=$(pwd | cksum | cut -d' ' -f1)  # cwd = shared NFS workdir
RAY_TMPDIR="/tmp/skyrl-ray-${JOB_KEY}"
# A 500-port block per job in [20000,60000): GCS, client-server, dashboard, plus a
# worker-port range clear of a co-tenant's default 10002-19999.
RAY_PORT_BASE=$(( 20000 + (10#$JOB_KEY % 80) * 500 ))
RAY_GCS_PORT=$RAY_PORT_BASE
RAY_CLIENT_PORT=$(( RAY_PORT_BASE + 1 ))
RAY_DASHBOARD_PORT=$(( RAY_PORT_BASE + 2 ))
# Pin Ray's per-node agent ports (BASE+3..+6) explicitly, BELOW the worker-port floor.
# Without this, Ray auto-picks them from the OS ephemeral range (32768-60999); when a high
# JOB_KEY pushes the worker range [BASE+50,BASE+499] into that ephemeral range, an auto-picked
# agent port can land INSIDE the worker range and Ray refuses to start with
# "ValueError: Ray component worker_ports is trying to use a port number N used by other
# components" (observed: JOB_KEY=1259 -> worker 49550-49999, dashboard_agent_grpc=49960).
# These are within this job's 500-port block, so they stay clear of co-tenant ports too.
RAY_DASH_AGENT_GRPC_PORT=$(( RAY_PORT_BASE + 3 ))
RAY_DASH_AGENT_HTTP_PORT=$(( RAY_PORT_BASE + 4 ))
RAY_RUNTIME_ENV_AGENT_PORT=$(( RAY_PORT_BASE + 5 ))
RAY_METRICS_EXPORT_PORT=$(( RAY_PORT_BASE + 6 ))
RAY_WORKER_PORT_MIN=$(( RAY_PORT_BASE + 50 ))
RAY_WORKER_PORT_MAX=$(( RAY_PORT_BASE + 499 ))
mkdir -p "$RAY_TMPDIR"
export TMPDIR="/tmp"
export RAY_TMPDIR="$RAY_TMPDIR"
# Namespace local caches by UID. /tmp is shared by all users on a node, so a
# bare /tmp/hf_cache created by an earlier root job is root-owned and not
# writable by other users (PermissionError under .../hub/models--...).
# flashinfer-jit-cache and flashinfer-python versions may have minor suffix
# mismatches (e.g., 0.6.12+cu129 vs 0.6.11.post2). The kernels are compatible.
# Must be set before Ray workers start (they don't inherit run script env).
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Disable DeepGemm JIT. Multi-process race condition during simultaneous
# kernel compilation causes "runtime != nullptr" assertion failure when
# multiple TP workers try to compile the same kernel concurrently.
# See: https://github.com/deepseek-ai/DeepGEMM/issues/301
export VLLM_USE_DEEP_GEMM=0
# Use NFS cache if pre-downloaded by setup, else fall back to local /tmp.
if [ -d "/workspace/hf_cache/hub" ]; then
  export HF_HOME="/workspace/hf_cache"
else
  export HF_HOME="/tmp/hf_cache-$(id -u)"
  mkdir -p "$HF_HOME"
fi
# Triton's JIT cache must also be LOCAL: shared NFS causes ESTALE during
# concurrent kernel compilation across nodes (errno 116, "Stale file handle").
export TRITON_CACHE_DIR="/tmp/triton_cache-$(id -u)"
mkdir -p "$TRITON_CACHE_DIR"

TASKS_FILE="${DATA_ROOT}/data/fleet/tasks_${MODALITY}.json"
DATA_DIR="${DATA_ROOT}/data/fleet/${DATA_DIR_NAME}"

# --- System diagnostics ---
echo "=== System Diagnostics ==="
free -h
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv 2>/dev/null || true
echo "--- /dev/shm ---"
df -h /dev/shm 2>/dev/null || echo "/dev/shm not mounted"
ls -la /dev/shm/ 2>/dev/null | head -5 || true
echo "--- GPU Topology ---"
nvidia-smi topo -m 2>/dev/null || true
echo "--- cgroup memory limits ---"
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "No cgroup memory limit found"
echo "--- ulimits ---"
ulimit -a 2>/dev/null || true
echo "--- NCCL env vars ---"
env | grep -i NCCL || echo "No NCCL env vars set"
echo "--- kernel overcommit ---"
cat /proc/sys/vm/overcommit_memory 2>/dev/null || true
echo "=== End Diagnostics ==="

# --- wandb login ---
python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

# --- Fabric Manager check (NVSwitch GPUs: B200, H200 SXM) ---
# On non-GCP clouds (RunPod, Lambda, etc.), Fabric Manager is required for NVLink
# P2P on NVSwitch systems. Without it, dist.broadcast() in FSDP causes SIGKILL.
#
# On GCP, NVSwitch is managed at the HOST level — the guest VM does not have
# NVSwitch devices, so FM reports "NV_WARN_NOTHING_TO_DO" and cannot start.
# This is EXPECTED. NVLink P2P works through GCP's host-managed fabric without FM.
# GCP also provides a custom NCCL shim (gIB) that manages all NCCL configuration.
# Do NOT set NCCL_P2P_DISABLE or NCCL_NVLS_ENABLE on GCP with RDMA —
# the shim's "Guest Config Checker" expects these to be unset.
# NCCL_CUMEM_ENABLE=0 is set below for GCP WITHOUT RDMA to disable multicast.
ON_GCP=false
if [ -d "/usr/local/gib" ]; then
  ON_GCP=true
elif [ -f "/sys/class/dmi/id/product_name" ] && grep -qi "google" /sys/class/dmi/id/product_name 2>/dev/null; then
  ON_GCP=true
fi

FM_STATUS=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo "unknown")
echo "Fabric Manager status: $FM_STATUS"
echo "On GCP: $ON_GCP"

if [ "$ON_GCP" = true ]; then
  echo "GCP detected — skipping Fabric Manager restart (host manages NVSwitch)"

  # GCP's deep learning images install /etc/profile.d/nccl_env.sh which auto-sources
  # /usr/local/gib/scripts/set_nccl_env.sh and adds /usr/local/gib/lib64 to LD_LIBRARY_PATH.
  # This sets NCCL_NET=gIB, forcing the gIB network plugin for RDMA/InfiniBand.
  #
  # Problem: gIB requires RDMA hardware (ConnectX NICs + multiple GPUDirect VPC networks).
  # SkyPilot provisions VMs with a single management NIC — no RDMA networking.
  # When NCCL_NET=gIB is forced but gIB can't init, NCCL fails with
  # "Failed to initialize any NET plugin" → SIGKILL during dist.broadcast().
  #
  # Fix: check for RDMA devices. If absent, strip gIB so NCCL falls back to
  # NVLink P2P for intra-node communication. Multi-node uses GKE with RDMA.
  if [ -d "/sys/class/infiniband" ] && [ "$(ls /sys/class/infiniband/ 2>/dev/null)" ]; then
    echo "RDMA devices found — keeping gIB for GPUDirect RDMA"
  else
    echo "No RDMA devices — disabling gIB"
    # Remove gIB from LD_LIBRARY_PATH (set by /etc/profile.d/nccl_env.sh)
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | sed 's|/usr/local/gib/lib64:||g; s|:/usr/local/gib/lib64||g; s|/usr/local/gib/lib64||g')
    # Unset NCCL_NET=gIB so NCCL can fall back to NVLink P2P
    unset NCCL_NET
    # Clear gIB-specific vars set by set_nccl_env.sh
    unset NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_P2P_NET_CHUNKSIZE NCCL_NVLS_CHUNKSIZE
    unset NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_IB_FIFO_TC
    unset NCCL_TUNER_CONFIG_PATH
    # Disable CUDA multicast (requires NVSwitch fabric manager for GPU multicast
    # team setup). Without this, vLLM TP>1 hangs on CUDASymmetricMemory init.
    export NCCL_CUMEM_ENABLE=0
    echo "Cleared gIB NCCL env vars. Using NVLink P2P (intra-node)."
  fi
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "NCCL vars:"
  env | grep -i NCCL || echo "  (none)"

  # Ensure /dev/shm is large enough for NCCL IPC (some GCP images have small default)
  SHM_SIZE=$(df --output=size /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
  echo "Current /dev/shm size: ${SHM_SIZE}K"
  if [ -n "$SHM_SIZE" ] && [ "$SHM_SIZE" -lt 16777216 ]; then
    echo "WARNING: /dev/shm is only ${SHM_SIZE}K — remounting to 16G for NCCL"
    sudo mount -o remount,size=16G /dev/shm 2>&1 || echo "Failed to remount /dev/shm"
    df -h /dev/shm
  fi
elif [ "$FM_STATUS" != "active" ]; then
  echo "WARNING: Fabric Manager not active. Attempting restart..."
  sudo nvidia-smi -pm 1 2>&1 || true
  sudo systemctl stop nvidia-fabricmanager 2>&1 || true
  sleep 1
  sudo systemctl start nvidia-fabricmanager 2>&1 || true
  sleep 5
  FM_STATUS=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo "unknown")
  echo "Fabric Manager status after restart: $FM_STATUS"
  if [ "$FM_STATUS" != "active" ]; then
    echo "=== WARNING: Fabric Manager failed to start ==="
    echo "Training may fail if this system has NVSwitch GPUs."
    sudo journalctl -u nvidia-fabricmanager --no-pager -n 10 2>&1 || true
  fi
fi

# --- Ray cluster setup (multi-node aware) ---
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_object_store_memory=10000000000
# Disable Ray's memory monitor to prevent spurious worker kills
export RAY_DISABLE_MEMORY_MONITOR=1
# NOTE: On GCP VMs without RDMA, gIB NCCL vars are stripped above.
# On GKE with RDMA, gIB is preserved for inter-node GPUDirect.

# === Consistent NCCL_IB_HCA across all allocated nodes (per-job intersection) ===
# IB device names are NOT identical across nodes on this cluster: e.g. mlx5_3 is a live
# InfiniBand port on some nodes while on others mlx5_3 is DOWN and mlx5_5 is live instead.
# Two failure modes follow if NCCL_IB_HCA isn't handled carefully:
#   (A) one list naming a NIC that's down on some node  -> that rail hangs the first collective
#   (B) different lists across ranks (per-node detection) -> NCCL deadlocks (rails don't line up)
# Both show up as a ~10-min gloo/NCCL timeout during init. Fix: ib-hca-intersection.sh has each
# node publish its own active IB HCAs to a shared (NFS) dir, then returns the INTERSECTION -> a
# single consistent list containing only NICs live on ALL nodes. Overrides /etc/environment.
#
# NB: this script runs under `set -euo pipefail`. The intersection MUST run in its own process
# (`bash <helper>`), not inline -- an inline detection pipeline returns non-zero on its last
# (non-IB) device and `pipefail`+`set -e` would abort the whole run. The `|| true` is a second
# guard so a missing helper / no-IB host falls back to the inherited /etc/environment value.
if [ -d /sys/class/infiniband ] && [ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
  _ib_helper="$(dirname "${BASH_SOURCE[0]}")/ib-hca-intersection.sh"
  _ib_csv=$(bash "$_ib_helper" "/workspace/.sky_ib_hca/${JOB_KEY:-nojob}" "${SKYPILOT_NUM_NODES:-1}" 2>/dev/null || true)
  if [ -n "$_ib_csv" ]; then
    export NCCL_IB_HCA="$_ib_csv"
    echo "[NCCL] consistent NCCL_IB_HCA across ${SKYPILOT_NUM_NODES:-1} node(s) = $NCCL_IB_HCA"
  else
    echo "[NCCL] intersection helper produced nothing; keeping inherited NCCL_IB_HCA=${NCCL_IB_HCA:-unset}"
  fi
fi

read -r head_ip _ <<< "$SKYPILOT_NODE_IPS"
ray_address="$head_ip:$RAY_GCS_PORT"

wait_for_ray() {
  local address=$1
  for _ in $(seq 1 24); do
    if env -u RAY_ADDRESS ray status --address "$address" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: Ray cluster at $address failed to become ready" >&2
  return 1
}

cleanup_existing_ray() {
  echo "Cleaning this job's Ray state (temp dir: $RAY_TMPDIR, gcs port: $RAY_GCS_PORT)..."
  # Only touch THIS job's namespaced state. Broad pkill of gcs_server/raylet or
  # fuser on a shared port would (a) no-op against a co-tenant's root-owned Ray
  # and (b) kill another of our own concurrent jobs on this shared node. Matching
  # on our unique RAY_TMPDIR (present in every Ray process's --temp-dir) and our
  # per-job port targets only this job.
  pkill -9 -f "$RAY_TMPDIR" 2>/dev/null || true
  fuser -k "${RAY_GCS_PORT}/tcp" 2>/dev/null || true
  rm -rf "$RAY_TMPDIR" 2>/dev/null || true
  mkdir -p "$RAY_TMPDIR"
  for _ in $(seq 1 10); do
    if ! fuser "${RAY_GCS_PORT}/tcp" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  sleep 2
}

cleanup_existing_ray

if [ "${SKYPILOT_NODE_RANK:-0}" = "0" ]; then
  # === Head node: start Ray head + launch training ===
  # --node-ip-address: on SLURM, force Ray to use the overlay IP (from SKYPILOT_NODE_IPS)
  # instead of auto-detecting the Docker-internal IP (172.19.x.x). Without this, the head
  # registers as a ghost node and the placement group can't schedule GPU bundles.
  env -u RAY_ADDRESS ray start --head --disable-usage-stats --port "$RAY_GCS_PORT" \
    --ray-client-server-port "$RAY_CLIENT_PORT" --dashboard-port "$RAY_DASHBOARD_PORT" \
    --dashboard-agent-grpc-port "$RAY_DASH_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port "$RAY_DASH_AGENT_HTTP_PORT" \
    --runtime-env-agent-port "$RAY_RUNTIME_ENV_AGENT_PORT" \
    --metrics-export-port "$RAY_METRICS_EXPORT_PORT" \
    --min-worker-port "$RAY_WORKER_PORT_MIN" --max-worker-port "$RAY_WORKER_PORT_MAX" \
    --object-store-memory=10000000000 \
    --node-ip-address="$head_ip" --temp-dir="$RAY_TMPDIR"
  wait_for_ray "$ray_address"
  # Tell ray.init() where to find the cluster (needed when --temp-dir differs
  # from Ray's default, since auto-detection looks in the default temp dir).
  export RAY_ADDRESS="$ray_address"

  TOTAL_GPUS=$((SKYPILOT_NUM_GPUS_PER_NODE * ${SKYPILOT_NUM_NODES:-1}))
  export TOTAL_GPUS
  # NUM_INFERENCE_ENGINES can be overridden via env var for TP>1 (engines = GPUs / TP)
  NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-$TOTAL_GPUS}
  echo "=== Head node: $TOTAL_GPUS GPUs across ${SKYPILOT_NUM_NODES:-1} node(s), $NUM_INFERENCE_ENGINES inference engines ==="

  # Build training command
  CMD_ARGS=()
  if [ "$USE_PYTHON_DIRECT" = true ]; then
    CMD_ARGS=(python -m "$ENTRYPOINT")
  else
    CMD_ARGS=(uv run --isolated --extra "$INFERENCE_BACKEND" -m "$ENTRYPOINT")
  fi

  # Common hydra overrides (data paths, placement, strategy, checkpoints)
  CMD_ARGS+=(
    "data.train_data=['${DATA_DIR}/train.parquet']"
    "data.val_data=['${DATA_DIR}/validation.parquet']"
    "environment.env_class=$ENV_CLASS"
  )

  # fleet_task-specific: pass tasks_file path
  if [ "$ENV_CLASS" = "fleet_task" ]; then
    CMD_ARGS+=("environment.skyrl_gym.fleet_task.tasks_file=$TASKS_FILE")
  fi

  CMD_ARGS+=(
    trainer.placement.colocate_all=true
    trainer.strategy=fsdp2
    "trainer.placement.policy_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.ref_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.policy_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "trainer.placement.ref_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "generator.num_inference_engines=$NUM_INFERENCE_ENGINES"
    "trainer.ckpt_path=${CKPT_ROOT}/ckpts"
    "trainer.export_path=${CKPT_ROOT}/exports"
    trainer.dump_training_trajectories=true
  )

  # Append model-specific hydra overrides (passed after --)
  if [ ${#HYDRA_OVERRIDES[@]} -gt 0 ]; then
    CMD_ARGS+=("${HYDRA_OVERRIDES[@]}")
  fi

  export HYDRA_FULL_ERROR=1
  echo "=== Launching Training ==="
  set +e
  "${CMD_ARGS[@]}"
  EXIT_CODE=$?
  set -e

  if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Training failed (exit code $EXIT_CODE) ==="
    echo "--- dmesg (last 50 lines, unfiltered) ---"
    sudo dmesg -T 2>/dev/null | tail -50 || true
    echo "--- dmesg (OOM/kill/segfault) ---"
    sudo dmesg -T 2>/dev/null | grep -iE "oom|kill|out of memory|segfault|sigsegv|general protection|cgroup" | tail -20 || true
    echo "--- memory ---"
    free -h
    echo "--- GPU memory ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
    echo "--- /dev/shm after crash ---"
    df -h /dev/shm 2>/dev/null || true
    echo "--- cgroup memory events ---"
    cat /sys/fs/cgroup/memory.events 2>/dev/null || cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null || true
    echo "--- Ray worker logs (last errors) ---"
    grep -r "SIGKILL\|SIGABRT\|SIGSEGV\|SYSTEM_ERROR\|RuntimeError\|NCCL" "$RAY_TMPDIR/session_latest/logs/" 2>/dev/null | tail -30 || true
    exit $EXIT_CODE
  fi

else
  # === Worker node: join Ray cluster and wait ===
  echo "=== Worker node (rank ${SKYPILOT_NODE_RANK}), joining Ray cluster at $ray_address ==="
  wait_for_ray "$ray_address"
  env -u RAY_ADDRESS ray start --address "$ray_address" --disable-usage-stats \
    --dashboard-agent-grpc-port "$RAY_DASH_AGENT_GRPC_PORT" \
    --dashboard-agent-listen-port "$RAY_DASH_AGENT_HTTP_PORT" \
    --runtime-env-agent-port "$RAY_RUNTIME_ENV_AGENT_PORT" \
    --metrics-export-port "$RAY_METRICS_EXPORT_PORT" \
    --min-worker-port "$RAY_WORKER_PORT_MIN" --max-worker-port "$RAY_WORKER_PORT_MAX" \
    --temp-dir="$RAY_TMPDIR"
  wait_for_ray "$ray_address"
  export RAY_ADDRESS="$ray_address"
  echo "Worker node joined. Sleeping..."
  sleep infinity
fi
