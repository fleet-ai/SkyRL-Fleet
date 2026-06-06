#!/bin/bash
# One-time setup for RunPod Slurm cluster access via SkyPilot.
#
# Remote setup (from your Mac):
#   SLURM_HOST=31.24.80.55 SLURM_PORT=13122 SLURM_USER=yourname bash scripts/setup-slurm.sh
#
# SLURM_USER sets your identity on the cluster. When not "root", the script
# creates a Linux user on the controller so squeue shows your name.
#
# All env vars (defaults in parens):
#   SLURM_HOST  (localhost)           Controller IP
#   SLURM_PORT  (22)                  Controller SSH port
#   SLURM_USER  (current user)        Your username (shows in squeue)
#   SLURM_SSH_KEY (~/.ssh/id_ed25519) Private key for SSH
#
# This script handles EVERYTHING needed for Slurm training:
#   1. Creates ~/.slurm/config (SSH connection to Slurm controller)
#   2. Ensures 'slurm' is in allowed_clouds in ~/.sky/config.yaml
#   3. Adds slurm.cluster_configs (workdir, gpu_partition_map) to ~/.sky/config.yaml
#   4. Restarts the SkyPilot API server to pick up the new config
#   5. Verifies connectivity and GPU detection
#   6. Configures InfiniBand (NCCL) on all cluster nodes
#
# The SkyPilot API server ignores allowed_clouds and cluster_configs from
# the project-level .sky.yaml, so these MUST be in ~/.sky/config.yaml.
set -euo pipefail

RUNPOD_CLUSTER_NAME="runpod-cluster-y8puzql3juawue"
# Connection defaults to the on-cluster login node, submitting as your own user
# (not root). All four are overridable via the env vars documented in the header.
SLURM_HOST="${SLURM_HOST:-localhost}"
SLURM_PORT="${SLURM_PORT:-22}"
SLURM_USER="${SLURM_USER:-$(whoami)}"
RUNPOD_KEY="${SLURM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
if [ ! -f "$RUNPOD_KEY" ]; then
  echo "ERROR: SSH key $RUNPOD_KEY not found."
  echo "Set SLURM_SSH_KEY to point to a valid private key for $SLURM_USER@$SLURM_HOST."
  exit 1
fi

# 1. Create Linux user on ALL nodes (if not root) so squeue shows your name
# and Slurm can chdir to /home/$USER on any compute node.
if [ "$SLURM_USER" != "root" ]; then
  echo "[1a/6] Creating user '$SLURM_USER' on all cluster nodes..."
  PUBKEY=$(cat "${RUNPOD_KEY}.pub" 2>/dev/null || ssh-keygen -y -f "$RUNPOD_KEY" 2>/dev/null || echo "")
  if [ -z "$PUBKEY" ]; then
    echo "ERROR: Could not read public key from ${RUNPOD_KEY}.pub or derive from $RUNPOD_KEY"
    exit 1
  fi
  ssh -o StrictHostKeyChecking=no -i "$RUNPOD_KEY" -p "$SLURM_PORT" root@"$SLURM_HOST" bash <<USEREOF
    # Create on controller
    id $SLURM_USER 2>/dev/null || useradd -m -s /bin/bash $SLURM_USER
    mkdir -p /home/$SLURM_USER/.ssh
    grep -qF '$PUBKEY' /home/$SLURM_USER/.ssh/authorized_keys 2>/dev/null || echo '$PUBKEY' >> /home/$SLURM_USER/.ssh/authorized_keys
    chmod 700 /home/$SLURM_USER/.ssh
    chmod 600 /home/$SLURM_USER/.ssh/authorized_keys
    chown -R $SLURM_USER:$SLURM_USER /home/$SLURM_USER/.ssh
    # Passwordless sudo (needed for CUDA toolkit installs during setup)
    echo '$SLURM_USER ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/$SLURM_USER
    chmod 440 /etc/sudoers.d/$SLURM_USER
    # Create on all compute nodes (Slurm needs /home/\$USER on every node)
    NODES=\$(sinfo -N -h -o '%N' 2>/dev/null | sort -u)
    for n in \$NODES; do
      srun --overlap -N1 --nodelist=\$n bash -c "id $SLURM_USER 2>/dev/null || useradd -m -s /bin/bash $SLURM_USER; chown -R $SLURM_USER:$SLURM_USER /home/$SLURM_USER 2>/dev/null; echo '$SLURM_USER ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/$SLURM_USER; chmod 440 /etc/sudoers.d/$SLURM_USER" 2>/dev/null || true
    done
    echo "User '$SLURM_USER' ready on all nodes"
USEREOF
fi

# 1b. SSH config for Slurm controller
echo "[1b/6] Setting up ~/.slurm/config ($SLURM_USER@$SLURM_HOST:$SLURM_PORT)..."
mkdir -p ~/.slurm
cat > ~/.slurm/config <<EOF
Host runpod-cluster $RUNPOD_CLUSTER_NAME
    HostName $SLURM_HOST
    Port $SLURM_PORT
    User $SLURM_USER
    IdentityFile $RUNPOD_KEY
    IdentitiesOnly yes
    BatchMode yes
    StrictHostKeyChecking no
EOF

# 1b. Make shared SkyPilot dirs writable by every user (not just root).
# /workspace is a shared filesystem. .sky_provision (provision scripts) and
# .sky_clusters (synced workdirs) get created by whoever launches first — often
# root — which then blocks other users' rsync during provisioning with
# "Permission denied". Sticky + world-writable (like /tmp) lets any user submit
# while still protecting each user's own files. Done over the root SSH
# connection because the dirs are root-owned.
echo "[1b/5] Making shared /workspace dirs writable across users..."
ssh -F ~/.slurm/config -l root "$RUNPOD_CLUSTER_NAME" '
  set -e
  # SkyPilot provisioning dirs: sticky + world-writable (like /tmp) so each user
  # keeps their own provision scripts / synced workdirs without clobbering others.
  mkdir -p /workspace/.sky_provision /workspace/.sky_clusters /workspace/clusters
  chmod 1777 /workspace/.sky_provision /workspace/.sky_clusters /workspace/clusters
  # Shared dataset cache: world-writable, NO sticky bit, so any user can
  # re-download and overwrite the version-pinned dataset files regardless of who
  # created them (the dataset content is identical per DATA_VERSION).
  mkdir -p /workspace/data/fleet
  chmod -R a+rwX /workspace/data
  stat -c "  %a %n" /workspace/.sky_provision /workspace/.sky_clusters /workspace/data
' || echo "  WARNING: could not set permissions (continuing) — fix manually as root: chmod 1777 /workspace/.sky_{provision,clusters}; chmod -R a+rwX /workspace/data"

# 2. Ensure slurm is in allowed_clouds
echo "[2/5] Checking allowed_clouds..."
if [ ! -f ~/.sky/config.yaml ]; then
  mkdir -p ~/.sky
  cat > ~/.sky/config.yaml <<'EOF'
allowed_clouds:
  - slurm
  - gcp
  - runpod
EOF
  echo "  Created ~/.sky/config.yaml with slurm enabled"
elif ! grep -q '^\s*- slurm' ~/.sky/config.yaml; then
  # Add slurm to allowed_clouds list
  sed -i.bak '/allowed_clouds:/a\  - slurm' ~/.sky/config.yaml && rm -f ~/.sky/config.yaml.bak
  echo "  Added slurm to allowed_clouds"
else
  echo "  slurm already in allowed_clouds"
fi

# 3. Add slurm.cluster_configs
echo "[3/5] Adding slurm.cluster_configs..."
if grep -q 'cluster_configs:' ~/.sky/config.yaml 2>/dev/null && grep -q "$RUNPOD_CLUSTER_NAME:" ~/.sky/config.yaml 2>/dev/null; then
  echo "  cluster_configs.$RUNPOD_CLUSTER_NAME already exists — skipping"
elif grep -q '^slurm:' ~/.sky/config.yaml 2>/dev/null; then
  # slurm section exists but no cluster_configs — append under it
  sed -i.bak "/^slurm:/a\\  provision_timeout: -1\\
  allowed_clusters:\\
    - $RUNPOD_CLUSTER_NAME\\
  cluster_configs:\\
    $RUNPOD_CLUSTER_NAME:\\
      workdir: /workspace\\
      gpu_partition_map:\\
        H200: gpu" ~/.sky/config.yaml && rm -f ~/.sky/config.yaml.bak
  echo "  Added cluster_configs under existing slurm section"
else
  cat >> ~/.sky/config.yaml <<EOF

slurm:
  allowed_clusters:
    - $RUNPOD_CLUSTER_NAME
  provision_timeout: -1
  cluster_configs:
    $RUNPOD_CLUSTER_NAME:
      workdir: /workspace
      gpu_partition_map:
        H200: gpu
EOF
  echo "  Added slurm.cluster_configs"
fi

# 4. Restart API server to pick up config
echo "[4/5] Restarting SkyPilot API server..."
sky api stop 2>/dev/null || true
sky api start 2>&1 | tail -3

# 5. Verify
echo "[5/6] Verifying..."
echo "  SSH connectivity..."
ssh -F ~/.slurm/config "$RUNPOD_CLUSTER_NAME" "sinfo -N" || { echo "FAIL: Cannot connect to Slurm controller"; exit 1; }
echo "  SkyPilot detection..."
sky check slurm 2>&1 | grep -E "Slurm:|enabled|disabled"

# 6. Configure InfiniBand (NCCL) on all cluster nodes
# Detects active IB HCAs and training interface, then writes NCCL settings
# to /etc/environment on every node so all Slurm processes inherit them.
echo "[6/6] Configuring InfiniBand (consistent cross-node intersection)..."

# Helper that computes the cross-node IB-HCA intersection. Lives next to this script and
# must be on the shared FS so every allocated node can execute it.
_IB_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ib-hca-intersection.sh"

# Bootstrap NIC (consistent across nodes): highest-MTU non-virtual interface.
SOCK_IFACE=$(ssh -F ~/.slurm/config -l root "$RUNPOD_CLUSTER_NAME" bash << 'EOF'
ip -o link show | awk '!/lo|docker|veth|br-/ && /mtu/ {
  split($2,a,"@"); for(i=1;i<=NF;i++) if($i=="mtu") m=$(i+1)
  if(m+0>M+0) { M=m; I=a[1] }
} END { print I }'
EOF
)
echo "  Interface: ${SOCK_IFACE:-none}"

# Device names (mlx5_N) are NOT identical across nodes here: mlx5_3 is a live IB port on some
# nodes while on others mlx5_3 is DOWN and mlx5_5 is live instead. Broadcasting one node's list
# names a dead NIC on the others, and per-node detection makes ranks DISAGREE -- either way NCCL
# DEADLOCKS the first cross-node collective. So every node computes the INTERSECTION (NICs live
# on ALL nodes) via the shared-NFS barrier helper and writes that one consistent list. Per-job
# launches recompute the per-allocation intersection in fleet-common-run.sh (keeping all rails
# when the allocation is homogeneous); this is the safe, always-consistent static floor.
ssh -F ~/.slurm/config -l root "$RUNPOD_CLUSTER_NAME" bash << EOF
N=\$(sinfo -h -o '%D' | awk '{s+=\$1} END{print s}')
export BARRIER=/workspace/.sky_ib_setup/setup-\$(date +%s)
echo "  Applying consistent NCCL_IB_HCA to \${N} node(s)..."
srun --ntasks="\$N" --ntasks-per-node=1 bash -c '
  HCA=\$(bash "${_IB_HELPER}" "\$BARRIER" "\$SLURM_NTASKS")
  sed -i "/^NCCL_/d" /etc/environment
  cat >> /etc/environment << ENVEOF
NCCL_IB_HCA=\$HCA
NCCL_IB_DISABLE=0
NCCL_IB_GDR_LEVEL=2
NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_TIMEOUT=23
NCCL_PROTO=LL128
NCCL_SOCKET_IFNAME=${SOCK_IFACE}
ENVEOF
  printf "%s\n" /workspace/rdma_libs /workspace/rdma_libs/libibverbs \\
    > /etc/ld.so.conf.d/rdma.conf
  ldconfig 2>/dev/null || true
  printf "  [%s] NCCL_IB_HCA=%s\n" "\$(hostname)" "\$HCA"
'
EOF

echo ""
echo "Done. Run 'sky show-gpus --cloud slurm' to see available GPUs."
