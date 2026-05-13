#!/bin/bash
# One-time setup for RunPod Slurm cluster access via SkyPilot.
# Requires: ~/.ssh/runpod_key (ask team for the key)
#
# This script:
#   1. Creates ~/.slurm/config (SSH connection to Slurm controller)
#   2. Adds slurm cluster_configs to ~/.sky/config.yaml (workdir, gpu_partition_map)
#   3. Restarts the SkyPilot API server to pick up the new config
#   4. Verifies connectivity and GPU detection
set -euo pipefail

if [ ! -f ~/.ssh/runpod_key ]; then
  echo "ERROR: ~/.ssh/runpod_key not found. Ask the team for the RunPod SSH key."
  exit 1
fi

# 1. SSH config for Slurm controller
echo "Setting up ~/.slurm/config..."
mkdir -p ~/.slurm
cat > ~/.slurm/config <<'EOF'
Host runpod-cluster
    HostName 31.24.80.22
    Port 16198
    User root
    IdentityFile ~/.ssh/runpod_key
    StrictHostKeyChecking no
EOF

# 2. Add Slurm cluster_configs to ~/.sky/config.yaml
# The API server reads from ~/.sky/config.yaml, not .sky.yaml project config,
# so cluster_configs must be in the user config.
echo "Updating ~/.sky/config.yaml..."
if [ -f ~/.sky/config.yaml ]; then
  if grep -q 'slurm:' ~/.sky/config.yaml; then
    echo "  slurm section already exists in ~/.sky/config.yaml — skipping."
    echo "  Ensure it has cluster_configs.runpod-cluster.workdir and gpu_partition_map."
  else
    cat >> ~/.sky/config.yaml <<'EOF'

slurm:
  cluster_configs:
    runpod-cluster:
      workdir: /workspace
      gpu_partition_map:
        H200: gpu
EOF
    echo "  Added slurm.cluster_configs to ~/.sky/config.yaml"
  fi
  # Ensure slurm is in allowed_clouds
  if ! grep -q 'slurm' ~/.sky/config.yaml; then
    echo "  WARNING: 'slurm' not in allowed_clouds. Add it manually."
  fi
else
  echo "  WARNING: ~/.sky/config.yaml not found. Create it with 'slurm' in allowed_clouds."
fi

# 3. Restart API server to pick up config
echo "Restarting SkyPilot API server..."
sky api stop 2>/dev/null || true
sky api start 2>&1 | tail -3

# 4. Verify
echo "Verifying SSH connectivity..."
ssh -F ~/.slurm/config runpod-cluster "sinfo -N" || { echo "FAIL: Cannot connect to Slurm controller"; exit 1; }

echo "Verifying SkyPilot detection..."
sky check slurm 2>&1 | tail -5

echo "Done. Run 'sky show-gpus --cloud slurm' to see available GPUs."
