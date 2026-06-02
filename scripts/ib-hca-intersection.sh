#!/bin/bash
# Print the comma-separated set of IB HCAs that are ACTIVE InfiniBand (>=200 Gb/s) on
# EVERY node of the current allocation — i.e. the cross-node INTERSECTION.
#
# Why: device names (mlx5_N) are NOT identical across nodes on this cluster (e.g. mlx5_3 is
# a live IB port on some nodes while mlx5_5 is live on others, with the other DOWN). NCCL
# DEADLOCKS the first cross-node collective if ranks advertise different NCCL_IB_HCA lists,
# OR if a shared list names a NIC that's down on some node. Using only the HCAs live on ALL
# nodes gives one consistent, all-live list.
#
# Run ONE instance per node (e.g. via `srun --ntasks-per-node=1`). Each node publishes its
# own active set to a shared (NFS) barrier dir, waits for all peers, then every node prints
# the identical intersection.
#
# Usage: ib-hca-intersection.sh <shared_nfs_dir> <num_nodes>
# Never abort mid-detection: the detection loop's last (non-IB) device returns non-zero, and a
# caller running under `set -e`/`pipefail` (e.g. fleet-common-run.sh) would otherwise be killed.
set +e +o pipefail
set -u
D=${1:?need shared dir}; NN=${2:-1}
mkdir -p "$D"

self=$(for d in $(ls /sys/class/infiniband/ 2>/dev/null | sort); do
  s=$(cat "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null)
  l=$(cat "/sys/class/infiniband/$d/ports/1/link_layer" 2>/dev/null)
  r=$(cat "/sys/class/infiniband/$d/ports/1/rate" 2>/dev/null | awk '{print $1+0}')
  [ "$s" = "4: ACTIVE" ] && [ "$l" = "InfiniBand" ] && [ "${r:-0}" -ge 200 ] && echo "$d"
done | sort -u)
printf '%s\n' "$self" > "$D/$(hostname).hca"

# Barrier: wait (up to 120s) for every node to publish its set.
for _ in $(seq 1 120); do
  [ "$(ls "$D"/*.hca 2>/dev/null | wc -l)" -ge "$NN" ] && break
  sleep 1
done

common=""
for f in "$D"/*.hca; do
  if [ -z "$common" ]; then common=$(sort -u "$f")
  else common=$(comm -12 <(printf '%s\n' "$common") <(sort -u "$f")); fi
done
printf '%s\n' "$common" | sed '/^$/d' | paste -sd,
