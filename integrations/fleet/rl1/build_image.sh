#!/usr/bin/env bash
# Build + push the trainer image ON the rl1 builder node (Job + dind).
#
# GitHub-hosted runners cannot build this image (disk); the builder node has
# local NVMe, native amd64, and cluster-side bandwidth. Repeat builds reuse
# the previous image as layer cache.
#
# Usage: ./build_image.sh [git-ref]   (default: current branch)
# Required env: GH_TOKEN (push ghcr; the repo itself is public)
set -euo pipefail

KUBE_CONTEXT="${KUBE_CONTEXT:-fleet-training-rl1-us-east-1}"
KUBECTL=(kubectl --context "$KUBE_CONTEXT" -n fleet-train-jobs)
REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
REF="${1:-$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)}"
SHA=$(git -C "$REPO_DIR" rev-parse --short=8 "$REF")
CACHE_TAG="${CACHE_TAG:-latest}"
: "${GH_TOKEN:?}"

"${KUBECTL[@]}" create secret generic img-build-secrets \
  --from-literal=GH_TOKEN="$GH_TOKEN" --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -

"${KUBECTL[@]}" delete job "skyrl-img-build-${SHA}" --ignore-not-found --wait=true

OPENENV_REF="${OPENENV_REF:-}"   # empty = Dockerfile's pinned default
export SHA REF CACHE_TAG OPENENV_REF
envsubst '$SHA $REF $CACHE_TAG $OPENENV_REF' < "$(dirname "$0")/build_job.yaml.tmpl" | "${KUBECTL[@]}" apply -f -

echo "build job: skyrl-img-build-${SHA}"
echo "logs:      kubectl --context $KUBE_CONTEXT -n fleet-train-jobs logs -f job/skyrl-img-build-${SHA} -c build"
echo "image:     ghcr.io/fleet-ai/skyrl-fleet/trainer:${SHA} (on success)"
