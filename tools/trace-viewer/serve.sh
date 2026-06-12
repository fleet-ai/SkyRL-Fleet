#!/usr/bin/env bash
# Build the manifest (if missing) and serve the SkyRL trace viewer.
#
#   ./serve.sh [PORT] [HOST]
#
# Examples:
#   ./serve.sh                 # http://localhost:8792
#   ./serve.sh 9000            # custom port
#   ./serve.sh 8792 0.0.0.0    # bind all interfaces (share on a LAN/remote box)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8792}"
HOST="${2:-0.0.0.0}"

if [ ! -f public/data/manifest.json ]; then
  echo "No manifest found — generating sample data + manifest…"
  python3 gen_sample_data.py
  python3 build_manifest.py
fi

echo "Serving SkyRL trace viewer at http://${HOST}:${PORT}  (Ctrl-C to stop)"
cd public
exec python3 -m http.server "$PORT" --bind "$HOST"
