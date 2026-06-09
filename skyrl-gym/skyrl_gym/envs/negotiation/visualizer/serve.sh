#!/usr/bin/env bash
# Build the JSON (if needed) and serve the Deal-or-No-Deal dataset visualizer.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8791}"

if [ ! -f public/data/manifest.json ]; then
  echo "Building datasets -> JSON ..."
  python3 build.py
fi

echo "Serving visualizer at http://localhost:${PORT}"
cd public
python3 -m http.server "$PORT"
