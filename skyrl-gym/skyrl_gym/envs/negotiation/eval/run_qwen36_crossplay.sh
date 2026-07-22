#!/usr/bin/env bash
set -uo pipefail
set -a; source /workspace/allie/.env 2>/dev/null; set +a
VENV=/workspace/allie/skyrl-neg-wt/.venv/bin/python
cd /workspace/allie/skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/negotiation/eval
common=(--policy-model qwen/qwen3.6-27b --policy-label Qwen3.6-27b
  --policy-base-url https://openrouter.ai/api/v1 --policy-only
  --protocol single --match-train-sampling
  --dataset dnd --split val --n 16 --max-turns 6 --seed 1 --concurrency 8)
for cond in can_ask:canask can_ask_modified:canaskmod deception:deception; do
  elicit="${cond%%:*}"; tag="${cond##*:}"
  echo "=== $(date +%H:%M:%S) starting elicit=$elicit -> qwen36_$tag ==="
  $VENV run_crossplay.py "${common[@]}" --elicit "$elicit" \
    --out-prefix "crossplay_matrix_qwen36_$tag"
  echo "=== $(date +%H:%M:%S) done $tag ==="
done
echo "ALL_DONE"
