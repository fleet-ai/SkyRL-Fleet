#!/usr/bin/env bash
# Local preflight for the 35B Witness SkyPilot config.
#
# This does not launch SkyPilot or load the model. It verifies that the local
# file_mount sources point at the latest harness worktrees and that the YAML is
# parseable on this machine.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="tasks/witness-grpo-v5b7-phase2-r34-35b.yaml"
AGENT_DIR="${ARC_WITNESS_AGENT_DIR:-$HOME/arc-witness-agent-b7}"
ENVS_DIR="${WITNESS_ENVS_DIR:-$HOME/arc-witness-envs-b7}"

test -f "$CONFIG" || { echo "missing config: $CONFIG" >&2; exit 1; }
test -f "$AGENT_DIR/agent/core.py" || { echo "missing agent worktree: $AGENT_DIR" >&2; exit 1; }
test -f "$ENVS_DIR/witness_grid.py" || { echo "missing envs worktree: $ENVS_DIR" >&2; exit 1; }

if command -v ruby >/dev/null 2>&1; then
  ruby -e 'require "yaml"; y=YAML.load_file(ARGV[0]); abort("bad model") unless y.dig("envs", "MODEL") == "Qwen/Qwen3.5-35B-A3B"; puts "[ok] YAML parses: #{y["name"]}"' "$CONFIG"
else
  echo "[warn] ruby unavailable; skipping YAML parse"
fi

rg -q 'def compute_plan_diversity_penalty' "$AGENT_DIR/agent/runtime/process_reward.py"
rg -q 'def compute_rule_judge_reward' "$AGENT_DIR/agent/runtime/process_reward.py"
rg -q 'def compute_rubric_reward' "$AGENT_DIR/agent/runtime/process_reward.py"
rg -q 'ENABLE_SECONDARY_PROVIDER_FALLBACK' "$AGENT_DIR/agent/llm/client.py"
rg -q 'AGENT_MLLM_ENABLED' "$AGENT_DIR/agent/core.py"
echo "[ok] latest agent reward hooks + LLM fallback symbols present"

if grep -q '<plan>1,4,2,5,3</plan>' "$AGENT_DIR/agent/decision/meta_reasoning.py"; then
  echo "prompt example still present in $AGENT_DIR; Phase 2/2b expects it removed" >&2
  exit 1
fi
echo "[ok] prompt anchor removed"

echo "[ok] local preflight complete"
