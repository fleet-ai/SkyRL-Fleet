"""Tinker auto-train configuration constants.

Parallel to config.py for the SkyPilot launcher. The Tinker pipeline runs in
Tinker cloud, so it does not need GPU pool / SkyPilot / GCP — just an API key.

The Tinker pipeline produces 90/10 train/holdout splits per (dataset, modality)
pair and surfaces a pre_pass_rate (base model on holdout) and post_pass_rate
(final model on holdout) for the user's "demonstrate learning" gate.
"""

from __future__ import annotations

import os

# State file for the Tinker pipeline; kept separate from .auto_train_state.json
# so the SkyRL and Tinker pipelines mark (dataset, modality) pairs as processed
# independently against the same shared dataset list.
TINKER_STATE_KEY = ".tinker_auto_train_state.json"

# Slack channel for Tinker-specific notifications. Falls back to the standard
# SLACK_AUTO_TRAIN_CHANNEL env var (and then SLACK_CHANNEL_DEFAULT) when unset.
TINKER_SLACK_CHANNEL_ENV = "SLACK_TINKER_AUTO_TRAIN_CHANNEL"

# Default base model. moonshotai/Kimi-K2.6 is the largest LoRA-trainable model
# in Tinker's catalog (1T params, 32B active MoE, VL-capable). See
# tinker-cookbook/tinker_cookbook/model_info.py for the canonical list.
DEFAULT_BASE_MODEL = os.environ.get("TINKER_BASE_MODEL", "moonshotai/Kimi-K2.6")

# LoRA rank for Tinker training. 16 matches scripts/fleet-tinker-tool-use-run.sh.
DEFAULT_LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "16"))

# Training step budget. Override via workflow input or env var.
DEFAULT_NUM_STEPS = int(os.environ.get("TINKER_NUM_STEPS", "50"))

# Holdout fraction. Per user: 90/10.
HOLDOUT_FRACTION = 0.10

# Deterministic split seed; recorded in eval_results JSON for reproducibility.
HOLDOUT_SEED = int(os.environ.get("TINKER_HOLDOUT_SEED", "42"))

# Where the launcher writes per-pair eval JSON. Picked up by the workflow's
# upload-artifact step.
EVAL_RESULTS_DIR = os.environ.get("TINKER_EVAL_RESULTS_DIR", "eval_results")

# Which modalities the Tinker pipeline currently launches. All three Fleet
# modalities work through main_fleet_tinker.py's FleetTaskEnv harness; VL
# coverage comes from the chosen base model (Kimi-K2.6 / Qwen3.5 VL family).
TINKER_MODALITY_SUPPORT: frozenset[str] = frozenset({"tool_use", "browser_use", "computer_use"})

# Required env vars for the Tinker pipeline. AWS_* is for S3 export + state.
REQUIRED_TINKER_ENV_VARS: tuple[str, ...] = (
    "TINKER_API_KEY",
    "FLEET_API_KEY",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)
