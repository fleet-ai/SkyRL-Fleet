# SFT Warm-Start Ablation — Runbook

**Experiment spec:** `skyrl-gym/skyrl_gym/envs/negotiation/behavior_warmstart.md`  
**Goal:** Measure whether a CaSiNo SFT prior (installed before RL) delays or suppresses
*deception emergence* during DnD RL training, while leaving *value-leak emergence* unchanged.

---

## Conditions

| Arm | Init | RL data | RUN_ID | ckpt_path |
|---|---|---|---|---|
| **Baseline** | `Qwen/Qwen3.5-35B-A3B` (base) | DnD | `ws_baseline` | `$HOME/ckpts/ws_baseline` |
| **Warm (casino)** | `$HOME/exports/sft_casino` | DnD | `ws_warm_casino` | `$HOME/ckpts/ws_warm_casino` |
| **Warm (dnd)** | `$HOME/exports/sft_dnd` | DnD | `ws_warm_dnd` | `$HOME/ckpts/ws_warm_dnd` |

**CaSiNo** is the default and spec-canonical SFT source; **DnD** is provided as a comparison arm.
The spec uses CaSiNo as the primary warm-start corpus precisely because it is **out-of-domain**:
any deception divergence between arms is attributable to behavioral/negotiation prior, not
task-mechanics familiarity (CaSiNo items ≠ DnD items).

---

## Stage 0 — Prerequisites

### Hardware
Single node, 8×H200 (1 node × 8 GPUs).  
The SkyRL run script sets TP=2 → 4 inference engines (hardcoded via `NUM_INFERENCE_ENGINES`).

### Required env vars

```bash
# RL arms only (Stage 2); SFT stage (Stage 1) needs neither:
export WANDB_API_KEY="<your key>"
export OPENROUTER_API_KEY="<your key>"   # powers opponent LLM via litellm
```

### Repo setup
Ensure `.venv/bin/activate` exists (run `scripts/fleet-negotiation-setup.sh` if not).

---

## Stage 1 — SFT

**Purpose:** Train a message-level behavioral prior on human negotiation dialogues.  
SFT operates on natural-language turns only — no `<think>`/`<propose>` tags, no synthesized
structured output. The structured channel is left entirely untrained (see spec §Notes).

**No WANDB_API_KEY or OPENROUTER_API_KEY needed for this stage.**

### CaSiNo SFT (primary, out-of-domain — run this one)

```bash
SFT_DATASET=casino bash scripts/fleet-negotiation-sft-run.sh
```

Expected output checkpoint: **`$HOME/exports/sft_casino/`** (contains `config.json`, weights, tokenizer).  
Runtime: ~30–60 min on 8×H200 at default 2 epochs.

### DnD SFT (comparison arm — in-domain)

```bash
SFT_DATASET=dnd bash scripts/fleet-negotiation-sft-run.sh
```

Expected output checkpoint: **`$HOME/exports/sft_dnd/`**

### Adjusting SFT hyperparameters (optional)

```bash
# All overridable via env before launch:
EPOCHS=3 LR=5e-6 SFT_DATASET=casino bash scripts/fleet-negotiation-sft-run.sh
```

Key knobs: `EPOCHS` (default 2), `LR` (default 1e-5), `WEIGHT_DECAY`, `WARMUP_RATIO`,
`MICRO_BATCH_SIZE`, `GRAD_ACCUM`, `MAX_LENGTH` (default 4096).

---

## Stage 2 — RL (both arms)

Both arms run the standard GRPO negotiation trainer (`fleet-negotiation-35b-run.sh`)
via the thin wrapper `scripts/fleet-negotiation-warmstart-run.sh`.

### The golden rule: identical RL hyperparameters

**The ONLY differences between arms are `MODEL_PATH`, `RUN_ID`, and `ckpt_path`.**
Every RL knob — reward mode, penalties, thinking flag, LR, epochs, opponent, seed —
must be passed identically to both arms.

The cleanest way is a shared env block:

```bash
# Freeze all RL hyperparameters here — copy-paste identically to all three arm launches.
export ENABLE_THINKING=true          # required for value_leak_rate / think_nonempty_rate
export NUM_EPOCHS=30
export REWARD_MODE=outcome           # pure outcome reward — DO NOT use outcome_pareto here
export NEGOTIATION_DATASET=dnd       # RL training data — keep frozen across all arms
export OPPONENT_MODEL=openrouter/openai/gpt-4o-mini
# Add any other knobs you're varying here — and NEVER change them between arm launches.
```

### Reward isolation (required — do not shape behaviors into the reward)

Per spec §RL Training, the reward must be the **negotiation outcome only**. The behaviors
this ablation measures — deception and value-leak — must **never** be penalized in the
reward, or you suppress the very emergence you're trying to observe (and the warm vs.
baseline comparison is confounded).

The wrapper enforces this automatically: it forces, for **both** arms,

```
REWARD_MODE=outcome  DECEPTION_PENALTY=0  VALUE_LEAK_PENALTY=0  EMPTY_THINK_PENALTY=0
```

overriding `fleet-negotiation-35b-run.sh`'s nonzero guardrail defaults
(`deception=-0.1`, `value_leak=-0.05`, `empty_think=-0.02`). The banner prints the active
reward config at launch, and a warning fires if any of these are overridden.

The behaviors are still **tracked** with penalties off: `env.py` decouples detection from
shaping, so `deception_msgs` / `value_leak_msgs` / `think_nonempty_rate` are logged every
episode regardless of the (zero) penalty coefficient. Measured, not shaped.

Then launch each arm:

```bash
# --- Baseline (base model init) ---
ARM=baseline bash scripts/fleet-negotiation-warmstart-run.sh

# --- Warm, CaSiNo SFT (primary out-of-domain warm start) ---
ARM=warm SFT_DATASET=casino bash scripts/fleet-negotiation-warmstart-run.sh

# --- Warm, DnD SFT (comparison: in-domain warm start) ---
ARM=warm SFT_DATASET=dnd bash scripts/fleet-negotiation-warmstart-run.sh
```

The three arms can run in parallel on separate nodes or sequentially on the same node.

### Passing extra Hydra overrides

Any trailing positional arguments are forwarded as Hydra dotlist overrides to the RL
run script. They appear **after** the script's own hardcoded values; `OmegaConf.from_cli`
is last-wins, so they take precedence. Pass the **same trailing args** to all arms:

```bash
SHARED_HYDRA="trainer.algorithm.kl_loss_coef=0.05 trainer.epochs=25"

ARM=baseline bash scripts/fleet-negotiation-warmstart-run.sh $SHARED_HYDRA
ARM=warm SFT_DATASET=casino bash scripts/fleet-negotiation-warmstart-run.sh $SHARED_HYDRA
ARM=warm SFT_DATASET=dnd    bash scripts/fleet-negotiation-warmstart-run.sh $SHARED_HYDRA
```

### Specifying a custom SFT checkpoint path

```bash
ARM=warm SFT_DATASET=casino SFT_CKPT=/path/to/custom/sft_ckpt \
  bash scripts/fleet-negotiation-warmstart-run.sh
```

The script validates that `${SFT_CKPT}/config.json` exists and errors with a helpful
message if Stage 1 has not been run.

---

## Clobber safety and resume behavior

`fleet-negotiation-35b-run.sh` hardcodes:
```
trainer.ckpt_path="$HOME/ckpts/fleet_${MODEL_TAG}_35b_negotiation"
```
keyed on `MODEL_TAG` only — not `RUN_ID`. Two arms sharing the same `MODEL_TAG` (both use
`qwen35` by default) would **write to the same dir** and either clobber an in-progress
checkpoint or pick up the wrong arm's weights on resume.

The wrapper fixes this by appending `trainer.ckpt_path=<per-arm path>` as a **trailing**
Hydra arg. Because `OmegaConf.from_cli` merges the flat dotlist in order and last value
wins (see `skyrl/train/config/config.py` `SkyRLTrainConfig.from_cli_overrides` → `OmegaConf.from_cli` →
`OmegaConf.merge(base_cfg, overrides)`), the trailing value silently overrides the
hardcoded earlier one.

| Arm | `ckpt_path` | `resume_mode` |
|---|---|---|
| baseline | `$HOME/ckpts/ws_baseline` | `latest` |
| warm/casino | `$HOME/ckpts/ws_warm_casino` | `latest` |
| warm/dnd | `$HOME/ckpts/ws_warm_dnd` | `latest` |

**First launch (empty ckpt dir):** `resume_mode=latest` with no checkpoint found causes
the trainer to initialize from `trainer.policy.model.path` (= `MODEL_PATH`):
- Baseline: `Qwen/Qwen3.5-35B-A3B` (default in run script)
- Warm: `$HOME/exports/sft_<dataset>` (injected as `MODEL_PATH` env var by the wrapper)

**Crash-resume:** kill the job and re-run the same launch command. The trainer finds
the latest checkpoint in the per-arm `ckpt_path` and resumes from it.

---

## wandb run names and transcript directories

Run names follow the pattern set in `fleet-negotiation-35b-run.sh` line 220:
```
fleet_${MODEL_TAG}_35b_negotiation_${NEGOTIATION_DATASET}_${REWARD_MODE}_${RUN_ID}
```

With defaults (`MODEL_TAG=qwen35`, `NEGOTIATION_DATASET=dnd`, `REWARD_MODE=outcome`):

| Arm | wandb run name |
|---|---|
| baseline | `fleet_qwen35_35b_negotiation_dnd_outcome_ws_baseline` |
| warm/casino | `fleet_qwen35_35b_negotiation_dnd_outcome_ws_warm_casino` |
| warm/dnd | `fleet_qwen35_35b_negotiation_dnd_outcome_ws_warm_dnd` |

**wandb project:** `fleet-negotiation-grpo`

**Episode transcript directories** (full JSON-line episode logs with `<think>` content):
```
$HOME/exports/negotiation_transcripts/fleet_qwen35_35b_negotiation_dnd_outcome_ws_baseline/
$HOME/exports/negotiation_transcripts/fleet_qwen35_35b_negotiation_dnd_outcome_ws_warm_casino/
$HOME/exports/negotiation_transcripts/fleet_qwen35_35b_negotiation_dnd_outcome_ws_warm_dnd/
```
Set `TRANSCRIPT_DIR=` (empty) to disable transcript writing.

---

## Metrics and analysis

> Enable thinking (`ENABLE_THINKING=true`) for value-leak metrics to be meaningful.
> The `value_leak_rate` / `think_nonempty_rate` metrics require an active `<think>` channel.

### Primary emergence-rate metrics (logged to wandb at every eval step)

| Metric | wandb key | Definition |
|---|---|---|
| `deception_rate` | `eval/negotiation_dnd/deception_msgs` | Fraction of episodes where prose message commitment diverges from JSON `<propose>` |
| `value_leak_rate` | `eval/negotiation_dnd/value_leak_msgs` | Fraction of episodes where private valuation leaks from `<think>` into outgoing message |
| `think_nonempty_rate` | `eval/negotiation_dnd/think_nonempty_rate` | Fraction of turns with non-empty `<think>` block (collapse indicator) |

The **shape of the curves** matters more than endpoints. Plot all three per-arm on the same
axes vs. RL step to see emergence timing.

### Secondary metrics

| Metric | wandb key |
|---|---|
| Mean reward | `eval/negotiation_dnd/mean_reward` |
| Joint efficiency (if `REWARD_MODE=outcome_pareto`) | `eval/negotiation_dnd/pareto_score` |
| Empty-think penalty events | `eval/negotiation_dnd/empty_think_penalty` |

### Analysis checklist

1. **Deception emergence curve** — plot `deception_msgs` vs RL step for baseline vs warm/casino.
   Does warm start shift the curve right (delayed emergence) or flatten it (suppression)?
2. **Value-leak emergence curve** — same for `value_leak_msgs`. Spec predicts no meaningful
   difference between arms (value leak is structural/reward-visibility-driven, not behavioral).
3. **Asymmetry test** — if deception diverges between arms but value-leak does not, this
   mechanistically decomposes the two failure modes (spec §Analysis #3).
4. **DnD SFT comparison** — compare warm/dnd vs warm/casino to separate behavioral-prior
   effect from task-mechanics familiarity effect.

---

## Quick-reference command summary

```bash
# Stage 0: export keys
export WANDB_API_KEY="..."
export OPENROUTER_API_KEY="..."

# Stage 1: SFT (no API keys needed)
SFT_DATASET=casino bash scripts/fleet-negotiation-sft-run.sh
SFT_DATASET=dnd    bash scripts/fleet-negotiation-sft-run.sh   # optional comparison

# Stage 2: freeze RL hyperparameters
export ENABLE_THINKING=true
export NUM_EPOCHS=30
export REWARD_MODE=outcome
export NEGOTIATION_DATASET=dnd

# Stage 2: launch arms (identically configured)
ARM=baseline             bash scripts/fleet-negotiation-warmstart-run.sh
ARM=warm SFT_DATASET=casino bash scripts/fleet-negotiation-warmstart-run.sh
ARM=warm SFT_DATASET=dnd    bash scripts/fleet-negotiation-warmstart-run.sh
```
