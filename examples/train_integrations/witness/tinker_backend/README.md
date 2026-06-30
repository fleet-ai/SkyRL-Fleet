# Witness RL on the cloud-tinker backend (LoRA track, ported into Deniz's loop)

A second training backend for witness GRPO RL, alongside the slurm/SkyRL (FSDP+vLLM+Ray) path.
**Default stays slurm** — this is opt-in via `BACKEND=tinker`.

## Why this exists / what it is NOT
Tinker (Thinking Machines) is **LoRA-only** and the hosted cloud **cannot load our merged SFT
checkpoint**. So this is a **LoRA-on-stock-`Qwen/Qwen3.5-9B` track**, not a drop-in continuation of
the slurm exp-1/exp-2 (full-FT from the merged SFT). Frame it in the paper as a *portable-recipe
arm*. The win: runs on **tinker's GPUs** (no slurm/local GPUs) → scale out past an occupied cluster.
Full rationale: `research-artifacts/.../2026-06-23_tinker_backend_feasibility.md`.

## Architecture — reuses Deniz's Fleet×Tinker loop, swaps only the env
`main_witness_tinker.py` **imports and reuses** `integrations/fleet/entrypoints/main_fleet_tinker.py`
verbatim for everything env-agnostic — GRPO advantages (`compute_advantages_grpo`), the Datum build
(`prepare_training_data`: DAPO overlong filter, target-shift, the team's logprob/shape fixes),
metrics, and the `save_weights_for_sampler → sample → forward_backward → optim_step` cycle. The ONLY
witness-specific code: rollout collection via the existing **bridge** (`harness/agent_wrapper.py`
`AgentRolloutWrapper(mode="bridged")`) instead of `FleetTaskEnv`.

**Why per-ORAI-call datums (the one real design choice):** the witness ORAI loop is
**single-turn-per-call** — each call is a fresh `(system,user)` prompt, NOT an accumulating
conversation. So the policy sampled each response under its own single-turn context; to keep the
training context == the sampling context (correct importance ratios), each ORAI call becomes its own
datum (prompt = that single-turn chat), and the **trajectory-level GRPO advantage is broadcast across
a trajectory's calls** (outcome-supervised GRPO; trajectory return = sum of per-call
`RolloutEvent.reward`). GRPO groups = `n_samples_per_prompt` rollouts of the same game.

| file | role |
|---|---|
| `main_witness_tinker.py` | witness rollout (bridge) → Deniz's loop; per-call datums; the T1 injection seam |
| `train_sft.py` | SFT-as-LoRA on hosted Qwen3.5-9B → the warm-start `tinker://` checkpoint (loop-agnostic) |

## Prereqs
- `tinker` (0.22.3) + `tinker_cookbook` + `transformers` are in `arc-witness-agent/.venv`.
- `TINKER_API_KEY` (launcher reads `temp/tinker.md`).
- `PYTHONPATH` includes `arc-witness-agent` + `arc-witness-envs` + `SkyRL-Fleet` + this `witness/` dir (launcher sets it).
- SFT data (confirmed): `~/Documents/obsidian/research-artifacts/arc-witness-runs/sft_data/wp_mt_combined/sft_pairs_judged.jsonl`
  (402 rows, `{"messages":[system,user,assistant], metadata}` — matches `train_sft.py` `sft_format=messages`).

## Run (you run these — they hit the paid tinker API)
```bash
PY=~/Documents/obsidian/research-repos/arc-witness-agent/.venv/bin/python
cd ~/Documents/obsidian/research-repos/SkyRL-Fleet/examples/train_integrations/witness
export TINKER_API_KEY=…
export PYTHONPATH=…/arc-witness-agent:…/arc-witness-envs:…/SkyRL-Fleet:$PWD

# 1) SFT-as-LoRA warm-start → prints a tinker:// checkpoint path
$PY -m tinker_backend.train_sft \
  sft_jsonl=~/Documents/obsidian/research-artifacts/arc-witness-runs/sft_data/wp_mt_combined/sft_pairs_judged.jsonl \
  num_epochs=3 lora_rank=32   # 3 epochs matches the SkyRL v6 SFT recipe

# 2) TINY smoke first (cheap — validate the live seams below before a full run)
SFT_LORA_CKPT=tinker://<step> GAMESET=witness NUM_STEPS=2 GROUP_SIZE=2 MAX_LEVELS=1 \
  bash …/research-artifacts/arc-witness-runs/scripts/2026-06-23_witness_rl_tinker_launch.sh

# 3) Full GRPO from the warm-start (8-game witness baseline)
BACKEND=tinker GAMESET=witness INJECT_MODE=off SFT_LORA_CKPT=tinker://<step> \
  bash …/research-artifacts/arc-witness-runs/scripts/2026-06-22_witness_rl_r4_v6mt_9b_launch.sh

# 4) T1 oversight arms (the new experiment): A1-train (real GT rules) + A1-shuf (control)
BACKEND=tinker INJECT_MODE=rules SFT_LORA_CKPT=tinker://<step> bash …/2026-06-22_witness_rl_r4_v6mt_9b_launch.sh
BACKEND=tinker INJECT_MODE=shuf  SFT_LORA_CKPT=tinker://<step> bash …/2026-06-22_witness_rl_r4_v6mt_9b_launch.sh
```
`BACKEND=slurm` (or unset) → the existing slurm launcher, unchanged.

## T1 (curriculum GT-rule injection)
`main_witness_tinker.collect_witness_rollout` injects `oracle.rule_card(game)` into the agent at
episode start with decaying `p(step)=max(0, 1 − step/(0.4·max_steps))` (blueprint §6.2).
`--inject-mode rules` (A1-train) / `shuf` (A1-shuf, wrong-game card) / `off`. Eval always uses `off`
(L0-clean: `p=0` long before the end).

## Live-validation seams (run the TINY smoke first; tinker API is paid)
1. **Bridge concurrency.** `collect_batch` runs many `AgentRolloutWrapper(mode="bridged")` rollouts
   concurrently, each spawning a daemon agent thread + bridge Events; blocking bridge calls run in a
   `ThreadPoolExecutor`. Confirm no cross-rollout interference at `max_concurrent=8` (lower it if so).
2. **Catalog.** `service_client` + `create_lora_training_client(base_model="Qwen/Qwen3.5-9B")` — confirm
   `Qwen3.5-9B` is in the live catalog (`get_server_capabilities`); a 2026-06-11 note said the *35B*
   variant was retired.
3. **Credit assignment.** Per-call datum + trajectory-broadcast advantage is the defensible default;
   if you want per-step credit, change the advantage broadcast in `_traj_to_percall_rollouts`.
4. **`train_sft` builder return.** Returns the dataset; if the installed `supervised.train` expects a
   `(train, val)` tuple from the builder, return `(ds, None)`.
5. **Reward scale / `loss_fn`.** Default `importance_sampling`; the trajectory return is a sum of
   geometric+outcome per-call rewards — sanity-check magnitude on the smoke run.
