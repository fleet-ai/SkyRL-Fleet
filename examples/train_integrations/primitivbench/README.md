# PrimitivBench mini-flywheel (Sprint 2)

First empirical datapoint for **Score(g) = Δ(held-out reasoning | agent trained in g)** —
the Growth-Score protocol from the PrimitivBench two-ring design
(`game-creation-criterion/reports/2026-06-09_post-presentation-strategy.md` §5, D-14).

## Hypothesis (falsifiable)

The funnel-selected top-12 portfolio (`portfolio_v1.json`: 2× Stage-4 k≥2 passers +
sweet-spot/ICL-slope picks, placebo-hardened) produces a larger Δ on held-out
reasoning than an **episode-matched bottom-of-funnel portfolio**
(`portfolio_placebo_v1.json`: k=0 games incl. 4 trivial p=1.0). I.e., the funnel
ordering carries training value, not just "more game data".

## Arms (D-15, 2026-06-10)

| Arm                    | Train data                                | Source                                         |
| ---------------------- | ----------------------------------------- | ---------------------------------------------- |
| A (baseline)           | witness-13                                | existing witness parquet / prior runs          |
| B (treatment)          | witness-13 + top-12 portfolio             | `--merge-witness` → `armB_mixed.parquet`       |
| C (**active control**) | witness-13 + placebo-12 (episode-matched) | `--merge-witness` → `armC_mixed.parquet`       |
| S (optional)           | arm B data, random rewards                | spurious-reward floor (see student-model note) |

**Headline = Δ(B−C)** (quantity-matched, isolates content quality).
Δ(B−A) is reported descriptively only — it confounds quality with quantity
(criterion repo `reports/2026-06-10_growth-score-precedents-and-validity.md`, fatal-2).

**Recipe base = the CURRENT witness stack** (v5b7-phase2 line, NOT the legacy v3
`witness-grpo.yaml`): witness rows run through `env_class=witness_agent` (the full
ORAI agent bridged from arc-witness-agent, file_mounted) with the R-series reward
stack (R5_pre config: rubric ON via claude-haiku, plan-div OFF, judge OFF, secondary
fallback ON). PB rows carry `env_class=primitivbench` → the **generic text harness**
(render + enumerated actions, `<action>N</action>`), deliberately NOT the witness
scaffold, so no generator style is favored. Per-row env_class routing makes the mix
work in one run. **Paired seeds**: same `TRAIN_SEED` across arms B and C.

**Student model (D-18, updated 2026-06-11)**: arm A = the existing 9B R5_pre run
(wandb `mtezmmq0`): Qwen3.5-9B initialized from the witness SFT-v5 merged checkpoint.
Arms B/C must match both (MODEL + POLICY_CHECKPOINT_S3 in the YAML). A 35B-A3B arm A
(`p2_r5_35b_opus`) exists but was still stabilizing multi-node as of 2026-06-03; if it
becomes the canonical baseline, re-launch B/C with its exact config instead. Qwen
spurious-reward caveat stands (arXiv 2506.10947): arm C absorbs the bulk; add arm S
(random-reward floor) or a cross-family validation before v1.0.

## Held-out (3 domains, breadth-gated)

1. Witness held-out: tw01/09/10/12 — **identical to the current stack's VAL_GAME_IDS**,
   so this domain is evaluated automatically during training (eval_interval), and the
   former held-out/train overlap concern is moot: the v5b7 split already pre-registers it.
2. ARC-AGI-3 public subset locally + Kaggle submission (uncontaminated external)
3. One external text-reasoning suite (pick at run time; decontam-checked vs portfolio)

## Launch (cluster — the actual flow)

Everything ships with the launch: 24 game dirs pre-vendored under `games/` +
both portfolio jsons under `portfolios/` (in this repo, rsynced via the YAML's
local-workdir dev-loop), and arc-witness-agent / arc-witness-envs via file_mounts
(same as the witness YAMLs). Setup runs `fleet-witness-setup.sh` unchanged
(witness-8 dataset prep, SFT ckpt download, spot-resume), then adds the PB
portfolio prep + merge and a PB env smoke. Run delegates to `fleet-witness-run.sh`
with three trailing Hydra overrides (train/val data, trainer.seed) that win via `"$@"`.

```bash
cd SkyRL-Fleet
# arm B (treatment)
sky launch tasks/primitivbench-grpo.yaml \
  --env PB_ARM=B --env RUN_LABEL=pb_armB_s42 \
  --env WANDB_API_KEY=... --env OPENROUTER_API_KEY=... \
  --env AWS_ACCESS_KEY_ID=... --env AWS_SECRET_ACCESS_KEY=... \
  -y
# arm C (active control) — same TRAIN_SEED, only PB_ARM + RUN_LABEL change
sky launch tasks/primitivbench-grpo.yaml --env PB_ARM=C --env RUN_LABEL=pb_armC_s42 ... -y
```

Cheap preflight without GPUs-burn: add `--env SMOKE_ONLY=1` (validates mounts,
venv, env imports, reward hooks, then exits before the trainer).

### Pre-flight checklist

1. **Align with arm A (mtezmmq0)**: diff this YAML's envs against the mtezmmq0
   wandb config — LR / KL / entropy / NUM_EPOCHS / SECONDARY_MODEL especially.
   The defaults here are reconstructed from the 35B YAML's "matches 9B R5" comments,
   not read from the run itself.
2. `OPENROUTER_API_KEY` required (rubric reward + secondary fallback hit OpenRouter).
3. `TRAIN_SEED` identical across arms B and C (paired-seed design).
4. file_mounts paths point at your local arc-witness-agent / arc-witness-envs
   worktrees (same convention as witness-grpo-v5b7-phase2-r5-35b.yaml).
5. External text-reasoning suite for held-out domain 3: pick + decontam-check
   before eval (not needed for launch).

Registration note: `main_witness.py` now also registers `primitivbench` (lazy
entry_point, inert for witness-only datasets), so the standard witness entrypoint
drives the mixed run; `entrypoints/main_primitivbench.py` remains for standalone use.

## Local dataset prep (for reference / regeneration)

```bash
cd SkyRL-Fleet
python3 examples/train_integrations/primitivbench/prepare_primitivbench_dataset.py \
    --portfolio examples/train_integrations/primitivbench/portfolios/portfolio_v1.json \
    --games-src /Users/guanghan.ning/Documents/obsidian/research-survey/game-creation/05-pilot-study/orchestrator/curated_v2/games \
    --seeds-per-game 64 \
    --output examples/train_integrations/primitivbench/data/pb_train.parquet
# placebo: same with portfolio_placebo_v1.json (+ --val-output pb_placebo_val.parquet)
# --games-src defaults to the in-repo games/, so on the cluster vendoring is a no-op
```

`data_source = primitivbench/<game>` → SkyRL per-dataset eval splits give
**per-game metrics from a single run** (per-game Score attribution for free).

## Decision criteria (pre-registered, updated 2026-06-10 per D-15/D-16)

- **Δ(B−C) > 0** on held-out domains (report per-domain effect sizes with bootstrap CIs,
  not just sign counts) → the funnel selects **training value** → Growth Score protocol validated
- Δ(B−C) ≈ 0 but Δ(B−A) > 0 → extra game data helps regardless of funnel rank →
  quantity/diversity effect; funnel weights need recalibration — still publishable
- All Δ ≈ 0 → instance-density / curriculum diagnosis (single-level games; spec v2.1 motivation) — still publishable
- Funnel/proxy scores vs per-game contribution → the proxy↔truth calibration table (S45-难点③④ answer)
- Construct note (D-16): Growth Score measures **training value** of created games;
  intellectual-depth claims are carried by the Ring-1 k≥2 gate, not by this Δ.
- Pre-registered per-game prediction: PB-pilot-001__gemini-2.5-pro is arm B's top contributor.

## Files

- `env.py` — PrimitivBenchEnv (BaseTextEnv; generic text harness; verified against `skyrl_gym/envs/base_text_env.py` API)
- `prepare_primitivbench_dataset.py` — vendor games + emit parquet (arms B/C); cluster-safe (vendoring no-ops when games ship in-repo)
- `entrypoints/main_primitivbench.py` — registers witness + witness_agent + primitivbench envs
- `portfolios/` — portfolio_v1.json (arm B) + portfolio_placebo_v1.json (arm C), committed for cluster access
- `games/` — 24 vendored game.py dirs (top-12 + placebo-12), committed
- `../../../tasks/primitivbench-grpo.yaml` — SkyPilot launch task (PB_ARM=B|C; reuses fleet-witness-setup/run.sh)
- `smoke_env.py` — keyless local smoke (stubbed skyrl_gym; scripted player)
