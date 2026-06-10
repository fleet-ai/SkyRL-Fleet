# PrimitivBench mini-flywheel (Sprint 2)

First empirical datapoint for **Score(g) = Δ(held-out reasoning | agent trained in g)** —
the Growth-Score protocol from the PrimitivBench two-ring design
(`game-creation-criterion/reports/2026-06-09_post-presentation-strategy.md` §5, D-14).

## Hypothesis (falsifiable)

Adding the funnel-selected 12-game portfolio (`portfolio_v1.json`: 2× Stage-4 k≥2
passers + sweet-spot/ICL-slope picks, placebo-hardened) to the witness-13 training
base produces a measurable Δ on held-out reasoning vs the witness-13 baseline.

## Arms

| Arm | Train data | Source |
|---|---|---|
| A (baseline) | witness-13 | existing witness parquet / prior runs |
| B (treatment) | witness-13 + portfolio-12 | `--merge-witness` output `armB_mixed.parquet` |
| C (optional) | portfolio-12 only | `pb_train.parquet` |

Same recipe for all arms (model, GRPO hparams, steps). The PB env is a **generic
text harness** (render + enumerated actions, `<action>N</action>`) — deliberately
NOT the witness semantic-ASCII scaffold, so no generator style is favored.

## Held-out (3 domains, breadth-gated)

1. Witness held-out: tw01/09/10/12 (pre-registered, workshop-paper protocol)
2. ARC-AGI-3 public subset locally + Kaggle submission (uncontaminated external)
3. One external text-reasoning suite (pick at run time; decontam-checked vs portfolio)

## Quickstart

```bash
cd SkyRL-Fleet
# 1. dataset (vendors game files + emits parquet; seeds 1000+ disjoint from proxy battery's 100-204)
python3 examples/train_integrations/primitivbench/prepare_primitivbench_dataset.py \
    --portfolio  <pilot>/orchestrator/curated_v2/portfolio_v1.json \
    --games-src  <pilot>/orchestrator/curated_v2/games \
    --seeds-per-game 64 \
    --output examples/train_integrations/primitivbench/data/pb_train.parquet \
    --merge-witness <existing witness train parquet>

# 2. train arm B (mixed rows; each row carries its env_class)
python3 -m examples.train_integrations.primitivbench.entrypoints.main_primitivbench \
    data.train_data_path=examples/train_integrations/primitivbench/data/armB_mixed.parquet \
    <same overrides as the witness GRPO recipe>
```

`data_source = primitivbench/<game>` → SkyRL per-dataset eval splits give
**per-game metrics from a single run** (per-game Score attribution for free).

## Decision criteria (pre-registered)

- Δ(B−A) > 0 on ≥2 of 3 held-out domains → portfolio adds value → Growth Score protocol validated
- Δ ≈ 0 → instance-density / curriculum diagnosis (single-level games; spec v2.1 motivation) — still publishable
- Funnel/proxy scores vs per-game contribution → the proxy↔truth calibration table (S45-难点③④ answer)

## Files

- `env.py` — PrimitivBenchEnv (BaseTextEnv; generic text harness)
- `prepare_primitivbench_dataset.py` — vendor games + emit parquet (arms B/C)
- `entrypoints/main_primitivbench.py` — registers witness + primitivbench envs
- `games/` — vendored portfolio game.py files (created by the prep script)
- `smoke_env.py` — keyless local smoke (stubbed skyrl_gym; scripted player)
