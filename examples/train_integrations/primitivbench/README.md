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

| Arm | Train data | Source |
|---|---|---|
| A (baseline) | witness-13 | existing witness parquet / prior runs |
| B (treatment) | witness-13 + top-12 portfolio | `--merge-witness` → `armB_mixed.parquet` |
| C (**active control**) | witness-13 + placebo-12 (episode-matched) | `--merge-witness` → `armC_mixed.parquet` |
| S (optional) | arm B data, random rewards | spurious-reward floor (see student-model note) |

**Headline = Δ(B−C)** (quantity-matched, isolates content quality).
Δ(B−A) is reported descriptively only — it confounds quality with quantity
(criterion repo `reports/2026-06-10_growth-score-precedents-and-validity.md`, fatal-2).

Same recipe for all arms (model, GRPO hparams, steps), **paired seeds + identical
data order** across arms (within-seed paired Δ; variance reduction per common-random-numbers).
The PB env is a **generic text harness** (render + enumerated actions,
`<action>N</action>`) — deliberately NOT the witness semantic-ASCII scaffold, so no
generator style is favored.

**Student-model note (D-18)**: the recipe student is Qwen3.5-9B (Qwen family).
Qwen2.5 models show GRPO gains under *random* rewards (arXiv 2506.10947); arm C
already controls "any game RL happened", but for publication add either arm S
(random-reward floor) or a one-time cross-family validation (Llama/OLMo student,
generator-ranking Spearman) before v1.0.

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

# 2. arm C dataset (active control: placebo portfolio, identical seeds-per-game)
python3 examples/train_integrations/primitivbench/prepare_primitivbench_dataset.py \
    --portfolio  <pilot>/orchestrator/curated_v2/portfolio_placebo_v1.json \
    --games-src  <pilot>/orchestrator/curated_v2/games \
    --seeds-per-game 64 \
    --output examples/train_integrations/primitivbench/data/pb_placebo.parquet \
    --val-output examples/train_integrations/primitivbench/data/pb_placebo_val.parquet \
    --merge-witness <existing witness train parquet> \
    --merged-output examples/train_integrations/primitivbench/data/armC_mixed.parquet

# 3. train arms B and C (same command, same seed, swap the parquet)
python3 -m examples.train_integrations.primitivbench.entrypoints.main_primitivbench \
    data.train_data_path=examples/train_integrations/primitivbench/data/armB_mixed.parquet \
    <same overrides as the witness GRPO recipe>
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

- `env.py` — PrimitivBenchEnv (BaseTextEnv; generic text harness)
- `prepare_primitivbench_dataset.py` — vendor games + emit parquet (arms B/C)
- `entrypoints/main_primitivbench.py` — registers witness + primitivbench envs
- `games/` — vendored portfolio game.py files (created by the prep script)
- `smoke_env.py` — keyless local smoke (stubbed skyrl_gym; scripted player)
