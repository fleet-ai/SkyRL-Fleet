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

## Launch (cluster — the actual flow)

Everything the cluster needs ships in this repo: 24 game dirs pre-vendored under
`games/` (top-12 + placebo-12, committed) and both portfolio jsons under
`portfolios/`. The SkyPilot task does the rest (witness base dataset prep +
portfolio merge happen on the node):

```bash
cd SkyRL-Fleet
# arm B (treatment)
sky launch tasks/primitivbench-grpo.yaml \
  --env FLEET_API_KEY=... --env WANDB_API_KEY=... \
  --env AWS_ACCESS_KEY_ID=... --env AWS_SECRET_ACCESS_KEY=... \
  --env ARM=B -y --down
# arm C (active control) — same command, same TRAIN_SEED, only ARM changes
sky launch tasks/primitivbench-grpo.yaml ... --env ARM=C -y --down
```

The YAML keeps every recipe override identical to `tasks/witness-grpo.yaml`
except: entrypoint, train/val data, `trainer.seed=$TRAIN_SEED`, run_name.

### Pre-flight checklist (two decisions are YOURS, blocking)

1. **MODEL**: must equal the student of the reusable arm-A baseline run.
   Repo defaults say `Qwen/Qwen3.5-9B` (run_witness.sh / witness YAML), but
   `merge_fsdp_checkpoint.py` shows Qwen3.5-35B-A3B checkpoints also exist.
   Pick the one whose witness baseline run you intend to reuse; if no clean
   baseline exists, arm A must be (re)trained at the chosen size.
2. **GAME_IDS**: must equal arm A's witness training set. WARNING: the witness
   YAML default (`tw10 tw09 tw13 tw04`) overlaps the pre-registered held-out
   list (tw01/09/10/12). Either the baseline run used a different game set, or
   the held-out list must be re-derived as games ∉ train. Resolve before launch.
3. Push the current branch (`guanghan/b7-phase2-judge-rewards`) and confirm
   `workdir.ref` in the YAML points at it.
4. `TRAIN_SEED` identical across arms B and C (paired-seed design).
5. External text-reasoning suite for held-out domain 3: pick + decontam-check
   before eval (not needed for launch).

## Local dataset prep (already done; for reference / regeneration)

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
- `../../../tasks/primitivbench-grpo.yaml` — SkyPilot launch task (ARM=B|C)
- `smoke_env.py` — keyless local smoke (stubbed skyrl_gym; scripted player)
