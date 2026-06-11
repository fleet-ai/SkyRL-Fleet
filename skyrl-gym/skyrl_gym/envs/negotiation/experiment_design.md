# Experiment Design — Pareto Reward × Preference Elicitation (negotiation RLVR)

Purpose: test whether we can lift **integrative efficiency** (Pareto / joint-efficiency) in
self-play item-division negotiation, and *attribute* any gain to the right cause — a missing
**incentive** (reward doesn't target the frontier) vs. missing **information** (agents never learn
each other's private values). Training is GRPO on Qwen3.5-9B via `scripts/fleet-negotiation-9b-run.sh`.

## Background — what the eval traces already told us

From `eval/REPORT.md` ("Did it not think that far, or is the thinking faulty?"), an autopsy of the 9
non-Pareto agreements in the 35B *thinking* run (chain-of-thought captured in
`eval/results/qwen_qwen3.5-35b-a3b_dnd_val_single-think_n20.json`, field `transcript[].thinking`):

- The per-side reasoning is **correct** — agents reliably compute their own scores. The Pareto gap
  is **not** a reasoning-depth failure.
- **Proposer solves the wrong problem (dominant, 9/9):** every proposer notes *"I don't know their
  values"* and defaults to a fair/even split (8/9). It optimizes own-score against an assumed-symmetric
  partner and **never elicits or signals values**, so it can't route items to whoever values them most.
- **Accepter thinks far enough but won't act (acceptance tax, 7/9 spot a better deal, 8/9 accept anyway):**
  it rubber-stamps to avoid no-deal (= 0 reward) within the turn budget.
- **Execution slip (rare):** correct thinking, inverted `<propose>` tag.

**Key mechanism (the crux of this experiment).** In the single-proposer protocol the proposer both
*controls* the split and, after eliciting, would *know* both value vectors. But an own-score proposer
will only **dump items it values at zero** (free Pareto move) and will **hoard items it values
low-but-positive even when the partner values them highly** (it only needs to clear the accepter's
> 0 reservation). Therefore:

- **Elicitation supplies the information** (fixes "didn't think that far").
- **The Pareto/joint term supplies the incentive** to give away the low-but-positive items the
  proposer would otherwise hoard (converts information into efficiency).
- **Neither alone closes the gap — the interaction is the mechanism.**

## Hypotheses

- **H1 (incentive-only is weak):** `outcome+pareto` *without* elicitation gives only a small Pareto gain
  (free zero-value dumping ± an emergent self-play "handshake") and risks a small agreement-rate dip.
- **H2 (information needs incentive):** elicitation *without* the Pareto reward mostly yields a better
  **extractor** — own-score up, splits more lopsided (equity gap widens), Pareto roughly flat.
- **H3 (interaction):** `outcome+pareto + elicitation` is the only cell that moves Pareto/joint-eff
  substantially. The elicitation factor has the larger main effect; there is a positive interaction.
- **H4 (one-sided ≈ two-sided in self-play, diverges in cross-play):** since the single-proposer
  proposer has info + control, one-sided elicitation ≈ two-sided in self-play; two-sided pulls ahead
  in **cross-play** (a partner-agnostic disclosure norm generalizes; a self-play handshake does not).

## Factors and runs

Two factors:

- **Reward** = `{outcome, outcome+pareto}`  (`REWARD_MODE=outcome | outcome_pareto`, already implemented)
- **Elicitation** = `{none, one-sided, two-sided}`  (**not yet implemented — see below**)
  - *one-sided*: only the proposer probes / the accepter discloses → proposer learns the partner's values.
  - *two-sided*: mutual disclosure → both sides learn the other's values.

### Core 4 runs (as requested)

| # | Run name | Reward | Elicitation | Tests |
|---|---|---|---|---|
| R0 | `outcome` (baseline) | outcome | none | current floor |
| R1 | `outcome_pareto` | outcome+pareto | none | H1 (incentive only) |
| R2 | `elicit_one_sided` | outcome+pareto | one-sided | proposer-info is enough? |
| R3 | `elicit_two_sided` | outcome+pareto | two-sided | + accepter policing / cross-play |

> **Design note (read before launching).** As written, R0→R1 isolates the Pareto reward, but
> R1→R2→R3 layers elicitation *on top of* the Pareto reward, so the elicitation arms are a *ladder*,
> not a clean factorial. To cleanly separate the **main effects + interaction** (H2/H3), run the full
> 2×2 instead: `{outcome, outcome+pareto} × {none, two-sided}` (4 cells), then add `one-sided`
> as a targeted 5th probe for H4. **Recommended:** do the clean 2×2 as the primary result and treat
> one-sided as the robustness follow-up. If compute is tight, the Core-4 ladder is acceptable but
> note R1→R2 confounds "added elicitation" with nothing else (reward held fixed), which is fine, while
> "is the Pareto reward necessary for elicitation to work" then needs an `outcome + two-sided` cell.

## What exists vs. what to build

**Exists (no work):**
- `REWARD_MODE=outcome` and `REWARD_MODE=outcome_pareto` in `env.py`
  (`reward = you_norm + pareto_coef * pareto_bonus` on agreement; `PARETO_COEF` default 0.5).
- `joint_efficiency` and `pareto` are already computed and logged per episode.

**Recommended reward refinement (small):**
- The current `pareto_bonus` is **binary** (1.0 iff on the enumerated frontier). Binary is sparse and
  orthogonal to slice size (it can reward a lopsided-but-frontier split). Add a **continuous** variant
  `REWARD_MODE=outcome_jointeff` → `reward = you_norm + coef * joint_efficiency` (value already on
  `out.joint_efficiency`). Prefer this as the "pareto" arm; keep it on agreed deals only so the
  no-deal=0 deterrent is preserved. Sweep `coef ∈ {0.25, 0.5, 1.0}`.

**To build — preference elicitation (the main implementation task):**
- Add an elicitation phase to the negotiation protocol in `prompts.py` (+ `env.py` if structured).
  Two implementation tiers, in increasing fidelity:
  1. **Prompt-induced (minimal viable):** modify the system prompt so the proposer must *ask* for the
     partner's priorities (one-sided) or both sides *state* their priorities (two-sided) before any
     `<propose>`. No env mechanics change; relies on the dialogue. Cheapest; start here.
  2. **Structured / env-mediated truthful disclosure (cleaner signal):** add a protocol tag
     (`<ask>` / `<values>`) and have the **env inject the true values** when queried, so the info
     channel is enforced and *honest*. This isolates the information effect from the strategic-honesty
     problem (see Risks). Adds one non-decision turn; account for it in `MAX_TURNS`.
- **Decision to surface (see Open decisions):** is disclosure **truthful/enforced** or **strategic**
  (agents may misrepresent)? Self-play under own-score reward will likely learn to *lie* — interesting
  to study, but a confound for "does elicitation raise efficiency." Recommend the env-mediated
  **truthful** variant for the headline runs, with strategic disclosure as a follow-up.

## How to launch each cell

Base script: `scripts/fleet-negotiation-9b-run.sh` (Qwen3.5-9B, GRPO, dnd — dnd carries the signal;
casino saturates). Override via env vars / hydra. Examples:

```bash
# R0 baseline
REWARD_MODE=outcome RUN_ID=r0_outcome bash scripts/fleet-negotiation-9b-run.sh

# R1 incentive only (prefer continuous joint-eff once added)
REWARD_MODE=outcome_pareto PARETO_COEF=0.5 RUN_ID=r1_pareto bash scripts/fleet-negotiation-9b-run.sh

# R2 / R3 add elicitation (new env flag to be implemented, e.g. NEGOTIATION_ELICIT)
REWARD_MODE=outcome_pareto NEGOTIATION_ELICIT=one_sided RUN_ID=r2_elicit1 bash scripts/fleet-negotiation-9b-run.sh
REWARD_MODE=outcome_pareto NEGOTIATION_ELICIT=two_sided RUN_ID=r3_elicit2 bash scripts/fleet-negotiation-9b-run.sh
```

- Keep `NEGOTIATION_DATASET=dnd`, `NEGOTIATION_PROTOCOL=single`, opponent fixed (`OPPONENT_MODEL`,
  default `openrouter/openai/gpt-4o-mini`) across all cells.
- If elicitation adds a disclosure turn, bump `MAX_TURNS` consistently and re-run
  `prepare_dataset.py --max_turns` (the script already wires this).
- **Thinking stays OFF for the 9B training runs** (budget starvation, see REPORT.md). The thinking
  analysis above is diagnostic only; do not flip `enable_thinking` for these training runs unless you
  also raise the generation budget (see REPORT.md "thinking ON vs OFF" / the `MAX_GENERATE_LENGTH`
  guidance — ~8192 with thinking on).

## Metrics

Primary (the thing we're trying to move):
- **joint_efficiency** (continuous) and **pareto rate** (of agreements + overall).

Headline-but-secondary:
- **avg_outcome_reward** (own normalized score), **agreement_rate**.

Guardrails / diagnostics (these reveal the failure modes above — log every cell):
- **no_deal_rate** — efficiency gains can come out of agreement rate (acceptance tax). Watch for dips,
  esp. in R1 (H1).
- **equity gap** = |you_score − them_score| — H2 predicts elicitation-without-pareto *widens* this.
- **value-capture** (proposer vs accepter, see `analyze_value_alignment.py`) — the accepter side is
  where value leaks; elicitation should lift the accepter's capture.
- For elicitation arms: **disclosure honesty** (if strategic disclosure allowed) and how often the
  proposal actually uses the disclosed values.

## Evaluation plan (do not skip)

- **Cross-play, not just self-play.** A Pareto/joint reward in self-play can learn a collusive
  *handshake* that maximizes joint only against itself and fails to transfer. Evaluate every checkpoint
  on the cross-play matrix (`eval/run_crossplay.py`) against held-out partners (the frontier + open
  models already in the matrix). H4 lives here: expect one-sided ≈ two-sided in self-play but two-sided
  ahead in cross-play.
- Re-export trained-policy transcripts to the visualizer (`eval/export_to_viz.py`) and spot-read the
  thinking/dialogue to confirm the *mechanism* (are agents actually eliciting + routing?), not just the
  aggregate numbers.

## Predicted results (so the run is interpretable / falsifiable)

| Cell | Pareto / joint-eff | Outcome | Equity gap | Agreement |
|---|---|---|---|---|
| R0 outcome | baseline | baseline | baseline | ~100% |
| R1 +pareto, no elicit | small ↑ | ≈ / slight ↓ | ≈ | possible slight ↓ |
| (outcome + elicit, no pareto) | ≈ flat | ↑ | **↑ (extractor)** | ~100% |
| R2/R3 +pareto + elicit | **large ↑** | ≈ / ↑ | ↓ vs extractor | ~100% |

- **Confirms H3** if the Pareto/joint-eff jump is concentrated in the pareto+elicit cell and the
  pareto-only and elicit-only cells are individually weak.
- **Falsifies the thesis** if pareto-only (R1) already closes most of the gap (would mean the
  constraint was motivational, not informational, contradicting the trace autopsy).

## Risks & confounds

- **Self-play collusion / handshake** → mitigate with cross-play eval (above).
- **Strategic misrepresentation** under elicitation + own-score reward (agents learn to lie) →
  use env-mediated truthful disclosure for headline runs; study lying separately.
- **Reward weighting** (`coef`) trades own-score vs efficiency; sweep it and report the frontier, not a
  single point.
- **Ladder vs factorial confound** (see Design note) — prefer the clean 2×2 for attribution.
- **dnd only** carries signal; casino saturates (~92% joint-eff for everyone) — don't waste cells on it
  except as a saturation control.

## Open decisions for the human

1. Clean 2×2 (`{outcome, +pareto} × {none, two-sided}` + one-sided probe) **or** the Core-4 ladder?
2. Binary `pareto_bonus` or the continuous `joint_efficiency` reward (recommended)? `coef` sweep range?
3. Elicitation fidelity: prompt-induced first, or go straight to env-mediated truthful disclosure?
4. Disclosure honesty: truthful/enforced (recommended for headline) vs strategic/learnable?
