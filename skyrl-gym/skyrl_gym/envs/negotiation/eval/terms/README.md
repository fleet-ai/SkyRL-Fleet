# TERMS-Bench Eval Harness (reconstruction)

This directory is a **faithful reconstruction** of the bilateral price-negotiation
instantiation of **TERMS-Bench** (*TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond
Deal Rate*; Zhang, Zhang, Pappu, El, Blanchet, Athey, Liu, Zou; 2026; arXiv:2605.13909),
used to diagnose trained negotiation checkpoints offline, separate from the RL training loop.

> **Provenance / important caveat.** The official code
> (`github.com/zou-group/terms-bench`, cited in the paper) was **not public** when this
> harness was built. The simulator kernel, scenario generator, and metrics were
> reimplemented from the paper's appendices (equation/table references live in `SPEC.md`
> and in `config.py` comments). Results will **not** be bit-identical to the published
> leaderboard, and several constants are calibration guesses (see *Fidelity caveats* below).

## The idea: environment-as-verifier

TERMS-Bench is a **Bayesian-game bilateral price negotiation**. The key design point is
that the counterpart is a **fixed stochastic simulator, not an LLM**:

- The counterpart (buyer or seller, opposite the agent) is the parametric kernel `pi_B`
  in `kernel.py`. Its acceptance / walk-away / counter-offer behavior is driven by a hidden
  type `(reservation, urgency, stance)` plus history features over the agent's offers.
- The evaluated **agent emits JSON actions** (`decision`, `price`, `message`, `belief`).
- **Metrics are computed by the environment** from prices, reservations, and the agent's
  self-reported belief. There is **no LLM grader** anywhere in the loop.
- The natural-language **"voice" layer is cosmetic**: by default messages are deterministic
  templates keyed on `(decision, cue)`. An optional `--voice-model` may render prettier
  prose but must never change the economic action or price.

## Cost

Evaluating a **locally-served checkpoint is effectively FREE in API terms**: the counterpart
is the simulator, the environment does all verification, and the policy is served locally, so
you only pay for **GPU time**. The single optional API cost is an LLM **voice layer** for
cosmetic messages (a few dollars at most for the full suite). With the default templates the
voice layer is **$0**.

## Diagnostic axes & metrics (`SPEC.md` §4)

The benchmark's thesis is "beyond deal rate", so it reports several axes, each conditioned to
avoid the deal-rate illusion:

| Axis | Metric(s) | Notes |
|---|---|---|
| Terminal value | `SE+ = AGR+ * CSE+` | surplus efficiency on feasible deals |
| Agreement calibration | `AGR+` (agree when feasible), `FAGR-` (agree when infeasible, lower better) | `safe-term = 1 - FAGR-` |
| Opponent modeling | `BE_type = (BE_r + BE_kappa + Brier_eta)/3` | scored from the agent's **self-reported belief** |
| Protocol compliance | `CritViol%` | price-bound, individual-rationality, invalid/unparseable action |

`CSE+` is conditional surplus efficiency `u_A / delta` over agreed feasible episodes;
monotonicity and turn-budget violations are tracked **separately** (not in `CritViol%`).
Conditional metrics are `null` when their denominator is empty (never imputed to 0).
**All aggregates are reported overall and broken down by regime and by family.**

## Regimes & counterpart families

**3 regimes** (`config.REGIMES`):

- `overlap` — positive ZOPA; a feasible deal exists.
- `urgency_shift` — same geometry as overlap, but the counterpart is drawn more urgent.
- `no_deal` — infeasible gap; the rational outcome is no agreement (tests `FAGR-`).

**6 counterpart families** (`config.FAMILIES_ORDER`): `candid`, `taciturn`, `expressive`,
`strategic`, `stochastic`, `adversarial`. Families vary stance priors, concession dynamics,
price noise, and **cue mode** (accurate / uninformative / noisy / pressuring), so the cosmetic
message channel ranges from informative to actively misleading.

## Usage

Mirror the repo's serve-then-point-the-harness pattern. The runner is `run_terms_eval.py`
and it writes results to `terms/results/`.

```bash
set -a; . /workspace/allie/.env; set +a
# serve the HF checkpoint with vLLM on :8000, then:
cd skyrl-gym/skyrl_gym/envs/negotiation/eval/terms
python3 run_terms_eval.py --model <ckpt_name> --base-url http://localhost:8000/v1 \
    --n 144 --no-think          # quick snapshot
python3 run_terms_eval.py --model <ckpt_name> --base-url http://localhost:8000/v1 \
    --full-suite                # full 1,800-episode suite
python3 run_terms_eval.py ... --dry-run   # build scenarios + self-test, no API calls
```

`--n` is the **total target** episode count and is balanced across the
`3 regimes x 2 agent_roles x 2 openers x 6 families` grid (so `--n 144` is one episode per
cell). `--full-suite` uses `n_per_cell = 25`, reproducing the paper's 100 episodes per
`(regime, family)` and **1,800 total**. `--dry-run` builds the scenario set and runs the
kernel/metric self-tests **without any API calls**, which is the fastest way to verify wiring.

## Fidelity caveats

These constants are **CALIBRATION DEFAULTS** in `config.py` — they are *not* numerically
pinned by the paper text and were chosen to reproduce the qualitatively described behavior.
Expect to tune them to match the official suite:

- **Regime geometry**: ZOPA width `zopa_min=10.0`, `zopa_max=40.0` (overlap / urgency_shift);
  infeasible gap `gap_min=5.0`, `gap_max=30.0` (no_deal), on a `[0,100]` price scale.
- **Urgency Beta params**: baseline `Beta(urgency_alpha=2.0, urgency_beta=2.0)` (mean 0.5);
  shifted `Beta(urgency_shift_alpha=5.0, urgency_shift_beta=2.0)` (mean ~0.71).
- **Strategic-cue concession threshold** `tau_conc=0.0` (any realized concession raises the
  Concede logit).

Other values (`K=10` horizon, price bounds, acceptance/walk-away/counter-offer coefficients)
follow the paper's tables, but only the four above are explicitly unspecified there. To adjust
any of them, edit `TermsConfig` / `FAMILIES` in `config.py`; consult `SPEC.md` for the exact
equations each constant feeds.

## Files in this directory

- `config.py` — single source of truth: `TermsConfig` hyperparameters, `FAMILIES` presets,
  shared dataclasses (`Scenario`, `AgentAction`, `EpisodeResult`, ...). All `CALIBRATION
  DEFAULT` values are flagged here.
- `scenarios.py` — deterministic regime generator and episode grid / seeding scheme
  (`SPEC.md` §1), producing identical scenario sets across evaluated models.
- `kernel.py` — the stochastic counterpart simulator `pi_B`: acceptance, walk-away,
  counter-offer, opening offer, and cue generation (`SPEC.md` §2).
- `prompts.py` — agent system/user prompt construction, output-JSON parsing (tolerant), and
  the cosmetic voice/template layer (`SPEC.md` §3).
- `metrics.py` — per-episode and aggregate metric computation: `SE+`, `AGR+`, `CSE+`,
  `FAGR-`, `BE_type`, `CritViol%`, sliced by regime and family (`SPEC.md` §4).
- `run_terms_eval.py` — CLI runner: serves scenarios to the policy, steps the kernel, scores
  episodes, and writes JSON to `terms/results/`.
- `SPEC.md` — the equation-level design reference (the source for this reconstruction).
