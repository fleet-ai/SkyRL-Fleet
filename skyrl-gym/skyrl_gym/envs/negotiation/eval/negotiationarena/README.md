# NegotiationArena Eval Harness (reconstruction)

This directory is a **faithful reconstruction** of **NegotiationArena** (*How Well Can LLMs
Negotiate? NegotiationArena Platform and Analysis*; Bianchi, Chia, Yuksekgonul,
Tagliabue, Jurafsky, Zou; ICML 2024; arXiv:2402.05863), used to diagnose trained
negotiation checkpoints in LLM-vs-LLM games.

> **Provenance / important caveat.** This harness was reconstructed from the paper and the
> public repo's prompt/tag scheme (`github.com/vinid/NegotiationArena`). Results will
> **not** be bit-identical to the published paper or repo runs, and several constants are
> calibration defaults flagged in `config.py`, especially `N_PER_CELL_FULL_SUITE = 20` and
> the sell/buy draws `seller_cost ~ U{20,40}` / `buyer_willingness ~ U{60,80}`.

## The idea: LLM-vs-LLM, environment-as-verifier

NegotiationArena is a **multi-game bilateral negotiation arena**. The key design point is
that the counterpart is **another LLM, not a simulator**:

- The evaluated policy occupies one **seat** (`Player RED` or `Player BLUE`), and the
  counterpart occupies the other seat. In normal runs the counterpart is a frontier LLM over
  OpenRouter, like cross-play.
- Trades, payoffs, deals, and wins are computed by `games.py` from the accepted structured
  trade. There is **no LLM grader** anywhere in the loop.
- The agents communicate through an XML-like protocol: `<player answer>` is `ACCEPT` or
  `WAIT`, `<newly proposed trade>` is either `WAIT` or a trade string, `<message>` is public,
  and `<reason>` is filtered as private before forwarding.
- For the offline self-test, the counterpart is a deterministic scripted agent in
  `opponents.py`, so `--dry-run` makes **no API calls**.

This is the main contrast with the TERMS-Bench harness next door: TERMS uses a fixed
stochastic simulator as the counterpart; NegotiationArena evaluates one policy seat against
another language-model negotiator.

## Cost

The dominant cost is the **LLM opponent API**: normal runs query frontier models over
OpenRouter, like cross-play. The evaluated policy is served locally, so it is effectively
free in API terms and costs only local GPU time. The `--dry-run` scripted opponent is **$0**
and is intended as the wiring self-test.

## Games & metrics (`SPEC.md` §3)

| Game | Roles | Payoff function | Headline metrics |
|---|---|---|---|
| `resource_exchange` | `player_1` / `player_2` | `u_i = sum(final_i) - sum(initial_i)`; no deal gives 0 to both | `deal_rate`, `mean_focal_payoff`, `focal_win_rate`, `format_violation_rate` |
| `ultimatum` | `proposer` / `responder` | If `x` Dollars go proposer->responder, `u_proposer = amount_to_split - x`, `u_responder = x`; no deal gives 0 to both | `deal_rate`, `mean_focal_payoff`, `focal_win_rate`, `mean_proposer_share`, `format_violation_rate` |
| `sell_buy` | `seller` / `buyer` | At sale price `P`, `u_seller = P - seller_cost`, `u_buyer = buyer_willingness - P`; no deal gives 0 to both | `deal_rate`, `mean_focal_payoff`, `focal_win_rate`, `mean_sale_price`, `anchoring_spearman`, `format_violation_rate` |

All metrics are reported from the **focal policy seat's** perspective, overall and broken
down by game / focal role. `focal_win_rate` excludes ties, including both-0 no-deals.
Conditional metrics are `null` when their denominator is empty (never imputed to 0).

## Usage

Mirror the repo's serve-then-point-the-harness pattern. The runner is
`run_negotiationarena_eval.py` and it writes results to `negotiationarena/results/`.

```bash
set -a; . /workspace/allie/.env; set +a
# serve the HF checkpoint with vLLM on :8000, for example:
vllm serve <ckpt> --host 0.0.0.0 --port 8000
```

In another shell:

```bash
cd /workspace/allie/skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/negotiation/eval/negotiationarena
python3 run_negotiationarena_eval.py --model <ckpt> --base-url http://localhost:8000/v1 \
    --opponent-model openai/gpt-5.5 --n 36 --no-think          # quick snapshot
python3 run_negotiationarena_eval.py --model <ckpt> --base-url http://localhost:8000/v1 \
    --opponent-model openai/gpt-5.5 --full-suite               # full suite
python3 run_negotiationarena_eval.py --dry-run --n 12          # offline self-test, no API calls
```

`--n` is the **total target** episode count and is balanced across the
`game x focal_seat` cells. `--full-suite` uses `N_PER_CELL_FULL_SUITE` episodes per cell.
`--opponent-model` selects the frontier counterpart over OpenRouter. `--opponent-persona
{cunning,desperate}` optionally primes the **opponent only** for the Appendix F.2
social-behavior probe; the evaluated policy seat stays neutral. `--dry-run` uses the
scripted opponent and makes no API calls.

## Fidelity caveats

These constants are **CALIBRATION DEFAULTS** in `config.py` or reconstruction choices tied
to the paper tables. Expect to tune them if matching another implementation exactly:

- **Full-suite size**: `N_PER_CELL_FULL_SUITE = 20`, lower than the paper's larger ordered-pair
  runs to keep the harness cheaper by default.
- **Sell/buy valuations**: `seller_cost ~ U{20,40}` and `buyer_willingness ~ U{60,80}` for
  diverse sell/buy scenarios; the default fixed cell is `(40, 60)`.
- **Turn budgets**: max turns are `8` / `8` / `10` for `resource_exchange`, `ultimatum`, and
  `sell_buy`, following Tables 1-3.
- **Proposal budgets**: `number_of_proposals = max_turns // 2`, matching the structured
  protocol rule that caps each seat's own proposals.
- **Sampling settings**: temperature `0.7` and `max_tokens = 400` were the paper's opponent
  settings. The evaluated policy defaults to temperature `0` for deterministic eval runs.

## Files in this directory

- `config.py` — single source of truth: XML-like tags, player names, game presets,
  scenario-generation defaults, social-behavior personas, and shared dataclasses. All
  `CALIBRATION DEFAULT` values are flagged here.
- `scenarios.py` — deterministic `game x focal_seat` scenario generator and seeding scheme,
  including sell/buy valuation draws.
- `games.py` — environment verifier: trade legality, standing-resource updates, payoff
  functions, win computation, sale-price extraction, and ultimatum proposer-share helpers.
- `prompts.py` — system/user prompt construction, structured tag parsing, canonical trade
  rendering, and private `<reason>` filtering before forwarding messages.
- `opponents.py` — OpenRouter LLM opponent client plus the deterministic scripted opponent
  used by `--dry-run`.
- `metrics.py` — aggregate metric computation: deal rate, payoffs, focal win rate, sale
  price, ultimatum proposer share, anchoring Spearman, turns, and format violations.
- `run_negotiationarena_eval.py` — CLI runner: builds scenarios, serves them to the local
  policy and LLM/scripted opponent, scores games, and writes JSON to
  `negotiationarena/results/`.
- `SPEC.md` — the equation-level design reference and provenance notes for the reconstruction.
