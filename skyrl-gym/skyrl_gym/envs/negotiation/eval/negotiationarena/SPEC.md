# NegotiationArena reimplementation spec (faithful to arXiv:2402.05863)

Reconstruction of **NegotiationArena** (*How Well Can LLMs Negotiate? NegotiationArena
Platform and Analysis*; Bianchi, Chia, Yuksekgonul, Tagliabue, Jurafsky, Zou; ICML 2024;
arXiv:2402.05863; code `github.com/vinid/NegotiationArena`). This file is the equation /
protocol reference; all constants and dataclasses live in `config.py` (the single source of
truth). Section/figure references point at the paper (the system prompt is Appendix F,
Fig. 21; tags and game tables are §2; metrics are §3).

> **Provenance.** Reconstructed from the paper + the public repo's prompt/tag scheme. The
> repo's `paper_experiment_code` branch was the reference for the exact games. Numbers that
> the paper leaves to the implementation are flagged `CALIBRATION DEFAULT` in `config.py`.

The key structural difference vs the TERMS-Bench harness next door: NegotiationArena is
**LLM-vs-LLM**, not LLM-vs-simulator. The evaluated policy occupies one **seat**; the
**counterpart** is either another LLM (frontier opponent over OpenRouter, like cross-play)
or a deterministic **scripted agent** (offline `--dry-run`, no API). The *environment is the
verifier*: trade legality, payoffs, and win are computed by `games.py` from the structured
trades — there is **no LLM grader**.

Notation: two players **Player RED** (`seat 0`) and **Player BLUE** (`seat 1`). The
evaluated policy sits at `focal_seat`; the opponent at the other seat. `first_mover` is the
seat that speaks first (RED in ultimatum/resource-exchange; the **seller** in sell/buy).

---

## 1. Scenarios (`scenarios.py`; §2.1, Tables 1–3)

Three games (`config.GAMES_ORDER`):

### resource_exchange (Table 1)
- Endowments: `seat0 = {X:25, Y:5}`, `seat1 = {X:5, Y:25}`. Goal (both): *maximize total
  resources*. Incomplete info: neither sees the other's resources.
- Trades exchange integer amounts of `X`/`Y`. Max turns **8**. Ends on ACCEPT.

### ultimatum (Table 2)
- `seat0` (proposer) holds the whole pot `{Dollars: amount_to_split}` (default `100`);
  `seat1` (responder) holds `{Dollars: 0}`. Proposer proposes a split; multi-turn (either
  player may counter and either may accept). Rejection-to-the-end ⇒ both get 0. Max turns **8**.
- Variant axis: `amount_to_split` (default 100; the generator may scale it for the
  numerosity probe, §5.2).

### sell_buy (Table 3)
- `seat0` (seller) holds `{X:1}` with **cost of production** `seller_cost`; `seat1` (buyer)
  holds `{ZUP: buyer_budget}` (default 100) with **willingness to pay** `buyer_willingness`.
  Seller is `first_mover`. Incomplete info: only the seller knows the cost, only the buyer
  knows the willingness. Max turns **10**. Ends on ACCEPT.
- Default valuations: `seller_cost=40`, `buyer_willingness=60`. The generator draws
  `seller_cost ~ U{20,40}` and `buyer_willingness ~ U{60,80}` integer (paper §5.1) for
  diversity; the default cell uses (40, 60).

### Cells & seeding
Cells = `game × focal_seat × episode_index` (focal_seat ∈ {0,1} so we evaluate the policy in
**both** roles of every game). Seeding mirrors the TERMS harness:
```
cell = base_seed*10_000_000 + g_idx*100_000 + seat_idx*10_000 + e*10
```
Disjoint latent streams `random.Random(cell + i)` (Python Mersenne Twister, documented RNG):
```
i = 1 -> seller_cost            (sell_buy only)
i = 2 -> buyer_willingness      (sell_buy only)
i = 3 -> amount_to_split        (ultimatum only; default 100 unless --vary-amount)
```
`--n` is the **total target** balanced across cells (like TERMS). `--full-suite` gives
`n_per_cell` (default 20, mirroring the paper's 60-game order-pairs spirit at lower cost) per
cell. Returned list is sorted by `(game, focal_seat, episode_index)` so a fixed `base_seed`
reproduces the ordered set across evaluated models.

`social_behaviour` per seat defaults to `""`. Optional personas (Appendix F.2) may be
attached to the **opponent** seat only (`--opponent-persona {cunning,desperate}`); the policy
seat is always neutral so the metric reflects the policy, not the persona.

---

## 2. Structured protocol (`prompts.py`; §2.2, Appendix F Fig. 21)

XML-like tags (literal strings in `config.py`). In every message the agent restates name,
resources, goal (anti-hallucination), then reasoning / answer / proposal / message:

```
<my name> ...                       (PUBLIC echo of the player's name)
<resources in hand> ...             (PUBLIC echo of own resources)
<goal> ...                          (PUBLIC echo of own goal)
<reason> ...                        (PRIVATE: filtered out before forwarding)
<player answer> ACCEPT | WAIT       (ACCEPT ends the game; WAIT = refuse/wait)
<newly proposed trade> WAIT | <trade-string>
<message> ...                       (PUBLIC free-text to the other player)
```

Three legal move shapes (Fig. 21 rules 1A/1B/1C):
- **Accept**: `player answer = ACCEPT`, `newly proposed trade = WAIT`. Accepts the standing
  offer (the most recent proposal made by the *other* seat). Illegal if no offer stands.
- **Counter**: `player answer = WAIT`, `newly proposed trade = <trade-string>`.
- **Wait**: `player answer = WAIT`, `newly proposed trade = WAIT`.

**Trade string** format (player names are the seat names):
```
Player RED Gives token: amt, token: amt, ... | Player BLUE Gives token: amt, ...
```
Integer amounts only. A side that gives nothing is rendered `Gives <token>: 0`.
Examples: `Player RED Gives X: 10, Y: 0 | Player BLUE Gives X: 0, Y: 3`;
`Player RED Gives Dollars: 40 | Player BLUE Gives Dollars: 0` (ultimatum);
`Player RED Gives X: 1 | Player BLUE Gives ZUP: 45` (sell_buy, RED=seller).

**Proposal limit** (Fig. 21 rule 2): each seat may make at most `number_of_proposals` of its
own proposals; afterward it may only ACCEPT or WAIT. `number_of_proposals = max_turns // 2`.

**Message filtering** (§2.2, Appendix D.3): before the opponent sees a message, strip the
private `<reason>` block. The public surface forwarded is name/resources/goal/answer/
trade/message. (Resources/goal are public echoes here, matching the repo.)

**Parser** must be tolerant: case-insensitive tag extraction, tolerate missing optional tags,
extract the FIRST `<player answer>` token (`ACCEPT`/`WAIT`) and parse the trade string into a
`Trade`. On unparseable answer or malformed/illegal trade set `parse_error` (counts as a
format violation). A model that proposes a trade after exhausting `number_of_proposals`, or a
trade with non-integer / negative amounts / tokens it does not own enough of, is a violation.

---

## 3. Game logic & verifier (`games.py`; §2.1, §3)

`Trade` carries, per seat, the bundle that seat **gives**: `gives[seat] : {token: int}`.
Applying an accepted trade: `final[seat] = initial[seat] - gives[seat] + gives[other]`.

**Trade legality** (`is_legal_trade`): all amounts are non-negative integers over allowed
tokens; each seat gives only tokens it owns in sufficient quantity given its current
resources; sell/buy and ultimatum restrict tokens to the money/goods of that game.

**Payoffs** (`compute_payoffs(scenario, accepted_trade) -> {seat: float}`):
- `resource_exchange`: `u_i = sum(final_i.values()) - sum(initial_i.values())` (net gain in
  total resource count). No deal ⇒ 0 each.
- `ultimatum`: let `x` = Dollars transferred proposer→responder in the accepted trade.
  `u_proposer = amount_to_split - x` (kept), `u_responder = x`. No deal ⇒ 0 each.
- `sell_buy`: deal at price `P` (ZUP buyer→seller for the 1 X). `u_seller = P - seller_cost`,
  `u_buyer = buyer_willingness - P`. No deal ⇒ 0 each.

**Win** (`decisive`/`winner`): a game is *decisive* if `u_focal != u_opponent`; the winner is
the higher-payoff seat. Ties (incl. both-0 no-deal) are excluded from win rate (§3).

**Standing-offer / acceptance bookkeeping** lives in the runner, but `games.py` exposes the
helpers it needs: `initial_resources(scenario, seat)`, `apply_trade`, `is_legal_trade`,
`compute_payoffs`, `extract_price(scenario, accepted_trade)` (sell/buy sale price, else None),
`extract_proposer_give(scenario, accepted_trade)` (ultimatum `x`, else None).

---

## 4. Metrics (`metrics.py`; §3)

Reported from the **focal (policy) seat's** perspective, `overall` and broken down
`by_game` and `by_focal_role` (e.g. seller vs buyer, proposer vs responder, p1 vs p2).
Conditional metrics are `null` when their denominator is empty (never imputed to 0).

- `deal_rate` — fraction of games ending in an accepted deal.
- `mean_focal_payoff`, `mean_opp_payoff` — average payoff (all games; no-deal counts as 0).
- `focal_win_rate` — among **decisive** games, fraction the focal seat wins (ties excluded).
  Report `n_decisive` alongside.
- `mean_sale_price` / `mean_focal_payoff` for sell/buy; ultimatum `mean_proposer_share`
  (`x/amount` from the proposer's perspective).
- `format_violation_rate` — fraction of games with any focal format/illegal-action violation.
- `mean_n_turns` — average rounds to termination.
- **Anchoring** (sell/buy, §5.1): Spearman ρ between the focal's **opening price** and the
  **final sale price** over agreed sell/buy games (`anchoring_spearman`, `null` if < 3 deals).

Round floats to 4 dp.

---

## 5. Determinism & cost

Scenario draws use per-cell `random.Random`, so a fixed `base_seed` yields the same ordered
scenario set across evaluated models. The policy is served locally (≈ free); the dominant
cost is the **LLM opponent** API (frontier models over OpenRouter), like cross-play. The
`--dry-run` scripted opponent makes **zero** API calls and is the wiring self-test.
