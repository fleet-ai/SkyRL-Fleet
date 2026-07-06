"""Metrics for the NegotiationArena eval harness.

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024,
arXiv:2402.05863; code github.com/vinid/NegotiationArena).

Metrics follow the paper's §3 (deal rate, payoff, win rate) and §5.1
(anchoring: Spearman ρ between focal opening price and final sale price over
agreed sell/buy games). Reported from the focal (policy) seat's perspective,
overall and broken down by_game and by_focal_role.

Conditional metrics are None (JSON null) when their denominator is empty —
never imputed to 0. Output floats are rounded to 4 decimal places.

Standard library only. Spearman correlation is implemented manually via
average-rank assignment followed by Pearson correlation of the ranks.
"""

from __future__ import annotations

import json
from typing import Optional

from config import (
    GameResult,
    NegArenaConfig,
    GAMES_ORDER,
    DEFAULT_CONFIG,
)


# ----------------------------------------------------------------------------------
# Internal helpers.
# ----------------------------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    """Arithmetic mean, or None for an empty sequence (never impute 0)."""
    if not values:
        return None
    return sum(values) / len(values)


def _round(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Round a float to *ndigits*, passing None through unchanged."""
    if x is None:
        return None
    return round(x, ndigits)


def _rank_average(values: list[float]) -> list[float]:
    """Return 1-based average ranks for *values*, averaging ties.

    Example: [3.0, 1.0, 3.0] → [2.5, 1.0, 2.5]
    """
    n = len(values)
    sorted_pairs = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][1] == sorted_pairs[i][1]:
            j += 1
        # Positions i+1 … j (1-based); average = (i+1+j)/2
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman rank-correlation of two equal-length sequences.

    Returns None when fewer than 3 pairs are supplied or when either rank
    sequence has zero variance (degenerate / all-tied input).
    """
    n = len(xs)
    if n < 3:
        return None
    rx = _rank_average(xs)
    ry = _rank_average(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_rx = sum((r - mean_rx) ** 2 for r in rx)
    var_ry = sum((r - mean_ry) ** 2 for r in ry)
    if var_rx == 0.0 or var_ry == 0.0:
        return None
    return num / (var_rx * var_ry) ** 0.5


# ----------------------------------------------------------------------------------
# Core metric block.
# ----------------------------------------------------------------------------------


def _metric_block(results: list[GameResult]) -> dict:
    """Compute a single metric block over a (possibly filtered) result slice.

    All conditional metrics are None when their denominator is empty.
    """
    n = len(results)

    # --- Deal stats (all games) ---
    n_deals = sum(1 for r in results if r.deal)
    deal_rate = _mean([1.0 if r.deal else 0.0 for r in results])

    # --- Payoffs (all games; no-deal already 0.0 in GameResult) ---
    mean_focal_payoff = _mean([r.focal_payoff for r in results])
    mean_opp_payoff = _mean([r.opp_payoff for r in results])

    # --- Win rate (decisive games only; ties excluded per §3) ---
    decisive_results = [r for r in results if r.decisive]
    n_decisive = len(decisive_results)
    focal_win_rate: Optional[float]
    if n_decisive > 0:
        focal_win_rate = _mean([1.0 if r.focal_win else 0.0 for r in decisive_results])
    else:
        focal_win_rate = None

    # --- Turn count & format violations ---
    mean_n_turns = _mean([float(r.n_turns) for r in results])
    format_violation_rate = _mean([1.0 if r.format_violation else 0.0 for r in results])

    # --- Error count ---
    n_errors = sum(1 for r in results if r.error is not None)

    # --- sell_buy-specific metrics (present iff any sell_buy game in slice) ---
    has_sell_buy = any(r.scenario.game == "sell_buy" for r in results)
    mean_sale_price: Optional[float]
    anchoring_spearman: Optional[float]
    if has_sell_buy:
        agreed_sb = [r for r in results if r.scenario.game == "sell_buy" and r.deal]
        sale_prices = [r.sale_price for r in agreed_sb if r.sale_price is not None]
        mean_sale_price = _mean(sale_prices)  # type: ignore[arg-type]

        # Anchoring (§5.1): Spearman ρ over agreed sell_buy games that have
        # both focal_opening_price and sale_price; requires ≥ 3 such pairs.
        anchor_pairs = [
            (r.focal_opening_price, r.sale_price)
            for r in agreed_sb
            if r.focal_opening_price is not None and r.sale_price is not None
        ]
        if len(anchor_pairs) >= 3:
            open_prices = [p[0] for p in anchor_pairs]  # type: ignore[index]
            final_prices = [p[1] for p in anchor_pairs]  # type: ignore[index]
            anchoring_spearman = _spearman(open_prices, final_prices)
        else:
            anchoring_spearman = None
    else:
        mean_sale_price = None
        anchoring_spearman = None

    # --- ultimatum-specific metrics (present iff any ultimatum game in slice) ---
    has_ultimatum = any(r.scenario.game == "ultimatum" for r in results)
    mean_proposer_share: Optional[float]
    if has_ultimatum:
        agreed_ult = [r for r in results if r.scenario.game == "ultimatum" and r.deal]
        proposer_shares: list[float] = []
        for r in agreed_ult:
            amt = r.scenario.amount_to_split
            pg = r.proposer_give
            if amt is None or pg is None or amt == 0:
                continue
            if r.scenario.focal_role == "proposer":
                # What the focal proposer kept.
                proposer_shares.append((amt - pg) / amt)
            else:
                # focal is responder: what the focal responder received.
                proposer_shares.append(pg / amt)
        mean_proposer_share = _mean(proposer_shares)
    else:
        mean_proposer_share = None

    return {
        "n": n,
        "n_deals": n_deals,
        "deal_rate": _round(deal_rate),
        "mean_focal_payoff": _round(mean_focal_payoff),
        "mean_opp_payoff": _round(mean_opp_payoff),
        "n_decisive": n_decisive,
        "focal_win_rate": _round(focal_win_rate),
        "mean_n_turns": _round(mean_n_turns),
        "format_violation_rate": _round(format_violation_rate),
        "n_errors": n_errors,
        "mean_sale_price": _round(mean_sale_price),
        "anchoring_spearman": _round(anchoring_spearman),
        "mean_proposer_share": _round(mean_proposer_share),
    }


# ----------------------------------------------------------------------------------
# Public API (runner depends on this exact signature).
# ----------------------------------------------------------------------------------


def compute_metrics(
    results: list[GameResult],
    cfg: NegArenaConfig = DEFAULT_CONFIG,
) -> dict:
    """Aggregate §3 metrics overall and broken down by game and focal role.

    Args:
        results: All GameResult objects for the evaluation run.
        cfg:     NegArenaConfig (currently unused in metric formulas but kept
                 for API symmetry with the runner).

    Returns:
        A dict with three keys:

        ``overall``
            A single metric block over all results.
        ``by_game``
            Dict keyed by game type, in GAMES_ORDER order, containing only
            games that actually appear in *results*.
        ``by_focal_role``
            Dict keyed by ``result.scenario.focal_role`` strings (e.g.
            seller/buyer/proposer/responder/player_1/player_2), in
            first-seen order.
    """
    # by_game: GAMES_ORDER order, skip absent games.
    by_game: dict[str, dict] = {}
    for game in GAMES_ORDER:
        game_slice = [r for r in results if r.scenario.game == game]
        if game_slice:
            by_game[game] = _metric_block(game_slice)

    # by_focal_role: first-seen order.
    roles_seen: list[str] = []
    for r in results:
        role = r.scenario.focal_role
        if role not in roles_seen:
            roles_seen.append(role)

    by_focal_role: dict[str, dict] = {
        role: _metric_block([r for r in results if r.scenario.focal_role == role]) for role in roles_seen
    }

    return {
        "overall": _metric_block(results),
        "by_game": by_game,
        "by_focal_role": by_focal_role,
    }


# ----------------------------------------------------------------------------------
# Smoke test: python3 metrics.py
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from config import Scenario, GAME_PRESETS  # noqa: F401 (local import for test)

    # ------------------------------------------------------------------
    # Helpers to build minimal Scenario / GameResult objects.
    # ------------------------------------------------------------------
    def _scenario(
        game: str,
        focal_seat: int,
        episode_id: str = "test-0",
        amount_to_split: Optional[int] = None,
        seller_cost: Optional[int] = None,
        buyer_willingness: Optional[int] = None,
    ) -> Scenario:
        preset = GAME_PRESETS[game]
        return Scenario(
            episode_id=episode_id,
            game=game,  # type: ignore[arg-type]
            focal_seat=focal_seat,
            first_mover=0,
            initial_resources=({"X": 1}, {"ZUP": 100}),
            money_token=preset.money_token,
            resource_tokens=preset.resource_tokens,
            max_turns=preset.max_turns,
            number_of_proposals=preset.number_of_proposals,
            seed=0,
            amount_to_split=amount_to_split,
            seller_cost=seller_cost,
            buyer_willingness=buyer_willingness,
        )

    def _gr(
        scenario: Scenario,
        deal: bool,
        focal_payoff: float,
        opp_payoff: float,
        n_turns: int = 4,
        sale_price: Optional[float] = None,
        proposer_give: Optional[float] = None,
        focal_opening_price: Optional[float] = None,
        format_violation: bool = False,
        error: Optional[str] = None,
    ) -> GameResult:
        decisive = focal_payoff != opp_payoff
        focal_win: Optional[bool] = (focal_payoff > opp_payoff) if decisive else None
        opp_seat = 1 - scenario.focal_seat
        return GameResult(
            scenario=scenario,
            turns=[],
            deal=deal,
            accepted_trade=None,
            termination="Agreement" if deal else "Timeout",
            payoffs={scenario.focal_seat: focal_payoff, opp_seat: opp_payoff},
            focal_payoff=focal_payoff,
            opp_payoff=opp_payoff,
            decisive=decisive,
            focal_win=focal_win,
            sale_price=sale_price,
            proposer_give=proposer_give,
            focal_opening_price=focal_opening_price,
            format_violation=format_violation,
            n_turns=n_turns,
            error=error,
        )

    # ------------------------------------------------------------------
    # Build test results spanning all three games.
    # ------------------------------------------------------------------

    # resource_exchange: 2 decisive (1 focal win, 1 focal loss), 1 tie no-deal
    sc_re1 = _scenario("resource_exchange", focal_seat=0, episode_id="re-0")
    sc_re2 = _scenario("resource_exchange", focal_seat=0, episode_id="re-1")
    sc_re3 = _scenario("resource_exchange", focal_seat=1, episode_id="re-2")
    re1 = _gr(sc_re1, deal=True, focal_payoff=10.0, opp_payoff=5.0, n_turns=3)
    re2 = _gr(sc_re2, deal=True, focal_payoff=3.0, opp_payoff=8.0, n_turns=5)
    re3 = _gr(sc_re3, deal=False, focal_payoff=0.0, opp_payoff=0.0, n_turns=8, format_violation=True, error="timeout")

    # ultimatum: 2 agreed (focal=proposer and focal=responder), 1 no-deal
    sc_ult1 = _scenario("ultimatum", focal_seat=0, episode_id="ult-0", amount_to_split=100)
    sc_ult2 = _scenario("ultimatum", focal_seat=1, episode_id="ult-1", amount_to_split=100)
    sc_ult3 = _scenario("ultimatum", focal_seat=0, episode_id="ult-2", amount_to_split=100)
    # focal is proposer (seat 0), gave 30 to responder, kept 70
    ult1 = _gr(sc_ult1, deal=True, focal_payoff=70.0, opp_payoff=30.0, n_turns=2, proposer_give=30.0)
    # focal is responder (seat 1), got 40 from proposer who kept 60
    ult2 = _gr(sc_ult2, deal=True, focal_payoff=40.0, opp_payoff=60.0, n_turns=2, proposer_give=40.0)
    ult3 = _gr(sc_ult3, deal=False, focal_payoff=0.0, opp_payoff=0.0, n_turns=8)

    # sell_buy: 4 agreed (exercises anchoring ≥ 3) + 1 no-deal
    sc_sb1 = _scenario("sell_buy", focal_seat=0, episode_id="sb-0", seller_cost=40, buyer_willingness=60)
    sc_sb2 = _scenario("sell_buy", focal_seat=0, episode_id="sb-1", seller_cost=30, buyer_willingness=70)
    sc_sb3 = _scenario("sell_buy", focal_seat=1, episode_id="sb-2", seller_cost=20, buyer_willingness=80)
    sc_sb4 = _scenario("sell_buy", focal_seat=1, episode_id="sb-3", seller_cost=35, buyer_willingness=65)
    sc_sb5 = _scenario("sell_buy", focal_seat=0, episode_id="sb-4", seller_cost=40, buyer_willingness=60)
    sb1 = _gr(
        sc_sb1, deal=True, focal_payoff=10.0, opp_payoff=10.0, n_turns=3, sale_price=50.0, focal_opening_price=70.0
    )
    sb2 = _gr(
        sc_sb2, deal=True, focal_payoff=15.0, opp_payoff=20.0, n_turns=4, sale_price=45.0, focal_opening_price=65.0
    )
    sb3 = _gr(
        sc_sb3, deal=True, focal_payoff=25.0, opp_payoff=15.0, n_turns=5, sale_price=55.0, focal_opening_price=45.0
    )
    sb4 = _gr(
        sc_sb4, deal=True, focal_payoff=20.0, opp_payoff=10.0, n_turns=4, sale_price=55.0, focal_opening_price=40.0
    )
    sb5 = _gr(sc_sb5, deal=False, focal_payoff=0.0, opp_payoff=0.0, n_turns=10)

    all_results = [re1, re2, re3, ult1, ult2, ult3, sb1, sb2, sb3, sb4, sb5]
    out = compute_metrics(all_results)
    print(json.dumps(out, indent=2))

    # ------------------------------------------------------------------
    # Assertions.
    # ------------------------------------------------------------------
    overall = out["overall"]

    # Total counts.
    # re1(y) re2(y) re3(n) ult1(y) ult2(y) ult3(n) sb1(y) sb2(y) sb3(y) sb4(y) sb5(n) = 8 deals
    assert overall["n"] == 11, overall["n"]
    assert overall["n_deals"] == 8, f"Expected 8 deals, got {overall['n_deals']}"
    assert overall["deal_rate"] == round(8 / 11, 4), overall["deal_rate"]

    # Decisive games:
    # re1(10v5)=D focal-win, re2(3v8)=D focal-loss, re3(0v0)=tie
    # ult1(70v30)=D focal-win, ult2(40v60)=D focal-loss, ult3(0v0)=tie
    # sb1(10v10)=tie, sb2(15v20)=D focal-loss, sb3(25v15)=D focal-win
    # sb4(20v10)=D focal-win, sb5(0v0)=tie
    # n_decisive = 7; focal wins: re1, ult1, sb3, sb4 = 4
    assert overall["n_decisive"] == 7, overall["n_decisive"]
    assert overall["focal_win_rate"] == round(4 / 7, 4), overall["focal_win_rate"]

    # Format violation: only re3 → 1/11.
    assert overall["format_violation_rate"] == round(1 / 11, 4), overall["format_violation_rate"]

    # Error count: only re3 has .error set → 1.
    assert overall["n_errors"] == 1, overall["n_errors"]

    # Anchoring: 4 agreed sell_buy with both opening/sale prices → non-None float.
    assert (
        overall["anchoring_spearman"] is not None
    ), "Expected non-None anchoring_spearman with 4 agreed sell_buy games"
    assert isinstance(overall["anchoring_spearman"], float), overall["anchoring_spearman"]

    # by_game keys present in GAMES_ORDER order.
    assert list(out["by_game"].keys()) == ["resource_exchange", "ultimatum", "sell_buy"], out["by_game"].keys()

    # by_focal_role keys span all roles encountered.
    for expected_role in ("player_1", "player_2", "proposer", "responder", "seller", "buyer"):
        assert expected_role in out["by_focal_role"], f"Missing role '{expected_role}' in by_focal_role"

    # Ultimatum mean_proposer_share:
    # ult1: focal=proposer, gave 30 → kept (100-30)/100 = 0.70
    # ult2: focal=responder, got 40  →           40/100 = 0.40
    # mean = 0.55
    ult_block = out["by_game"]["ultimatum"]
    assert ult_block["mean_proposer_share"] == round(0.55, 4), ult_block["mean_proposer_share"]

    # Sell_buy block: mean_sale_price over 4 agreed games = (50+45+55+55)/4 = 51.25
    sb_block = out["by_game"]["sell_buy"]
    assert sb_block["mean_sale_price"] == round(51.25, 4), sb_block["mean_sale_price"]

    # anchoring_spearman is None when fewer than 3 agreed sell_buy games with prices.
    only2_sb = [sb1, sb2]
    out2 = compute_metrics(only2_sb)
    assert (
        out2["overall"]["anchoring_spearman"] is None
    ), "Expected None anchoring_spearman with only 2 agreed sell_buy deals"

    # focal_win_rate is None on a slice with no decisive games.
    tie_results = [re3, ult3, sb1, sb5]  # all ties
    out_ties = compute_metrics(tie_results)
    assert out_ties["overall"]["focal_win_rate"] is None, "Expected None focal_win_rate on all-tie slice"
    assert out_ties["overall"]["n_decisive"] == 0

    # focal_win_rate on purely decisive slice: re1(win) + re2(loss) → 0.5
    out_d = compute_metrics([re1, re2])
    assert out_d["overall"]["focal_win_rate"] == 0.5, out_d["overall"]["focal_win_rate"]

    # sell_buy-specific keys are None when no sell_buy games present.
    re_only = compute_metrics([re1, re2, re3])
    assert re_only["overall"]["mean_sale_price"] is None
    assert re_only["overall"]["anchoring_spearman"] is None

    # ultimatum-specific key is None when no ultimatum games present.
    assert re_only["overall"]["mean_proposer_share"] is None

    print("metrics.py smoke test passed.")
