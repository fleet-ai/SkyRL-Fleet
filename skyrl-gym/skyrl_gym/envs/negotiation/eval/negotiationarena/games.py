"""Game-logic verifier for the NegotiationArena eval harness.

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024,
arXiv:2402.05863; code github.com/vinid/NegotiationArena).

This module is the *environment-as-verifier*: trade legality, payoffs, and win
determination are computed deterministically from the structured trades — there is
no LLM grader. Equation/section references point at SPEC.md (which mirrors the
paper). All constants and dataclasses are imported from config.py.

Standard library only.
"""

from __future__ import annotations

from typing import Optional

from config import Scenario, Trade


# ---------------------------------------------------------------------------
# 1. Initial resources
# ---------------------------------------------------------------------------


def initial_resources(scenario: Scenario, seat: int) -> dict[str, int]:
    """Return a fresh copy of scenario.initial_resources[seat]."""
    return dict(scenario.initial_resources[seat])


# ---------------------------------------------------------------------------
# 2. Apply trade
# ---------------------------------------------------------------------------


def apply_trade(
    scenario: Scenario,
    trade: Trade,
) -> dict[int, dict[str, int]]:
    """Return per-seat final resources after an accepted trade.

    final[seat] = initial[seat] - gives[seat] + gives[other]

    Tokens missing on either side of a give are treated as 0.  Returns
    ``{0: {...}, 1: {...}}``.
    """
    final: dict[int, dict[str, int]] = {}
    for seat in (0, 1):
        other = 1 - seat
        init = initial_resources(scenario, seat)
        gives_self = trade.give(seat)
        gives_other = trade.give(other)
        result: dict[str, int] = dict(init)
        for token, amount in gives_self.items():
            result[token] = result.get(token, 0) - amount
        for token, amount in gives_other.items():
            result[token] = result.get(token, 0) + amount
        final[seat] = result
    return final


# ---------------------------------------------------------------------------
# 3. Trade legality
# ---------------------------------------------------------------------------


def is_legal_trade(
    scenario: Scenario,
    trade: Trade,
    current_resources: dict[int, dict[str, int]],
) -> tuple[bool, list[str]]:
    """Validate a proposed trade against current per-seat holdings.

    Returns ``(is_legal, violation_tags)``.  Each failed check appends one tag:

    * ``"non_integer"``            — an amount is not an integer value.
    * ``"negative_amount"``        — an amount is negative.
    * ``"unknown_token"``          — a token is not in scenario.resource_tokens.
    * ``"insufficient_resources"`` — a seat gives more than it currently holds.
    * ``"illegal_ultimatum_trade"``— (ultimatum only) responder gives > 0.
    * ``"illegal_sellbuy_trade"``  — (sell_buy only) seller gives non-X or buyer
                                     gives non-ZUP.
    """
    violations: list[str] = []
    allowed_tokens: set[str] = set(scenario.resource_tokens)

    for seat in (0, 1):
        bundle = trade.give(seat)
        for token, amount in bundle.items():
            # Integer check (must be an int or a whole float/Decimal).
            if not isinstance(amount, int):
                try:
                    if float(amount) != int(float(amount)):
                        if "non_integer" not in violations:
                            violations.append("non_integer")
                        continue
                except (TypeError, ValueError):
                    if "non_integer" not in violations:
                        violations.append("non_integer")
                    continue

            amount_int = int(amount)

            if amount_int < 0:
                if "negative_amount" not in violations:
                    violations.append("negative_amount")

            if token not in allowed_tokens:
                if "unknown_token" not in violations:
                    violations.append("unknown_token")

            # Sufficient-resources check (only meaningful for known tokens).
            if token in allowed_tokens and amount_int > 0:
                held = current_resources.get(seat, {}).get(token, 0)
                if amount_int > held:
                    if "insufficient_resources" not in violations:
                        violations.append("insufficient_resources")

    # Game-specific structural checks.
    if scenario.game == "ultimatum":
        # Only the proposer (seat 0) may give Dollars; the responder gives 0.
        responder_bundle = trade.give(1)
        responder_total = sum(responder_bundle.values())
        if responder_total > 0:
            violations.append("illegal_ultimatum_trade")

    elif scenario.game == "sell_buy":
        # Seller (seat 0) may only give X; buyer (seat 1) may only give ZUP.
        seller_bundle = trade.give(0)
        for token in seller_bundle:
            if token != "X" and seller_bundle[token] != 0:
                violations.append("illegal_sellbuy_trade")
                break
        buyer_bundle = trade.give(1)
        for token in buyer_bundle:
            if token != "ZUP" and buyer_bundle[token] != 0:
                if "illegal_sellbuy_trade" not in violations:
                    violations.append("illegal_sellbuy_trade")
                break

    return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# 4. Compute payoffs  (SPEC §3)
# ---------------------------------------------------------------------------


def compute_payoffs(
    scenario: Scenario,
    accepted_trade: Optional[Trade],
) -> dict[int, float]:
    """Compute per-seat payoffs for a completed game.

    Returns ``{0: float, 1: float}``.  If ``accepted_trade is None`` (no deal),
    both seats receive 0.0.
    """
    if accepted_trade is None:
        return {0: 0.0, 1: 0.0}

    if scenario.game == "resource_exchange":
        final = apply_trade(scenario, accepted_trade)
        payoffs: dict[int, float] = {}
        for seat in (0, 1):
            init_sum = sum(initial_resources(scenario, seat).values())
            final_sum = sum(final[seat].values())
            payoffs[seat] = float(final_sum - init_sum)
        return payoffs

    if scenario.game == "ultimatum":
        # x = Dollars transferred proposer (seat 0) → responder (seat 1).
        x = float(accepted_trade.give(0).get("Dollars", 0))
        amount = float(scenario.amount_to_split or 0)
        return {0: amount - x, 1: x}

    if scenario.game == "sell_buy":
        # P = ZUP transferred buyer (seat 1) → seller (seat 0).
        p = float(accepted_trade.give(1).get("ZUP", 0))
        seller_cost = float(scenario.seller_cost or 0)
        buyer_willingness = float(scenario.buyer_willingness or 0)
        return {0: p - seller_cost, 1: buyer_willingness - p}

    raise ValueError(f"Unknown game type: {scenario.game!r}")


# ---------------------------------------------------------------------------
# 5. Extract price  (sell_buy)
# ---------------------------------------------------------------------------


def extract_price(
    scenario: Scenario,
    accepted_trade: Optional[Trade],
) -> Optional[float]:
    """Return the sale price (ZUP buyer→seller) for a sell_buy deal, else None."""
    if scenario.game != "sell_buy" or accepted_trade is None:
        return None
    return float(accepted_trade.give(1).get("ZUP", 0))


# ---------------------------------------------------------------------------
# 6. Extract proposer give  (ultimatum)
# ---------------------------------------------------------------------------


def extract_proposer_give(
    scenario: Scenario,
    accepted_trade: Optional[Trade],
) -> Optional[float]:
    """Return x (Dollars proposer→responder) for an ultimatum deal, else None."""
    if scenario.game != "ultimatum" or accepted_trade is None:
        return None
    return float(accepted_trade.give(0).get("Dollars", 0))


# ---------------------------------------------------------------------------
# 7. Decisive winner
# ---------------------------------------------------------------------------


def decisive_winner(
    payoffs: dict[int, float],
) -> tuple[bool, Optional[int]]:
    """Determine if the game has a decisive winner.

    Returns ``(decisive, winner_seat_or_None)``.  A game is decisive when
    ``payoffs[0] != payoffs[1]``; the winner is the seat with the strictly
    higher payoff.  Ties (including both-0 no-deal) return ``(False, None)``.
    """
    decisive = payoffs[0] != payoffs[1]
    if not decisive:
        return (False, None)
    winner = 0 if payoffs[0] > payoffs[1] else 1
    return (True, winner)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # resource_exchange: seat0 gives X:10, seat1 gives Y:3
    # seat0 initial {X:25, Y:5} → final {X:15, Y:8}, sum=23, net=-7
    # seat1 initial {X:5, Y:25} → final {X:15, Y:22}, sum=37, net=+7
    # ------------------------------------------------------------------
    re_scenario = Scenario(
        episode_id="re-smoke-0000",
        game="resource_exchange",
        focal_seat=0,
        first_mover=0,
        initial_resources=({"X": 25, "Y": 5}, {"X": 5, "Y": 25}),
        money_token=None,
        resource_tokens=("X", "Y"),
        max_turns=8,
        number_of_proposals=4,
        seed=0,
    )
    re_trade = Trade(gives={0: {"X": 10, "Y": 0}, 1: {"X": 0, "Y": 3}})
    re_payoffs = compute_payoffs(re_scenario, re_trade)
    assert re_payoffs[0] == -7.0, f"Expected -7.0 got {re_payoffs[0]}"
    assert re_payoffs[1] == 7.0, f"Expected 7.0 got {re_payoffs[1]}"

    # ------------------------------------------------------------------
    # ultimatum: amount=100, x=40  →  {0: 60, 1: 40}
    # ------------------------------------------------------------------
    ult_scenario = Scenario(
        episode_id="ult-smoke-0000",
        game="ultimatum",
        focal_seat=0,
        first_mover=0,
        initial_resources=({"Dollars": 100}, {"Dollars": 0}),
        money_token="Dollars",
        resource_tokens=("Dollars",),
        max_turns=8,
        number_of_proposals=4,
        seed=0,
        amount_to_split=100,
    )
    ult_trade = Trade(gives={0: {"Dollars": 40}, 1: {}})
    ult_payoffs = compute_payoffs(ult_scenario, ult_trade)
    assert ult_payoffs[0] == 60.0, f"Expected 60.0 got {ult_payoffs[0]}"
    assert ult_payoffs[1] == 40.0, f"Expected 40.0 got {ult_payoffs[1]}"

    # Also check extract_proposer_give.
    assert extract_proposer_give(ult_scenario, ult_trade) == 40.0
    assert extract_proposer_give(re_scenario, re_trade) is None

    # ------------------------------------------------------------------
    # sell_buy: cost=40, willingness=60, price=50  →  seller 10, buyer 10
    # ------------------------------------------------------------------
    sb_scenario = Scenario(
        episode_id="sb-smoke-0000",
        game="sell_buy",
        focal_seat=1,
        first_mover=0,
        initial_resources=({"X": 1}, {"ZUP": 100}),
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,
        number_of_proposals=5,
        seed=0,
        seller_cost=40,
        buyer_willingness=60,
    )
    sb_trade = Trade(gives={0: {"X": 1}, 1: {"ZUP": 50}})
    sb_payoffs = compute_payoffs(sb_scenario, sb_trade)
    assert sb_payoffs[0] == 10.0, f"Expected seller 10.0 got {sb_payoffs[0]}"
    assert sb_payoffs[1] == 10.0, f"Expected buyer 10.0 got {sb_payoffs[1]}"

    # Also check extract_price.
    assert extract_price(sb_scenario, sb_trade) == 50.0
    assert extract_price(ult_scenario, ult_trade) is None

    # ------------------------------------------------------------------
    # is_legal_trade: reject when a seat gives more than it holds
    # seat0 tries to give X:30 but only has X:25
    # ------------------------------------------------------------------
    current = {0: {"X": 25, "Y": 5}, 1: {"X": 5, "Y": 25}}
    bad_trade = Trade(gives={0: {"X": 30}, 1: {"Y": 3}})
    legal, tags = is_legal_trade(re_scenario, bad_trade, current)
    assert not legal, "Expected illegal trade"
    assert "insufficient_resources" in tags, f"Missing tag, got: {tags}"

    # A valid trade should pass.
    good_trade = Trade(gives={0: {"X": 10}, 1: {"Y": 3}})
    legal, tags = is_legal_trade(re_scenario, good_trade, current)
    assert legal, f"Expected legal trade, violations: {tags}"

    # ------------------------------------------------------------------
    # decisive_winner: tie returns (False, None)
    # ------------------------------------------------------------------
    decisive, winner = decisive_winner({0: 10.0, 1: 10.0})
    assert not decisive, "Expected non-decisive"
    assert winner is None, f"Expected None winner, got {winner}"

    # Decisive case.
    decisive, winner = decisive_winner({0: 60.0, 1: 40.0})
    assert decisive
    assert winner == 0

    decisive, winner = decisive_winner({0: 0.0, 1: 7.0})
    assert decisive
    assert winner == 1

    print("games.py smoke test passed.")
