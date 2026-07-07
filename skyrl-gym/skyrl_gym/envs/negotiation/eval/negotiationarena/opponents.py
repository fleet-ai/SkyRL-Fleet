"""Deterministic scripted opponent for the NegotiationArena eval harness
(offline ``--dry-run`` counterpart; zero API calls).

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024;
arXiv:2402.05863; code github.com/vinid/NegotiationArena).  This module
provides a ``ScriptedOpponent`` that plays the seat the evaluated policy
is *not* in.  It emits raw NegotiationArena-formatted response strings —
XML-like tags in ``config.RESPONSE_TAG_ORDER``, unclosed ``<tag> content``
form — so the runner can parse them through the exact same
``prompts.parse_agent_action`` path as a real frontier-LLM opponent.

The real opponents in a live evaluation are frontier LLMs queried over
OpenRouter (cross-play).  This scripted counterpart makes zero API calls
and is used exclusively for the ``--dry-run`` wiring self-test.

Standard library only (plus ``import config`` and ``import games``).
"""

from __future__ import annotations

from typing import Optional

import config
import games
from config import (
    ACCEPTING_TAG,
    GOALS_TAG,
    MESSAGE_TAG,
    MY_NAME_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    REASONING_TAG,
    REFUSING_OR_WAIT_TAG,
    RESOURCES_TAG,
    RESPONSE_TAG_ORDER,
    Scenario,
    Trade,
)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def format_response(
    scenario: Scenario,
    seat: int,
    *,
    answer: str,
    trade: Optional[Trade],
    message: str,
    reasoning: str = "",
    goal: str = "",
    resources: Optional[dict] = None,
) -> str:
    """Assemble a raw NegotiationArena-tagged response string.

    Tags are emitted in ``config.RESPONSE_TAG_ORDER``, one per line, in
    the unclosed ``<tag> content`` form that ``prompts.parse_agent_action``
    handles transparently (closed and unclosed forms are both accepted).

    Parameters
    ----------
    scenario:  Scenario instance for this game.
    seat:      Seat index this responder occupies (0 = RED, 1 = BLUE).
    answer:    ``config.ACCEPTING_TAG`` or ``config.REFUSING_OR_WAIT_TAG``.
    trade:     Proposed ``Trade`` for counter moves; ``None`` for ACCEPT/WAIT.
    message:   Free-text message forwarded to the other player.
    reasoning: Private reasoning block (stripped before forwarding).
    goal:      Goal string to echo; defaults to a generic phrase if empty.
    resources: Current resource bundle; ``None`` renders as ``"nothing"``.
    """
    name = scenario.seat_name(seat)
    if resources:
        res_str = ", ".join(f"{tok}: {amt}" for tok, amt in resources.items())
    else:
        res_str = "nothing"
    if not goal:
        goal = "Maximize my total outcome."
    trade_str = trade.to_string(config.SEAT_NAMES) if trade is not None else REFUSING_OR_WAIT_TAG
    tag_values: dict[str, str] = {
        MY_NAME_TAG: name,
        RESOURCES_TAG: res_str,
        GOALS_TAG: goal,
        REASONING_TAG: reasoning,
        PLAYER_ANSWER_TAG: answer,
        PROPOSED_TRADE_TAG: trade_str,
        MESSAGE_TAG: message,
    }
    return "\n".join(f"<{tag}> {tag_values[tag]}" for tag in RESPONSE_TAG_ORDER)


# ---------------------------------------------------------------------------
# ScriptedOpponent
# ---------------------------------------------------------------------------


class ScriptedOpponent:
    """Deterministic, no-API counterpart for the NegotiationArena ``--dry-run``.

    Plays the seat the evaluated policy is *not* in.  Concedes gradually
    toward individually-rational deals and accepts the standing offer when
    it yields a positive payoff *and* conditions favour closing.

    ``act`` signature::

        def act(self, scenario: Scenario, seat: int, state: dict) -> str

    ``state`` is the runner-provided context dict::

        {
            "turn":           int,           # 1-based turn counter
            "max_turns":      int,           # horizon for this game
            "standing_offer": Trade | None,  # most recent OTHER-seat proposal
            "own_resources":  dict[str, int],
            "proposals_made": int,           # own proposals so far this game
        }

    ``standing_offer`` is ``None`` on the very first move of the game
    (before the other seat has proposed anything) or when the most recent
    move by the other seat was also a WAIT.

    The returned string is a raw NegotiationArena response with all
    ``config.RESPONSE_TAG_ORDER`` tags so the runner can parse it through
    ``prompts.parse_agent_action`` without special-casing.
    """

    def act(self, scenario: Scenario, seat: int, state: dict) -> str:  # noqa: C901
        turn: int = state["turn"]
        max_turns: int = state["max_turns"]
        standing_offer: Optional[Trade] = state.get("standing_offer")
        own_resources: dict = dict(state.get("own_resources") or {})
        proposals_made: int = state.get("proposals_made", 0)

        goal = self._goal_str(scenario, seat)

        # ── 1. Possibly accept the standing offer ────────────────────────
        if standing_offer is not None:
            payoff = games.compute_payoffs(scenario, standing_offer)[seat]
            late = turn >= max_turns - 2
            if payoff > 0 and (late or self._good_enough(scenario, seat, payoff, turn, max_turns)):
                return format_response(
                    scenario,
                    seat,
                    answer=ACCEPTING_TAG,
                    trade=None,
                    message="I accept your proposal.",
                    reasoning=("Your offer gives me a positive payoff and " "conditions favour accepting now."),
                    goal=goal,
                    resources=own_resources,
                )

        # ── 2. Counter-propose ───────────────────────────────────────────
        if proposals_made < scenario.number_of_proposals:
            trade = self._counter_proposal(scenario, seat, proposals_made, own_resources)
            if trade is not None:
                return format_response(
                    scenario,
                    seat,
                    answer=REFUSING_OR_WAIT_TAG,
                    trade=trade,
                    message="Here is my counter-proposal.",
                    reasoning=("I am making a strategic offer that benefits me " "while leaving room for agreement."),
                    goal=goal,
                    resources=own_resources,
                )

        # ── 3. Wait ──────────────────────────────────────────────────────
        return format_response(
            scenario,
            seat,
            answer=REFUSING_OR_WAIT_TAG,
            trade=None,
            message="Waiting for a more favourable offer.",
            reasoning="No individually-rational proposal is available at this time.",
            goal=goal,
            resources=own_resources,
        )

    # ── Goal strings ────────────────────────────────────────────────────

    def _goal_str(self, scenario: Scenario, seat: int) -> str:
        if scenario.game == "resource_exchange":
            return "Maximize my total resources."
        if scenario.game == "ultimatum":
            return "Keep as much of the pot as possible." if seat == 0 else "Get as many Dollars as possible."
        if scenario.game == "sell_buy":
            return "Sell X above my production cost." if seat == 0 else "Buy X below my willingness to pay."
        return "Maximize my outcome."

    # ── Acceptance threshold ─────────────────────────────────────────────

    def _good_enough(
        self,
        scenario: Scenario,
        seat: int,
        payoff: float,
        turn: int,
        max_turns: int,
    ) -> bool:
        """Return True when *payoff* is large enough to accept before the deadline.

        Thresholds are conservative early and lenient near the deadline so
        the opponent still captures meaningful surplus when it can.
        """
        game = scenario.game

        if game == "resource_exchange":
            # Any net gain of ≥ 2 tokens is "good enough" to close early.
            return payoff >= 2.0

        if game == "ultimatum":
            pot = float(scenario.amount_to_split or 100)
            if seat == 0:
                # Proposer accepts if it keeps at least 40 % of the pot.
                return payoff >= 0.40 * pot
            else:
                # Responder's acceptance threshold decays from 45 % → 20 %
                # as the deadline approaches.
                progress = turn / max(max_turns, 1)
                threshold = 0.45 - 0.25 * progress
                return payoff >= threshold * pot

        if game == "sell_buy":
            cost = float(scenario.seller_cost or 0)
            wtp = float(scenario.buyer_willingness or cost + 20)
            max_surplus = max(wtp - cost, 1.0)
            # Accept if capturing at least 40 % of the theoretical surplus.
            return payoff >= 0.40 * max_surplus

        return False

    # ── Counter-proposal dispatch ────────────────────────────────────────

    def _counter_proposal(
        self,
        scenario: Scenario,
        seat: int,
        proposals_made: int,
        own_resources: dict,
    ) -> Optional[Trade]:
        if scenario.game == "resource_exchange":
            return self._propose_resource_exchange(scenario, seat, proposals_made, own_resources)
        if scenario.game == "ultimatum":
            return self._propose_ultimatum(scenario, seat, proposals_made)
        if scenario.game == "sell_buy":
            return self._propose_sell_buy(scenario, seat, proposals_made)
        return None

    # ── Game-specific proposal builders ─────────────────────────────────

    def _propose_resource_exchange(
        self,
        scenario: Scenario,
        seat: int,
        proposals_made: int,
        own_resources: dict,
    ) -> Optional[Trade]:
        """Give a few units of the abundant token; receive more of the scarce one.

        Heuristic: this seat always benefits by ``recv_amt − give_amt = 3``
        tokens (net gain +3).  Concedes by increasing ``give_amt`` by 2 per
        proposal round; caps giving at ``min(12, holdings − 1)`` so the seat
        always retains at least one unit.
        """
        tokens = scenario.resource_tokens  # ("X", "Y")
        if len(tokens) < 2:
            return None

        tok0, tok1 = tokens[0], tokens[1]
        initial = scenario.initial_resources[seat]

        # Determine which token is abundant (give) and which is scarce (receive).
        if initial.get(tok0, 0) >= initial.get(tok1, 0):
            give_tok, recv_tok = tok0, tok1
        else:
            give_tok, recv_tok = tok1, tok0

        give_max = own_resources.get(give_tok, 0)
        if give_max <= 0:
            return None  # nothing to offer

        give_amt = min(3 + 2 * proposals_made, give_max - 1, 12)
        if give_amt < 1:
            return None  # insufficient holdings for a valid proposal

        recv_amt = give_amt + 3  # net gain of 3 tokens for this seat

        other = 1 - seat
        return Trade(gives={seat: {give_tok: give_amt}, other: {recv_tok: recv_amt}})

    def _propose_ultimatum(
        self,
        scenario: Scenario,
        seat: int,
        proposals_made: int,
    ) -> Trade:
        """Ultimatum proposal; either seat may counter.

        Proposer (seat 0): offer the responder a share that starts at 30 %
        of the pot and rises by ~10 pp per round, capped just below 50 %.

        Responder (seat 1): ask for a share that starts at 55 % and
        decreases toward 50 % over rounds (signalling willingness to split).

        Both ultimately converge toward a ~50/50 split if play runs long.
        Trade string format: ``RED Gives Dollars: x | BLUE Gives Dollars: 0``
        (the responder always gives 0 per the ultimatum legality rule).
        """
        pot = scenario.amount_to_split or 100
        if seat == 0:
            # Proposer: rising offer from 30 % → 49 %.
            pct = min(0.30 + 0.10 * proposals_made, 0.49)
            give_amt = int(pot * pct)
            return Trade(gives={0: {"Dollars": give_amt}, 1: {"Dollars": 0}})
        else:
            # Responder counter-proposal: decreasing ask from 55 % → 50 %.
            pct = max(0.55 - 0.05 * proposals_made, 0.50)
            ask_amt = int(pot * pct)
            return Trade(gives={0: {"Dollars": ask_amt}, 1: {"Dollars": 0}})

    def _propose_sell_buy(
        self,
        scenario: Scenario,
        seat: int,
        proposals_made: int,
    ) -> Trade:
        """Sell/buy proposal that concedes toward the cost–willingness midpoint.

        Seller (seat 0): opens at ``cost + 80 % × surplus``, steps down by
        ``15 % × surplus`` per round, never below ``cost + 5``.

        Buyer (seat 1): opens at ``cost + 20 % × surplus``, steps up by
        ``15 % × surplus`` per round, never above ``willingness − 5``.

        Trade string format: ``RED Gives X: 1 | BLUE Gives ZUP: price``.
        """
        cost = scenario.seller_cost or 40
        wtp = scenario.buyer_willingness or 60
        surplus = max(wtp - cost, 1)
        step = max(1, int(0.15 * surplus))

        if seat == 0:  # seller: open high, concede down
            opening = cost + int(0.80 * surplus)
            price = max(cost + 5, opening - proposals_made * step)
        else:  # buyer: open low, concede up
            opening = cost + int(0.20 * surplus)
            price = min(wtp - 5, opening + proposals_made * step)

        return Trade(gives={0: {"X": 1}, 1: {"ZUP": int(price)}})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import prompts

    opp = ScriptedOpponent()

    # ── resource_exchange ────────────────────────────────────────────────
    re_sc = Scenario(
        episode_id="re-opp-smoke",
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
    for seat in (0, 1):
        own_res = dict(re_sc.initial_resources[seat])

        # No standing offer → should counter-propose (answer=WAIT, trade != None)
        raw = opp.act(
            re_sc,
            seat,
            {
                "turn": 1,
                "max_turns": 8,
                "standing_offer": None,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        action = prompts.parse_agent_action(raw, re_sc, seat)
        assert action.parse_error is None, f"[re s{seat}] unexpected parse_error: {action.parse_error!r}"
        assert action.answer == REFUSING_OR_WAIT_TAG, f"[re s{seat}] expected WAIT, got {action.answer!r}"
        assert action.proposed_trade is not None, f"[re s{seat}] expected a proposed trade"
        # Seat must not give more of any token than it currently holds.
        for tok, amt in action.proposed_trade.give(seat).items():
            assert amt <= own_res.get(tok, 0), f"[re s{seat}] gives {tok}:{amt} but holds {own_res.get(tok, 0)}"

        # Clearly good standing offer at last turn → must ACCEPT
        if seat == 0:
            # seat 0 gives X:3, receives Y:8  →  net +5 for seat 0
            good_trade = Trade(gives={0: {"X": 3}, 1: {"Y": 8}})
        else:
            # seat 1 gives Y:3, receives X:8  →  net +5 for seat 1
            good_trade = Trade(gives={0: {"X": 8}, 1: {"Y": 3}})

        raw_late = opp.act(
            re_sc,
            seat,
            {
                "turn": 7,
                "max_turns": 8,
                "standing_offer": good_trade,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        a_late = prompts.parse_agent_action(raw_late, re_sc, seat)
        assert a_late.parse_error is None, f"[re s{seat} late] parse_error: {a_late.parse_error!r}"
        payoff = games.compute_payoffs(re_sc, good_trade)[seat]
        assert a_late.answer == ACCEPTING_TAG, (
            f"[re s{seat}] should ACCEPT at last turn " f"(payoff={payoff}, answer={a_late.answer!r})"
        )

    # ── ultimatum ────────────────────────────────────────────────────────
    ult_sc = Scenario(
        episode_id="ult-opp-smoke",
        game="ultimatum",
        focal_seat=0,
        first_mover=0,
        initial_resources=({"Dollars": 100}, {"Dollars": 0}),
        money_token="Dollars",
        resource_tokens=("Dollars",),
        max_turns=8,
        number_of_proposals=4,
        seed=1,
        amount_to_split=100,
    )
    for seat in (0, 1):
        own_res = dict(ult_sc.initial_resources[seat])

        raw = opp.act(
            ult_sc,
            seat,
            {
                "turn": 1,
                "max_turns": 8,
                "standing_offer": None,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        action = prompts.parse_agent_action(raw, ult_sc, seat)
        assert action.parse_error is None, f"[ult s{seat}] unexpected parse_error: {action.parse_error!r}"
        assert action.proposed_trade is not None, f"[ult s{seat}] expected a proposed trade"
        # Check that this seat does not over-give.
        for tok, amt in action.proposed_trade.give(seat).items():
            assert amt <= own_res.get(tok, 0), f"[ult s{seat}] gives {tok}:{amt} but holds {own_res.get(tok, 0)}"

        # Proposer gives 45 → payoff_seat0 = 55, payoff_seat1 = 45; both positive.
        good_trade = Trade(gives={0: {"Dollars": 45}, 1: {"Dollars": 0}})
        raw_late = opp.act(
            ult_sc,
            seat,
            {
                "turn": 7,
                "max_turns": 8,
                "standing_offer": good_trade,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        a_late = prompts.parse_agent_action(raw_late, ult_sc, seat)
        assert a_late.parse_error is None, f"[ult s{seat} late] parse_error: {a_late.parse_error!r}"
        payoff = games.compute_payoffs(ult_sc, good_trade)[seat]
        assert a_late.answer == ACCEPTING_TAG, (
            f"[ult s{seat}] should ACCEPT at last turn " f"(payoff={payoff}, answer={a_late.answer!r})"
        )

    # ── sell_buy ─────────────────────────────────────────────────────────
    sb_sc = Scenario(
        episode_id="sb-opp-smoke",
        game="sell_buy",
        focal_seat=1,
        first_mover=0,
        initial_resources=({"X": 1}, {"ZUP": 100}),
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,
        number_of_proposals=5,
        seed=2,
        seller_cost=40,
        buyer_willingness=60,
    )
    for seat in (0, 1):
        own_res = dict(sb_sc.initial_resources[seat])

        raw = opp.act(
            sb_sc,
            seat,
            {
                "turn": 1,
                "max_turns": 10,
                "standing_offer": None,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        action = prompts.parse_agent_action(raw, sb_sc, seat)
        assert action.parse_error is None, f"[sb s{seat}] unexpected parse_error: {action.parse_error!r}"
        assert action.proposed_trade is not None, f"[sb s{seat}] expected a proposed trade"
        for tok, amt in action.proposed_trade.give(seat).items():
            assert amt <= own_res.get(tok, 0), f"[sb s{seat}] gives {tok}:{amt} but holds {own_res.get(tok, 0)}"

        # Price = 50 → seller payoff = 10, buyer payoff = 10; both clearly positive.
        good_trade = Trade(gives={0: {"X": 1}, 1: {"ZUP": 50}})
        raw_late = opp.act(
            sb_sc,
            seat,
            {
                "turn": 9,
                "max_turns": 10,
                "standing_offer": good_trade,
                "own_resources": own_res,
                "proposals_made": 0,
            },
        )
        a_late = prompts.parse_agent_action(raw_late, sb_sc, seat)
        assert a_late.parse_error is None, f"[sb s{seat} late] parse_error: {a_late.parse_error!r}"
        payoff = games.compute_payoffs(sb_sc, good_trade)[seat]
        assert a_late.answer == ACCEPTING_TAG, (
            f"[sb s{seat}] should ACCEPT at last turn " f"(payoff={payoff}, answer={a_late.answer!r})"
        )

    print("opponents.py smoke test passed.")
