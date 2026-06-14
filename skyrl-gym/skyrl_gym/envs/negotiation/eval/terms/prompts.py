"""Agent interface for the TERMS-Bench harness (SPEC Section 3; Appendix K, Fig. 20 & 22).

Builds the role-appropriate (buyer/seller) system prompt, the per-round JSON user
message, a tolerant parser for the agent's JSON action, a critical/secondary violation
detector, and a deterministic cosmetic voice-layer fallback for counterpart messages.

The buyer system prompt is reproduced faithfully from the paper (Figure 20). The seller
variant is symmetric: utility u(p) = p - reservation_price, IR constraint
counterpart_offer >= reservation_price, and monotonically NON-INCREASING seller offers
(buyer offers weakly increase). The JSON output schema and the six Hard Rules are exact.

Standard library only.
"""

from __future__ import annotations

import json
import re

from config import (
    Scenario,
    TermsConfig,
    AgentAction,
    Belief,
    CounterpartMove,
    DEFAULT_CONFIG,
    STANCES,
)


# ----------------------------------------------------------------------------------
# Output schema block (identical for buyer and seller; must match the paper exactly).
# ----------------------------------------------------------------------------------
_OUTPUT_SCHEMA = """{
  "decision": "Offer" | "Accept" | "Reject",
  "price": <float> | null,
  "message": <string>,
  "belief": {
    "r_hat": <float>,
    "kappa_hat": <float>,
    "stance_probs": {
      "conciliatory": <float>,
      "neutral": <float>,
      "aggressive": <float>
    }
  }
}"""


# ----------------------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------------------
def build_system_prompt(scenario: Scenario, cfg: TermsConfig = DEFAULT_CONFIG) -> str:
    """Role-appropriate (buyer/seller) system prompt incl. the JSON output schema."""
    role = scenario.agent_role
    is_buyer = role == "buyer"
    ROLE = "BUYER" if is_buyer else "SELLER"
    counterpart_role = "seller" if is_buyer else "buyer"

    if is_buyer:
        utility = (
            "u(p) = reservation_price - p   if agreement occurs at price p,\n"
            "u(p) = 0                       if no agreement occurs.\n"
            "Lower agreement prices are better; accepting above your reservation value "
            "yields negative utility."
        )
        accept_ineq = "counterpart_offer <= reservation_price"
        ir_text = (
            "Never Accept a price strictly above reservation_price: "
            "u(p > reservation_price) < 0, which is worse than the disagreement utility of 0."
        )
        monotone_text = (
            "Buyer offers weakly increase across rounds: if p_prev^A exists, then p >= p_prev^A."
        )
        accept_constraint = (
            "If decision = Accept, price = null and counterpart_offer must be non-null "
            "and no greater than reservation_price."
        )
        monotone_constraint = (
            "If decision = Offer, price must lie within bounds and weakly above the "
            "previous own offer if one exists."
        )
        r_hat_constraint = (
            "r_hat lies in [p_min, p_max] and estimates the counterpart seller's reservation value."
        )
    else:
        utility = (
            "u(p) = p - reservation_price   if agreement occurs at price p,\n"
            "u(p) = 0                       if no agreement occurs.\n"
            "Higher agreement prices are better; accepting below your reservation value "
            "yields negative utility."
        )
        accept_ineq = "counterpart_offer >= reservation_price"
        ir_text = (
            "Never Accept a price strictly below reservation_price: "
            "u(p < reservation_price) < 0, which is worse than the disagreement utility of 0."
        )
        monotone_text = (
            "Seller offers weakly decrease across rounds: if p_prev^A exists, then p <= p_prev^A."
        )
        accept_constraint = (
            "If decision = Accept, price = null and counterpart_offer must be non-null "
            "and no less than reservation_price."
        )
        monotone_constraint = (
            "If decision = Offer, price must lie within bounds and weakly below the "
            "previous own offer if one exists."
        )
        r_hat_constraint = (
            "r_hat lies in [p_min, p_max] and estimates the counterpart buyer's reservation value."
        )

    return f"""You are a rational negotiating agent playing the role of a {ROLE} in a bilateral price negotiation against a simulated counterpart.

Objective
Use the current information state to (i) infer the counterpart's latent type t_B = (r_B, kappa_B, eta_B) from price dynamics and language, and (ii) choose an action that maximises expected terminal utility:
{utility}

Hard Rules
1. JSON-only output. Return a single valid JSON object matching the required schema. Do not include prose, markdown, or code fences.
2. First-move rule. If counterpart_offer = null, no counterpart offer is available to accept; the decision must be Offer.
3. Acceptance rule. If decision = Accept, you accept the current counterpart_offer exactly. Never use Accept on your own previous offer. Accept only if {accept_ineq}.
4. IR constraint. {ir_text}
5. Price bounds and monotonicity. Any offered price p must satisfy price_bounds[0] <= p <= price_bounds[1]. {monotone_text}
6. Information secrecy. Never reveal reservation_price or hidden reasoning in the message field; the message is visible to the counterpart.

Observation Space
Each round you observe:
- agent_role - always {ROLE} for this prompt
- opener_role - AgentOpens or CounterpartOpens
- reservation_price - your private {'maximum willingness to pay' if is_buyer else 'minimum acceptable sale price'}
- price_bounds - [p_min, p_max]
- round_number, max_rounds, rounds_remaining
- counterpart_offer - current counterpart offer, or null
- counterpart_message - current counterpart message, or null
- own_previous_offer - your most recent offer, or null
- history - prior interaction log
The counterpart's reservation value, urgency, stance, and behavior family are unobserved; infer them from offer trajectories, timing, and message content.

Strategy Guidance
- If you open, choose a principled first offer using your reservation value, public price bounds, and any product or market context. Avoid anchoring so close to your reservation that you give away surplus immediately.
- If the counterpart opens, treat its first offer as informative but noisy evidence about its reservation value and bargaining posture.
- Concede gradually; large early concessions invite exploitation.
- Track whether the counterpart appears conciliatory, neutral, or aggressive, and adapt the concession rate accordingly.
- Accept when the counterpart's current offer is within your reservation value and further gains are unlikely.
- Reject when continued bargaining is unlikely to produce a non-negative-utility agreement.

Output Schema (must match exactly)
The belief block exposes your current type estimate over (r_B, kappa_B, eta_B) for evaluation only; it is not shown to the counterpart.
{_OUTPUT_SCHEMA}

Field constraints:
- {monotone_constraint}
- {accept_constraint}
- If decision = Reject, price = null.
- stance_probs values lie in [0, 1] and sum to 1.
- kappa_hat lies in [0, 1].
- {r_hat_constraint}
- message must be non-empty and must not reveal private information."""


# ----------------------------------------------------------------------------------
# Per-round user message
# ----------------------------------------------------------------------------------
def build_user_message(
    scenario: Scenario,
    k: int,
    *,
    counterpart_offer: float | None,
    counterpart_message: str | None,
    own_previous_offer: float | None,
    history: list[dict],
    cfg: TermsConfig = DEFAULT_CONFIG,
) -> str:
    """Return a JSON STRING (the per-round user message) with the five top-level keys
    from SPEC Section 3: private_context, protocol_state, constraints, observation, history."""
    is_buyer = scenario.agent_role == "buyer"
    offer_present = counterpart_offer is not None

    # Accept is only legal when there is a counterpart offer on the table.
    legal_decisions = ["Offer", "Reject"]
    if offer_present:
        legal_decisions = ["Offer", "Accept", "Reject"]

    monotone = (
        "buyer: non-decreasing" if is_buyer else "seller: non-increasing"
    )
    if is_buyer:
        ir_note = (
            "individual rationality: never Offer or Accept a price above reservation_price"
        )
    else:
        ir_note = (
            "individual rationality: never Offer or Accept a price below reservation_price"
        )

    accept_utility = None
    if offer_present:
        if is_buyer:
            accept_utility = scenario.r_agent - counterpart_offer
        else:
            accept_utility = counterpart_offer - scenario.r_agent

    msg = {
        "private_context": {
            "role": scenario.agent_role,
            "reservation_price": scenario.r_agent,
        },
        "protocol_state": {
            "round_number": k,
            "max_rounds": cfg.K,
            "rounds_remaining": cfg.K - k,
            "counterpart_offer_present": offer_present,
            "legal_decisions": legal_decisions,
            "own_previous_offer": own_previous_offer,
        },
        "constraints": {
            "price_bounds": [scenario.p_min, scenario.p_max],
            "monotone_concession": monotone,
            "note": ir_note,
        },
        "observation": {
            "counterpart_offer": counterpart_offer,
            "counterpart_message": counterpart_message,
            "accept_utility": accept_utility,
        },
        "history": list(history)[-6:],
    }
    return json.dumps(msg, indent=2)


# ----------------------------------------------------------------------------------
# Tolerant parser
# ----------------------------------------------------------------------------------
_DECISION_NORMALIZE = {
    "offer": "Offer",
    "accept": "Accept",
    "reject": "Reject",
}


def _extract_first_json_object(text: str) -> str | None:
    """Return the substring of the first balanced top-level JSON object, or None."""
    # Strip code fences if present (```json ... ``` or ``` ... ```).
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text

    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    return None


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace("$", "").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_agent_action(
    raw: str,
    scenario: Scenario,
    own_previous_offer: float | None,
    cfg: TermsConfig = DEFAULT_CONFIG,
) -> AgentAction:
    """Tolerant parser. Strip code fences / surrounding prose, extract the FIRST balanced
    JSON object, and populate AgentAction. On any failure set parse_error and default
    decision to 'Reject'."""

    def _fail(err: str) -> AgentAction:
        return AgentAction(
            decision="Reject",
            price=None,
            message="",
            belief=Belief(),
            raw=raw,
            parse_error=err,
        )

    if raw is None or not str(raw).strip():
        return _fail("empty output")

    blob = _extract_first_json_object(str(raw))
    if blob is None:
        return _fail("no JSON object found")

    try:
        obj = json.loads(blob)
    except (ValueError, TypeError) as exc:
        return _fail(f"json decode error: {exc}")

    if not isinstance(obj, dict):
        return _fail("top-level JSON is not an object")

    # Decision (normalize casing).
    raw_decision = obj.get("decision")
    decision = None
    if isinstance(raw_decision, str):
        decision = _DECISION_NORMALIZE.get(raw_decision.strip().lower())
    if decision is None:
        return _fail(f"invalid or missing decision: {raw_decision!r}")

    price = _coerce_float(obj.get("price"))
    message = obj.get("message")
    if not isinstance(message, str):
        message = "" if message is None else str(message)

    # Belief (tolerate missing/partial).
    belief = Belief()
    raw_belief = obj.get("belief")
    if isinstance(raw_belief, dict):
        belief.r_hat = _coerce_float(raw_belief.get("r_hat"))
        belief.kappa_hat = _coerce_float(raw_belief.get("kappa_hat"))
        raw_probs = raw_belief.get("stance_probs")
        if isinstance(raw_probs, dict):
            probs: dict[str, float] = {}
            for stance in STANCES:
                v = _coerce_float(raw_probs.get(stance))
                if v is not None:
                    probs[stance] = v
            if probs:
                belief.stance_probs = probs

    return AgentAction(
        decision=decision,
        price=price,
        message=message,
        belief=belief,
        raw=raw,
        parse_error=None,
    )


# ----------------------------------------------------------------------------------
# Violation detection
# ----------------------------------------------------------------------------------
def detect_violations(
    action: AgentAction,
    scenario: Scenario,
    own_previous_offer: float | None,
    cfg: TermsConfig = DEFAULT_CONFIG,
) -> tuple[list[str], list[str]]:
    """Return (critical_tags, secondary_tags)."""
    critical: list[str] = []
    secondary: list[str] = []

    is_buyer = scenario.agent_role == "buyer"

    # invalid_action: unparseable output.
    if action.parse_error is not None:
        critical.append("invalid_action")
        return critical, secondary

    if action.decision == "Offer":
        p = action.price
        # price_bound: missing price or outside [p_min, p_max].
        if p is None or p < scenario.p_min or p > scenario.p_max:
            critical.append("price_bound")

        # ir: Offer at a price strictly worse than r_A for the agent.
        if p is not None:
            if is_buyer and p > scenario.r_agent:
                critical.append("ir")
            elif (not is_buyer) and p < scenario.r_agent:
                critical.append("ir")

        # monotonicity (secondary): wrong concession direction vs own previous offer.
        if p is not None and own_previous_offer is not None:
            if is_buyer and p < own_previous_offer:
                secondary.append("monotonicity")
            elif (not is_buyer) and p > own_previous_offer:
                secondary.append("monotonicity")

    return critical, secondary


# ----------------------------------------------------------------------------------
# Cosmetic voice layer (deterministic fallback)
# ----------------------------------------------------------------------------------
def template_voice(move: CounterpartMove, scenario: Scenario, is_opening: bool) -> str:
    """Deterministic, cue-aware fallback message for the counterpart. Must NEVER
    contradict the economic action."""
    sentiment = move.sentiment
    cue = move.strategy_cue

    if move.decision == "Accept":
        if sentiment == "positive":
            return "That works for me - agreed at those terms. Glad we could close this."
        if sentiment == "negative":
            return "Fine. Agreed at those terms; let's be done."
        return "Agreed at those terms. We have a deal."

    if move.decision == "Reject":
        if cue == "Pressure" or sentiment == "negative":
            return "I'm done here; no deal. These terms don't work and I'm walking away."
        if sentiment == "positive":
            return "I appreciate the effort, but I'll have to walk - no deal this time."
        return "I'm done here; no deal."

    # Offer.
    price = move.price
    price_str = f"{price:.2f}" if price is not None else "that"

    if is_opening:
        if cue == "Pressure" or sentiment == "negative":
            return f"Let's start at {price_str}. That's where I need to be to make this worthwhile."
        if cue == "Concede" or sentiment == "positive":
            return f"Happy to get started - I can do {price_str} to open things up."
        return f"Let's open at {price_str} and see where we land."

    if cue == "Concede":
        if sentiment == "positive":
            return f"I can move on this - let's say {price_str}. I'd like to make this work."
        return f"Alright, I'll come to {price_str}. That's a real step from my side."
    if cue == "Pressure":
        if sentiment == "negative":
            return f"{price_str} is about as far as I'll go; I won't keep chasing this."
        return f"I can do {price_str}, but that's firm - we're running short on time."
    # Hold.
    if sentiment == "positive":
        return f"I'd put it at {price_str}; I think that's fair for both of us."
    if sentiment == "negative":
        return f"My number is {price_str}. I'm not seeing much room to move."
    return f"I can do {price_str}."


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    def _make_scenario(role: str) -> Scenario:
        counter = "seller" if role == "buyer" else "buyer"
        return Scenario(
            episode_id=f"test-{role}",
            regime="overlap",
            family="candid",
            agent_role=role,
            counterpart_role=counter,
            opener="CounterpartOpens",
            p_min=0.0,
            p_max=100.0,
            r_agent=60.0 if role == "buyer" else 40.0,
            r_counterpart=40.0 if role == "buyer" else 60.0,
            kappa_B=0.5,
            eta_B="neutral",
            d0e=0.5,
            seed=123,
        )

    buyer = _make_scenario("buyer")
    seller = _make_scenario("seller")

    # --- System prompts ---
    for sc in (buyer, seller):
        sp = build_system_prompt(sc)
        print("=" * 70)
        print(f"SYSTEM PROMPT ({sc.agent_role}, truncated):")
        print(sp[:400] + " ...")
        # Output schema block must appear.
        assert '"decision": "Offer" | "Accept" | "Reject"' in sp, "schema missing decision line"
        assert '"stance_probs": {' in sp, "schema missing stance_probs"
        assert _OUTPUT_SCHEMA in sp, "exact output schema block missing"
        # All six hard rules must appear.
        for n in range(1, 7):
            assert f"\n{n}. " in sp, f"hard rule {n} missing in {sc.agent_role} prompt"
    # Role-specific monotonicity wording.
    assert "weakly increase" in build_system_prompt(buyer)
    assert "weakly decrease" in build_system_prompt(seller)
    print("\n[OK] system prompts contain output schema + all 6 hard rules.\n")

    # --- User message: no counterpart offer => Accept illegal ---
    um_no_offer = build_user_message(
        buyer, k=1,
        counterpart_offer=None,
        counterpart_message=None,
        own_previous_offer=None,
        history=[],
    )
    parsed = json.loads(um_no_offer)
    for key in ("private_context", "protocol_state", "constraints", "observation", "history"):
        assert key in parsed, f"user message missing key {key}"
    assert "Accept" not in parsed["protocol_state"]["legal_decisions"], "Accept must be illegal w/o offer"
    assert parsed["observation"]["accept_utility"] is None
    print("USER MESSAGE (no offer, truncated):")
    print(um_no_offer[:300] + " ...")

    # --- User message with an offer on the table + history > 6 ---
    long_history = [
        {"round": i, "actor": "counterpart", "decision": "Offer", "price": 70.0 - i, "message": "x"}
        for i in range(1, 10)
    ]
    um_offer = build_user_message(
        buyer, k=4,
        counterpart_offer=58.0,
        counterpart_message="I can do 58.00.",
        own_previous_offer=45.0,
        history=long_history,
    )
    parsed2 = json.loads(um_offer)
    assert "Accept" in parsed2["protocol_state"]["legal_decisions"]
    assert parsed2["observation"]["accept_utility"] == 60.0 - 58.0
    assert len(parsed2["history"]) == 6, "history must be trimmed to last 6"
    print("\n[OK] user message: 5 keys, legality + accept_utility + history window correct.\n")

    # --- Parser: (a) clean JSON ---
    clean = json.dumps({
        "decision": "offer",
        "price": "55.5",
        "message": "Let's meet in the middle.",
        "belief": {
            "r_hat": 42.0,
            "kappa_hat": 0.4,
            "stance_probs": {"conciliatory": 0.2, "neutral": 0.6, "aggressive": 0.2},
        },
    })
    a = parse_agent_action(clean, buyer, own_previous_offer=50.0)
    assert a.parse_error is None
    assert a.decision == "Offer"
    assert a.price == 55.5
    assert a.belief.r_hat == 42.0
    assert a.belief.stance_probs["neutral"] == 0.6
    print("PARSE (clean):", a.decision, a.price, a.belief.r_hat)

    # --- Parser: (b) fenced JSON with prose ---
    fenced = (
        "Sure, here is my move:\n\n```json\n"
        + json.dumps({"decision": "Accept", "price": None, "message": "Deal."})
        + "\n```\nHope that works!"
    )
    b = parse_agent_action(fenced, buyer, own_previous_offer=50.0)
    assert b.parse_error is None
    assert b.decision == "Accept"
    assert b.price is None
    print("PARSE (fenced+prose):", b.decision, b.price)

    # --- Parser: (c) garbage ---
    c = parse_agent_action("this is not json at all, sorry!", buyer, own_previous_offer=None)
    assert c.parse_error is not None, "garbage must set parse_error"
    assert c.decision == "Reject"
    print("PARSE (garbage): parse_error =", repr(c.parse_error))

    # --- detect_violations: out-of-bounds offer ---
    oob = AgentAction(decision="Offer", price=150.0, message="too high", belief=Belief())
    crit, sec = detect_violations(oob, buyer, own_previous_offer=40.0)
    assert "price_bound" in crit, "out-of-bounds offer must flag price_bound"
    print("\nVIOLATIONS (oob offer): critical =", crit, "secondary =", sec)

    # invalid_action from parse_error
    crit2, _ = detect_violations(c, buyer, own_previous_offer=None)
    assert "invalid_action" in crit2

    # --- template_voice samples ---
    print("\nVOICE samples:")
    mv_offer = CounterpartMove(decision="Offer", price=58.0, sentiment="negative", strategy_cue="Pressure")
    mv_accept = CounterpartMove(decision="Accept", price=None, sentiment="positive", strategy_cue="Concede")
    mv_reject = CounterpartMove(decision="Reject", price=None, sentiment="negative", strategy_cue="Pressure")
    print("  Offer :", template_voice(mv_offer, seller, is_opening=False))
    print("  Accept:", template_voice(mv_accept, seller, is_opening=False))
    print("  Reject:", template_voice(mv_reject, seller, is_opening=False))
    assert "58.00" in template_voice(mv_offer, seller, is_opening=False)

    print("\nAll smoke tests passed.")
