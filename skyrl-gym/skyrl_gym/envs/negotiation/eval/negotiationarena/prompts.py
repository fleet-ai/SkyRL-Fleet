"""Agent interface for the NegotiationArena eval harness.

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024,
arXiv:2402.05863; code github.com/vinid/NegotiationArena).

This module builds role-appropriate system prompts (Appendix F, Fig. 21), the
opening user message, a tolerant XML-tag parser for both closed ``<tag>...</tag>``
and unclosed ``<tag> ... <next-tag>`` forms, a bilateral trade-string parser, and a
structural violation detector.

Standard library only.  Does NOT import games.py (economic legality is handled
there).  Safe for flat ``import config`` / ``from config import ...`` style.
"""

from __future__ import annotations

import re
from typing import Optional

from config import (
    ACCEPTING_TAG,
    AgentAction,
    MESSAGE_TAG,
    MY_NAME_TAG,
    PLAYER_ANSWER_TAG,
    PRIVATE_TAGS,
    PROPOSED_TRADE_TAG,
    REASONING_TAG,
    REFUSING_OR_WAIT_TAG,
    RESPONSE_TAG_ORDER,
    SEAT_NAMES,
    Scenario,
    Trade,
)


# ---------------------------------------------------------------------------
# Internal helpers — tolerant XML-like tag extraction
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> str:
    """Extract the content of the first ``<tag>`` occurrence.

    Handles both:
    - Closed form:   ``<tag> content </tag>``
    - Unclosed form: ``<tag> content <next-tag>...`` or ``<tag> content<EOF>``

    Content is stripped of leading/trailing whitespace.  Returns ``""`` if the
    tag is not found.
    """
    escaped = re.escape(tag)
    # Prefer the closed form: lazy match to the explicit close tag.
    m = re.search(
        rf"<{escaped}\s*>(.*?)</{escaped}\s*>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    # Unclosed form: content ends at the next ``<`` or end-of-string.
    m = re.search(
        rf"<{escaped}\s*>(.*?)(?=<|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _has_any_tag(text: str) -> bool:
    """Return True if *text* contains at least one recognised response tag."""
    return any(
        re.search(rf"<{re.escape(tag)}\s*>", text, re.IGNORECASE)
        for tag in RESPONSE_TAG_ORDER
    )


def _name_to_seat(name: str) -> Optional[int]:
    """Map a player-name string to a seat index (0 = RED, 1 = BLUE).

    Accepts ``"Player RED"``, ``"RED"``, ``"Player BLUE"``, ``"BLUE"`` and any
    casing variant.  Returns ``None`` if no match.
    """
    name_up = name.strip().upper()
    if "RED" in name_up:
        return 0
    if "BLUE" in name_up:
        return 1
    # Exact-match fallback against the canonical SEAT_NAMES tuple.
    for i, sn in enumerate(SEAT_NAMES):
        if name_up == sn.upper():
            return i
    return None


def _fmt_resources(bundle: dict) -> str:
    """Render a resource bundle as ``"tok: amt, tok: amt"``."""
    if not bundle:
        return "nothing"
    return ", ".join(f"{tok}: {amt}" for tok, amt in bundle.items())


# ---------------------------------------------------------------------------
# Goal-text factory  (see user-query spec, §2 SPEC.md)
# ---------------------------------------------------------------------------

def _goal_text(scenario: Scenario, seat: int) -> str:
    """Return the goal string for *seat* in *scenario*."""
    game = scenario.game
    opp_name = scenario.seat_name(scenario.other_seat(seat))

    if game == "resource_exchange":
        return (
            "Maximize your total resources. "
            "More resources in general are always better."
        )

    if game == "ultimatum":
        amount = scenario.amount_to_split
        if seat == 0:  # proposer
            return (
                f"You need to split {amount} Dollars with {opp_name}. "
                f"If {opp_name} rejects the split you both lose all. "
                "Keep as much as you can."
            )
        else:  # responder
            return (
                f"{opp_name} will propose how to split {amount} Dollars. "
                "If you reject every offer you both lose all. "
                "Get as many Dollars as you can."
            )

    if game == "sell_buy":
        if seat == 0:  # seller
            cost = scenario.seller_cost
            return (
                "You are the seller of object X. "
                f"It cost you {cost} ZUP to produce. "
                "Sell it for ZUP; maximize the selling price (sell above your cost). "
                "Only you know your production cost."
            )
        else:  # buyer
            wtp = scenario.buyer_willingness
            return (
                "You are the buyer. You want to buy object X. "
                f"You are willing to pay up to {wtp} ZUP. "
                "Minimize the price you pay (pay below your willingness). "
                "Only you know your willingness to pay."
            )

    return "Maximize your outcome."


# ---------------------------------------------------------------------------
# 1.  build_system_prompt  (Appendix F, Fig. 21)
# ---------------------------------------------------------------------------

def build_system_prompt(scenario: Scenario, seat: int) -> str:
    """Build the role-appropriate system prompt for *seat* in *scenario*.

    Reproduces the spirit and wording of NegotiationArena Appendix F, Fig. 21
    (arXiv:2402.05863): rules block (three move shapes), proposal limit, resource
    context, required response-tag block, and optional social-behaviour persona.

    Trade-string format:
        ``Player RED Gives token: amount, ... | Player BLUE Gives token: amount, ...``
    """
    name = scenario.seat_name(seat)
    opp_name = scenario.seat_name(scenario.other_seat(seat))
    initial = scenario.initial_resources[seat]
    resource_tokens_str = ", ".join(scenario.resource_tokens)
    initial_str = _fmt_resources(initial)
    goal = _goal_text(scenario, seat)
    n_proposals = scenario.number_of_proposals

    # One concrete trade-string example using the game's actual tokens.
    tok0 = scenario.resource_tokens[0]
    tok1 = scenario.resource_tokens[-1]
    trade_example = (
        f"Player RED Gives {tok0}: amount | Player BLUE Gives {tok1}: amount"
    )

    # Required-tag block (human-readable enumeration in protocol order).
    tag_lines = "\n".join(
        f"  <{tag}> ..." for tag in RESPONSE_TAG_ORDER
    )

    # Private-tag disclosure note.
    private_note = (
        "Note: the "
        + " and ".join(f"<{t}>" for t in PRIVATE_TAGS)
        + " block(s) are PRIVATE and will NOT be shown to the other player."
    )

    prompt = (
        f"You are playing a negotiation game as {name} against {opp_name}.\n"
        "\n"
        "=== Game Rules ===\n"
        "\n"
        f"1. On each turn you may take EXACTLY ONE of the following actions:\n"
        "\n"
        f"   A. ACCEPT the most recent trade proposed by {opp_name}:\n"
        f"      Set <{PLAYER_ANSWER_TAG}> to {ACCEPTING_TAG}\n"
        f"      Set <{PROPOSED_TRADE_TAG}> to {REFUSING_OR_WAIT_TAG}\n"
        "\n"
        "   B. COUNTER with a new trade proposal of your own:\n"
        f"      Set <{PLAYER_ANSWER_TAG}> to {REFUSING_OR_WAIT_TAG}\n"
        f"      Set <{PROPOSED_TRADE_TAG}> to the full trade string, e.g.:\n"
        f"        {trade_example}\n"
        "\n"
        "   C. WAIT (decline without proposing):\n"
        f"      Set <{PLAYER_ANSWER_TAG}> to {REFUSING_OR_WAIT_TAG}\n"
        f"      Set <{PROPOSED_TRADE_TAG}> to {REFUSING_OR_WAIT_TAG}\n"
        "\n"
        f"2. You may make at most {n_proposals} trade proposals (action B) in total.\n"
        f"   Once you have reached this limit you may only ACCEPT (A) or WAIT (C).\n"
        "\n"
        "3. The trade string format is:\n"
        "   Player RED Gives token: amount, token: amount"
        " | Player BLUE Gives token: amount, token: amount\n"
        "   A side that gives nothing may write WAIT or 0 for each token.\n"
        "\n"
        "=== Here is what you have access to ===\n"
        "\n"
        f"Your name             : {name}\n"
        f"Resources in the game : {resource_tokens_str}\n"
        f"Your initial resources: {initial_str}\n"
        f"Your goal             : {goal}\n"
        "\n"
        "=== Response Format ===\n"
        "\n"
        "All the responses you send should contain the following tags in this exact order:\n"
        "\n"
        f"{tag_lines}\n"
        "\n"
        f"{private_note}\n"
        "\n"
        f"The <{PLAYER_ANSWER_TAG}> must be exactly {ACCEPTING_TAG} or {REFUSING_OR_WAIT_TAG}.\n"
        f"The <{PROPOSED_TRADE_TAG}> must be exactly {REFUSING_OR_WAIT_TAG} or a valid trade string.\n"
        f"Do not include any text before <{MY_NAME_TAG}> or after <{MESSAGE_TAG}>."
    )

    # Append social-behaviour persona if one is assigned to this seat.
    behaviour = scenario.social_behaviour[seat]
    if behaviour:
        prompt += f"\n\n=== Your Negotiation Style ===\n\n{behaviour}"

    return prompt


# ---------------------------------------------------------------------------
# 2.  build_opening_user_message
# ---------------------------------------------------------------------------

def build_opening_user_message(scenario: Scenario, seat: int) -> str:
    """Return the kickoff user message for the given *seat*.

    In practice the runner uses this for the first mover
    (``seat == scenario.first_mover``).  The other seat's first user message is
    the first mover's filtered public reply; see ``filter_public``.
    """
    name = scenario.seat_name(seat)
    return (
        f"You are {name}. "
        "It is your turn to start the negotiation. "
        "Make your first move using the required format."
    )


# ---------------------------------------------------------------------------
# 3.  filter_public  — strip private-tag blocks before forwarding
# ---------------------------------------------------------------------------

def filter_public(raw: str) -> str:
    """Strip every ``PRIVATE_TAGS`` block from *raw*.

    Returns only the public surface (name / resources / goal / player-answer /
    newly-proposed-trade / message) that the other agent should see.

    Robust to both closed ``<tag>...</tag>`` and unclosed ``<tag>...`` forms.
    The result is stripped of leading/trailing whitespace.
    """
    result = raw
    for tag in PRIVATE_TAGS:
        escaped = re.escape(tag)
        # Remove explicit close-tag form first.
        result = re.sub(
            rf"<{escaped}\s*>.*?</{escaped}\s*>",
            "",
            result,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Remove unclosed form: from <tag> up to (not including) the next ``<``.
        result = re.sub(
            rf"<{escaped}\s*>.*?(?=<|\Z)",
            "",
            result,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return result.strip()


# ---------------------------------------------------------------------------
# 4.  parse_trade_string
# ---------------------------------------------------------------------------

def parse_trade_string(s: str, scenario: Scenario) -> Optional[Trade]:
    """Parse a canonical NegotiationArena trade string into a Trade.

    Accepted format (one or two sides separated by ``|``):
        ``Player RED Gives tok: amt, tok: amt | Player BLUE Gives tok: amt``

    - Player-name matching is case-insensitive; ``"Player RED"`` and ``"RED"``
      are both accepted (RED → seat 0, BLUE → seat 1).
    - Amounts are coerced to ``int`` when possible; otherwise kept as ``float``
      so the downstream legality checker can flag non-integer values.
    - ``"nothing"``, ``"WAIT"``, or ``"0"`` on a gives side → empty bundle.
    - Returns ``None`` if the string is empty / ``"WAIT"`` / no parseable side.
    """
    if not s:
        return None
    s = s.strip()
    if s.upper() in ("WAIT", ""):
        return None

    gives: dict[int, dict] = {}

    for raw_side in s.split("|"):
        raw_side = raw_side.strip()
        if not raw_side:
            continue

        # Match "<player-name> Gives <bundle-string>".
        # The optional "Player\s+" prefix covers both "Player RED" and bare "RED".
        m = re.match(
            r"((?:Player\s+)?\w+)\s+Gives\s+(.*)",
            raw_side,
            re.IGNORECASE,
        )
        if not m:
            continue

        seat = _name_to_seat(m.group(1))
        if seat is None:
            continue

        gives_str = m.group(2).strip()
        bundle: dict[str, int | float] = {}

        if gives_str.upper() not in ("NOTHING", "WAIT", "0", ""):
            for item in gives_str.split(","):
                item = item.strip()
                kv = re.match(r"(.+?)\s*:\s*(\S+)", item)
                if not kv:
                    continue
                token = kv.group(1).strip()
                amt_str = kv.group(2).strip()
                try:
                    bundle[token] = int(amt_str)
                except ValueError:
                    try:
                        bundle[token] = float(amt_str)
                    except ValueError:
                        bundle[token] = 0

        gives[seat] = bundle

    if not gives:
        return None

    # Ensure both seats are always present (missing side → empty bundle).
    for seat_idx in (0, 1):
        if seat_idx not in gives:
            gives[seat_idx] = {}

    return Trade(gives=gives)


# ---------------------------------------------------------------------------
# 5.  parse_agent_action
# ---------------------------------------------------------------------------

def parse_agent_action(raw: str, scenario: Scenario, seat: int) -> AgentAction:
    """Tolerant parser for a raw agent response.

    Extracts structured fields from XML-like tags (closed or unclosed forms).

    Returned ``AgentAction`` fields:
    - ``answer``         : ``ACCEPTING_TAG`` / ``REFUSING_OR_WAIT_TAG`` / ``None``
    - ``proposed_trade`` : a ``Trade`` or ``None``
    - ``message``        : free-text to the other player (``""`` if absent)
    - ``reasoning``      : private reasoning block (``""`` if absent)
    - ``raw``            : the original string
    - ``parse_error``    : description string on failure, else ``None``

    Rules:
    - Empty input or no recognised tags → ``parse_error="no tags found"``,
      ``answer=None``.
    - ``<player answer>`` tag absent but other tags present → default ``WAIT``.
    - ``<player answer>`` tag present but content is neither ``ACCEPT`` nor
      ``WAIT`` → ``answer=None`` (triggers ``format_missing_answer`` in
      ``detect_violations``).
    - ``<newly proposed trade>`` is ``WAIT`` or empty → no proposal.
    """
    raw = raw or ""

    if not raw.strip():
        return AgentAction(
            answer=None,
            proposed_trade=None,
            raw=raw,
            parse_error="empty output",
        )

    if not _has_any_tag(raw):
        return AgentAction(
            answer=None,
            proposed_trade=None,
            raw=raw,
            parse_error="no tags found",
        )

    # Extract all fields via the tolerant tag extractor.
    answer_raw = _extract_tag(raw, PLAYER_ANSWER_TAG)
    trade_raw  = _extract_tag(raw, PROPOSED_TRADE_TAG)
    message    = _extract_tag(raw, MESSAGE_TAG)
    reasoning  = _extract_tag(raw, REASONING_TAG)

    # Normalise <player answer> token.
    if not answer_raw:
        # Tag absent (or present but empty) with other tags present → default WAIT.
        answer: Optional[str] = REFUSING_OR_WAIT_TAG
    else:
        ans_up = answer_raw.upper()
        if ans_up == ACCEPTING_TAG:
            answer = ACCEPTING_TAG
        elif ans_up == REFUSING_OR_WAIT_TAG:
            answer = REFUSING_OR_WAIT_TAG
        else:
            # Unrecognisable content → None so format_missing_answer can fire.
            answer = None

    # Parse <newly proposed trade>.
    proposed_trade: Optional[Trade] = None
    trade_str = trade_raw.strip() if trade_raw else ""
    if trade_str and trade_str.upper() not in ("WAIT", ""):
        proposed_trade = parse_trade_string(trade_str, scenario)

    return AgentAction(
        answer=answer,
        proposed_trade=proposed_trade,
        message=message,
        reasoning=reasoning,
        raw=raw,
        parse_error=None,
    )


# ---------------------------------------------------------------------------
# 6.  detect_violations  — structural / format checks only
# ---------------------------------------------------------------------------

def detect_violations(
    action: AgentAction,
    scenario: Scenario,
    seat: int,
    proposals_made: int,
) -> list[str]:
    """Return a list of structural violation tags for *action*.

    Economic legality (non-integer amounts, tokens not owned, negative amounts,
    wrong-token types for the game) is handled in ``games.py`` by the runner.
    Only format/structural violations are reported here:

    ``"invalid_action"``
        ``parse_error`` is set — the response could not be parsed at all.

    ``"format_missing_answer"``
        ``answer`` is ``None`` but ``parse_error`` is ``None`` — the
        ``<player answer>`` tag was found but its content was unrecognisable.

    ``"proposal_after_limit"``
        The action carries a trade proposal and ``proposals_made`` has already
        reached ``scenario.number_of_proposals``.
    """
    violations: list[str] = []

    if action.parse_error is not None:
        violations.append("invalid_action")
        return violations

    if action.answer is None:
        violations.append("format_missing_answer")

    if action.has_proposal and proposals_made >= scenario.number_of_proposals:
        violations.append("proposal_after_limit")

    return violations


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import (
        SELL_BUY_BUYER_BUDGET,
        SELL_BUY_DEFAULT_COST,
        SELL_BUY_DEFAULT_WILLINGNESS,
    )

    # ------------------------------------------------------------------
    # Build a sell_buy scenario for the smoke tests.
    # ------------------------------------------------------------------
    sc = Scenario(
        episode_id="sell_buy-smoke-0000",
        game="sell_buy",
        focal_seat=1,
        first_mover=0,
        initial_resources=({"X": 1}, {"ZUP": SELL_BUY_BUYER_BUDGET}),
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,
        number_of_proposals=5,
        seed=0,
        seller_cost=SELL_BUY_DEFAULT_COST,
        buyer_willingness=SELL_BUY_DEFAULT_WILLINGNESS,
    )

    # ------------------------------------------------------------------
    # 1. Seller system prompt (seat 0)
    # ------------------------------------------------------------------
    seller_sp = build_system_prompt(sc, seat=0)

    assert f"<{MY_NAME_TAG}>" in seller_sp, \
        f"seller prompt missing <{MY_NAME_TAG}> tag"
    assert "Player RED" in seller_sp, \
        "seller prompt missing Player RED name"
    assert str(SELL_BUY_DEFAULT_COST) in seller_sp, \
        "seller prompt missing seller_cost"
    assert "Only you know" in seller_sp, \
        "seller prompt missing incomplete-info note"
    assert "seller" in seller_sp.lower(), \
        "seller prompt missing 'seller' in goal text"
    for tag in RESPONSE_TAG_ORDER:
        assert f"<{tag}>" in seller_sp, \
            f"seller prompt missing required tag <{tag}>"
    print("OK seller system prompt: tags + goal text present.")

    # ------------------------------------------------------------------
    # 2. Buyer system prompt (seat 1)
    # ------------------------------------------------------------------
    buyer_sp = build_system_prompt(sc, seat=1)

    assert f"<{MY_NAME_TAG}>" in buyer_sp, \
        f"buyer prompt missing <{MY_NAME_TAG}> tag"
    assert "Player BLUE" in buyer_sp, \
        "buyer prompt missing Player BLUE name"
    assert str(SELL_BUY_DEFAULT_WILLINGNESS) in buyer_sp, \
        "buyer prompt missing buyer_willingness"
    assert "Only you know" in buyer_sp, \
        "buyer prompt missing incomplete-info note"
    assert "buyer" in buyer_sp.lower(), \
        "buyer prompt missing 'buyer' in goal text"
    for tag in RESPONSE_TAG_ORDER:
        assert f"<{tag}>" in buyer_sp, \
            f"buyer prompt missing required tag <{tag}>"
    print("OK buyer system prompt: tags + goal text present.")

    # Social-behaviour variant: persona appended.
    sc_persona = Scenario(
        episode_id="sell_buy-smoke-persona",
        game="sell_buy",
        focal_seat=0,
        first_mover=0,
        initial_resources=({"X": 1}, {"ZUP": 100}),
        money_token="ZUP",
        resource_tokens=("X", "ZUP"),
        max_turns=10,
        number_of_proposals=5,
        seed=1,
        seller_cost=40,
        buyer_willingness=60,
        social_behaviour=("You are cunning and sly.", ""),
    )
    persona_sp = build_system_prompt(sc_persona, seat=0)
    assert "cunning" in persona_sp.lower(), \
        "social_behaviour not appended to system prompt"
    print("OK social_behaviour appended when non-empty.")

    # ------------------------------------------------------------------
    # 3. parse_trade_string: RED→seat0, BLUE→seat1
    # ------------------------------------------------------------------
    t = parse_trade_string("Player RED Gives X: 1 | Player BLUE Gives ZUP: 45", sc)
    assert t is not None, "parse_trade_string returned None for valid trade"
    assert t.gives[0] == {"X": 1}, \
        f"gives[0] wrong: {t.gives[0]}"
    assert t.gives[1] == {"ZUP": 45}, \
        f"gives[1] wrong: {t.gives[1]}"
    print("OK parse_trade_string: gives[0]=={'X':1}, gives[1]=={'ZUP':45}.")

    # WAIT string → None
    assert parse_trade_string("WAIT", sc) is None, \
        "parse_trade_string('WAIT') should return None"
    assert parse_trade_string("", sc) is None, \
        "parse_trade_string('') should return None"

    # Non-integer amount preserved as float.
    t_float = parse_trade_string(
        "Player RED Gives X: 1.5 | Player BLUE Gives ZUP: 0", sc
    )
    assert t_float is not None
    assert isinstance(t_float.gives[0].get("X"), float), \
        "non-integer amount must be kept as float"
    print("OK parse_trade_string: non-integer preserved as float.")

    # ------------------------------------------------------------------
    # 4. parse_agent_action: clean COUNTER (unclosed tags)
    # ------------------------------------------------------------------
    clean_raw = (
        "<my name> Player RED\n"
        "<resources in hand> X: 1\n"
        "<goal> Maximize your selling price.\n"
        "<reason> I will start high and gauge their response.\n"
        "<player answer> WAIT\n"
        "<newly proposed trade> Player RED Gives X: 1 | Player BLUE Gives ZUP: 55\n"
        "<message> Here is my opening offer."
    )
    action = parse_agent_action(clean_raw, sc, seat=0)
    assert action.parse_error is None, \
        f"Unexpected parse_error: {action.parse_error}"
    assert action.answer == REFUSING_OR_WAIT_TAG, \
        f"Expected WAIT, got {action.answer!r}"
    assert action.proposed_trade is not None, \
        "Expected a proposed trade"
    assert action.proposed_trade.gives[0] == {"X": 1}, \
        f"Trade gives[0] wrong: {action.proposed_trade.gives[0]}"
    assert action.proposed_trade.gives[1] == {"ZUP": 55}, \
        f"Trade gives[1] wrong: {action.proposed_trade.gives[1]}"
    assert "start high" in action.reasoning, \
        "reasoning not extracted correctly"
    assert "opening offer" in action.message, \
        "message not extracted correctly"
    print("OK parse_agent_action: clean COUNTER response parsed correctly.")

    # Closed-tag variant for <reason>.
    closed_raw = clean_raw.replace(
        "<reason> I will start high and gauge their response.\n",
        "<reason> I will start high and gauge their response. </reason>\n",
    )
    action_closed = parse_agent_action(closed_raw, sc, seat=0)
    assert action_closed.parse_error is None
    assert "start high" in action_closed.reasoning, \
        "closed-tag reasoning not extracted"
    print("OK parse_agent_action: closed-tag reasoning extracted correctly.")

    # ------------------------------------------------------------------
    # 5. parse_agent_action: ACCEPT
    # ------------------------------------------------------------------
    accept_raw = (
        "<my name> Player BLUE\n"
        "<resources in hand> ZUP: 100\n"
        "<goal> Minimize cost.\n"
        "<reason> This price is below my willingness to pay.\n"
        "<player answer> ACCEPT\n"
        "<newly proposed trade> WAIT\n"
        "<message> I accept your offer!"
    )
    accept_action = parse_agent_action(accept_raw, sc, seat=1)
    assert accept_action.parse_error is None
    assert accept_action.answer == ACCEPTING_TAG, \
        f"Expected ACCEPT, got {accept_action.answer!r}"
    assert accept_action.proposed_trade is None, \
        "ACCEPT+WAIT trade should yield no proposed_trade"
    assert accept_action.is_accept
    print("OK parse_agent_action: ACCEPT response parsed correctly.")

    # ------------------------------------------------------------------
    # 6. parse_agent_action: garbage → parse_error set
    # ------------------------------------------------------------------
    garbage_action = parse_agent_action(
        "this is completely unstructured text with no tags!", sc, seat=0
    )
    assert garbage_action.parse_error is not None, \
        "garbage input must set parse_error"
    assert garbage_action.answer is None
    print(
        f"OK parse_agent_action: garbage sets parse_error={garbage_action.parse_error!r}."
    )

    # Empty string.
    empty_action = parse_agent_action("", sc, seat=0)
    assert empty_action.parse_error == "empty output"
    print("OK parse_agent_action: empty string → parse_error='empty output'.")

    # ------------------------------------------------------------------
    # 7. filter_public: reason stripped, message and name kept
    # ------------------------------------------------------------------
    raw_with_reason = (
        "<my name> Player RED\n"
        "<resources in hand> X: 1\n"
        "<goal> Sell high.\n"
        "<reason> SECRET: I will start high and concede slowly.\n"
        "<player answer> WAIT\n"
        "<newly proposed trade> Player RED Gives X: 1 | Player BLUE Gives ZUP: 70\n"
        "<message> Let's negotiate."
    )
    public = filter_public(raw_with_reason)
    assert "SECRET" not in public, \
        "filter_public must strip <reason> content"
    assert "<reason>" not in public.lower(), \
        "filter_public must remove the <reason> tag itself"
    assert "Let's negotiate" in public, \
        "filter_public must keep <message> content"
    assert "Player RED" in public, \
        "filter_public must keep <my name> content"
    print("OK filter_public: reason stripped, message/name retained.")

    # Closed-reason variant: explicit </reason> close tag.
    raw_closed_reason = (
        "<my name> Player RED\n"
        "<resources in hand> X: 1\n"
        "<goal> Sell high.\n"
        "<reason> SECRET stuff </reason>\n"
        "<player answer> WAIT\n"
        "<newly proposed trade> WAIT\n"
        "<message> Nothing to propose yet."
    )
    pub2 = filter_public(raw_closed_reason)
    assert "SECRET" not in pub2, \
        "filter_public must strip closed-form <reason> content"
    assert "Nothing to propose" in pub2, \
        "filter_public must keep <message> after closed-form strip"
    print("OK filter_public: closed-tag reason stripped correctly.")

    # ------------------------------------------------------------------
    # 8. detect_violations
    # ------------------------------------------------------------------

    # invalid_action: garbage parse.
    viols_inv = detect_violations(garbage_action, sc, seat=0, proposals_made=0)
    assert "invalid_action" in viols_inv, \
        f"Expected invalid_action, got {viols_inv}"
    print("OK detect_violations: invalid_action flagged for garbage parse.")

    # proposal_after_limit: proposals_made == number_of_proposals.
    viols_limit = detect_violations(action, sc, seat=0, proposals_made=5)
    assert "proposal_after_limit" in viols_limit, \
        f"Expected proposal_after_limit, got {viols_limit}"
    print("OK detect_violations: proposal_after_limit flagged at limit.")

    # No violations for clean action within limit.
    viols_clean = detect_violations(action, sc, seat=0, proposals_made=1)
    assert viols_clean == [], \
        f"Expected no violations, got {viols_clean}"
    print("OK detect_violations: clean action within limit → no violations.")

    # format_missing_answer: unrecognisable <player answer> content.
    bad_answer_raw = (
        "<my name> Player RED\n"
        "<resources in hand> X: 1\n"
        "<goal> Sell high.\n"
        "<reason> Thinking.\n"
        "<player answer> SURE_LET_US_DEAL\n"
        "<newly proposed trade> WAIT\n"
        "<message> Hello."
    )
    bad_action = parse_agent_action(bad_answer_raw, sc, seat=0)
    assert bad_action.parse_error is None, \
        "bad answer tag should not set parse_error (tags are present)"
    assert bad_action.answer is None, \
        "unrecognisable answer content should yield answer=None"
    viols_fmt = detect_violations(bad_action, sc, seat=0, proposals_made=0)
    assert "format_missing_answer" in viols_fmt, \
        f"Expected format_missing_answer, got {viols_fmt}"
    print("OK detect_violations: format_missing_answer flagged for bad answer token.")

    print("\nprompts.py smoke test passed.")
