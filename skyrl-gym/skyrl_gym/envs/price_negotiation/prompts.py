"""Prompt construction for the single-issue price-negotiation task (shared by eval + RLVR env).

Distributive (one price on the line) counterpart to the integrative item-division
prompts in ``envs/negotiation/prompts.py``. The agent sees ONLY its own private
reservation (its walk-away), the public price bounds, the real item text, and the
listing price. The counterpart's reservation and the ZOPA are evaluator-side ground
truth and MUST NEVER appear in the rendered prompt.
"""

from __future__ import annotations

from typing import List

# Maximum characters of the (real, free-text) item description we surface in the
# prompt. The description is grounding flavor only — the reward never depends on it,
# so a long listing body just wastes context.
_DESC_MAX_CHARS = 400


SYSTEM_TEMPLATE = """\
You are negotiating the price of a single item against another party. This is a one-price
deal: every dollar you give up on price, the other party gains, and vice versa.

The item on the table:
  - Category: {item_category}
  - Title: {item_title}
  - Description: {item_description}
  - Listing price: {listing_price}

Your position:
{role_block}
- Every offer (yours and theirs) must be a price within the public range {bounds}.
- The listing price ({listing_price}) is just the seller's opening ask, not a limit on what
  can be proposed inside that range.

How it works:
- Take turns exchanging short messages (1-3 sentences) and concrete prices.
- To make a concrete OFFER, end your message with a single line of the exact form:
  <offer>{{"price": {offer_example}}}</offer>
  The number is the price you are proposing for the item.
- To ACCEPT the other party's most recent offer, reply with a line containing exactly:
  <accept>
  The deal then closes at the price they last offered.
- To WALK AWAY with no deal, reply with a line containing exactly:
  <reject>
- The price the other party (and the scoring) acts on is the number inside your <offer> tag.
  Do not state one price in prose and commit a different number in the tag.

Closing the deal — READ CAREFULLY:
- The negotiation ends the instant an offer is accepted; the deal closes at that accepted price.
- You have at most {max_turns} messages. If no offer is accepted within that limit, or either
  party walks away, it is NO DEAL and BOTH of you get nothing.
- A bad deal is worse than no deal: NEVER offer or accept a price worse than your walk-away of
  {reservation}. {walkaway_rule} Walking away with <reject> is the CORRECT move when no
  acceptable price exists — do not accept a price worse than your walk-away just to avoid no-deal.
- When an acceptable price IS reachable, close it: make an offer the other party will take, or
  accept a reasonable one. No deal earns nothing.

Be efficient and decisive."""


_BUYER_BLOCK = """\
- You want to BUY this item and pay as LITTLE as possible.
- Your maximum acceptable price (your walk-away) is {reservation}. NEVER offer or accept a price
  ABOVE this; if you cannot get a price at or below it, you should walk away.
- The seller has their own hidden minimum price that you cannot see."""

_SELLER_BLOCK = """\
- You are SELLING this item and want to get as MUCH as possible.
- Your minimum acceptable price (your walk-away) is {reservation}. NEVER offer or accept a price
  BELOW this; if you cannot get a price at or above it, you should walk away.
- The buyer has their own hidden maximum price that you cannot see."""

_BUYER_WALKAWAY = "As the buyer, that means do not pay MORE than {reservation}."
_SELLER_WALKAWAY = "As the seller, that means do not sell for LESS than {reservation}."


def _fmt_price(price: float) -> str:
    """Render a price as a readable whole-dollar string, e.g. ``$1,200``."""
    return f"${round(price):,}"


def _trim_description(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(no description provided)"
    if len(text) > _DESC_MAX_CHARS:
        text = text[:_DESC_MAX_CHARS].rstrip() + "..."
    return text


def build_system_prompt(
    *,
    agent_role: str,        # "buyer" or "seller"
    item_category: str,
    item_title: str,
    item_description: str,
    listing_price: float,
    r_agent: float,         # the agent's OWN reservation (its hard walk-away limit)
    p_min: float,
    p_max: float,
    max_turns: int,
) -> str:
    role = agent_role.strip().lower()
    if role not in ("buyer", "seller"):
        raise ValueError(f"agent_role must be 'buyer' or 'seller', got {agent_role!r}")

    reservation = _fmt_price(r_agent)
    if role == "buyer":
        role_block = _BUYER_BLOCK.format(reservation=reservation)
        walkaway_rule = _BUYER_WALKAWAY.format(reservation=reservation)
    else:
        role_block = _SELLER_BLOCK.format(reservation=reservation)
        walkaway_rule = _SELLER_WALKAWAY.format(reservation=reservation)

    # A concrete number for the <offer> example: midpoint of the public bounds,
    # rounded to a whole dollar, so the model anchors on the bare-number JSON form.
    offer_example = round((p_min + p_max) / 2)

    return SYSTEM_TEMPLATE.format(
        item_category=item_category,
        item_title=item_title,
        item_description=_trim_description(item_description),
        listing_price=_fmt_price(listing_price),
        role_block=role_block,
        bounds=f"[{_fmt_price(p_min)}, {_fmt_price(p_max)}]",
        offer_example=offer_example,
        reservation=reservation,
        walkaway_rule=walkaway_rule,
        max_turns=max_turns,
    )


OPENING_USER_MSG = (
    "You speak first. Open the negotiation with a brief message and, when ready, a concrete <offer>."
)

PARTNER_FIRST_MSG = (
    "The other party will speak first. Wait for their message, then respond."
)
