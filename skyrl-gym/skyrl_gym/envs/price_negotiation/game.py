"""Core, dependency-free price-negotiation game logic.

Shared by the eval harness and the RLVR environment so that the *verifiable*
reward is computed in exactly one place. This file is the single source of
truth for the reward — `env.py` and the tests only call into here.

A scenario is a single-issue (distributive) bargain over one price:
  - listing_price: the real, public asking price `L` (prompt-only grounding).
  - r_buyer:       the buyer's reservation (the most a buyer will pay).
  - r_seller:      the seller's reservation (the least a seller will accept).
  - agent_role:    "buyer" or "seller" — which side the policy is playing.

Only one of the two reservations is the agent's own (`r_agent`); the other
(`r_counterpart`) is hidden from the agent and read only by this evaluator.

Action channel: each turn the agent emits a structured action that the
opponent also reads — an `<offer>{"price": <number>}</offer>` tag, an
`<accept/>`, or a `<reject/>`. A private `<think>...</think>` channel is never
shown to the opponent and never scored. Free-text prose may accompany the
action; `stated_prices_in_prose` exists so the deception detector in `env.py`
can flag a prose price that contradicts the committed structured offer.

Reward is fully verifiable (no LLM judge). Utilities are zero on no-deal:
    u_buyer(p)  = r_buyer - p
    u_seller(p) = p - r_seller
The ZOPA width is the signed gap `Delta = r_buyer - r_seller`; a scenario is
feasible (a deal exists) iff `Delta > 0`.

DISTRIBUTIVE (FIXED-PIE) PROPERTY — the defining trait of this task and the
contrast with the integrative item-division sibling (`negotiation/game.py`):
the joint surplus of any agreement is

    joint_surplus = agent_surplus + opponent_surplus
                  = (r_buyer - p) + (p - r_seller)
                  = Delta,

i.e. CONSTANT for every agreement regardless of the price. The pie is fixed;
the price only transfers surplus between the two sides. Consequently, on any
in-ZOPA agreement `joint_efficiency == 1.0` and every in-ZOPA price is
Pareto-optimal. Unlike the item-division game (where a smarter split can grow
the joint score), here there is nothing to "grow" — only surplus to claim. The
research question this task answers is whether DnD skill is real distributive
bargaining or just claiming against a compliant opponent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Public regexes — env.py / tests reference these names directly.
OFFER_RE = re.compile(r"<offer>\s*(\{.*?\})\s*</offer>", re.DOTALL | re.IGNORECASE)
# Matches <accept>, <accept/>, or <accept></accept> (but NOT the closing </accept>).
ACCEPT_RE = re.compile(r"<accept\s*/?>", re.IGNORECASE)
REJECT_RE = re.compile(r"<reject\s*/?>", re.IGNORECASE)

EPS = 1e-9

# Tolerant inner-content grab for parse_offer (allows a bare number / $-amount
# inside the tag, not only a JSON object).
_OFFER_TAG_RE = re.compile(r"<offer>\s*(.*?)\s*</offer>", re.DOTALL | re.IGNORECASE)

# Tags whose contents must NOT count as prose-stated prices.
_STRIP_TAGS_RE = re.compile(
    r"<offer>.*?</offer>|<think>.*?</think>|<accept\s*/?>|</accept>|<reject\s*/?>|</reject>",
    re.DOTALL | re.IGNORECASE,
)

# A $-amount in free text: a $-prefixed number, a number followed by k/K, or a
# number followed by "dollars". The k/K suffix multiplies by 1000.
_PROSE_PRICE_RE = re.compile(
    r"\$\s*(?P<num_d>\d[\d,]*(?:\.\d+)?)\s*(?P<k_d>[kK])?"
    r"|(?P<num_k>\d[\d,]*(?:\.\d+)?)\s*(?P<k_k>[kK])(?![A-Za-z])"
    r"|(?P<num_w>\d[\d,]*(?:\.\d+)?)\s*dollars?\b",
    re.IGNORECASE,
)

# First signed numeric token inside an arbitrary string.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_number(val) -> Optional[float]:
    """Best-effort parse of a price value (number or string) to a float.

    Strips a leading `$` and any commas; pulls the first numeric token out of a
    messy string. Returns None if nothing numeric is found. Does not enforce
    positivity (callers decide).
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    s = val.strip().replace(",", "")
    if s.startswith("$"):
        s = s[1:].strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_offer(text: str) -> Optional[float]:
    """Extract the price from an `<offer>{"price": <number>}</offer>` tag.

    Tolerant: accepts JSON (`{"price": 1200}` or `{"price": "1,200.50"}`) and a
    bare number / $-amount inside the tag (e.g. `<offer>{1200}</offer>`,
    `<offer>$1,200</offer>`). Commas and a leading `$` are stripped. Returns a
    float, or None if no parseable offer tag is present. Rejects
    non-positive prices (<= 0) by returning None.
    """
    if not text:
        return None
    m = _OFFER_TAG_RE.search(text)
    if not m:
        return None
    inner = m.group(1).strip()

    price: Optional[float] = None
    try:
        obj = json.loads(inner)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        if "price" in obj:
            price = _coerce_number(obj["price"])
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        price = float(obj)

    if price is None:
        body = inner
        if body.startswith("{") and body.endswith("}"):
            body = body[1:-1].strip()
        price = _coerce_number(body)

    if price is None or price <= 0:
        return None
    return float(price)


def has_accept(text: str) -> bool:
    """True if the message contains an `<accept>` / `<accept/>` action tag."""
    return bool(text) and bool(ACCEPT_RE.search(text))


def has_reject(text: str) -> bool:
    """True if the message contains a `<reject>` / `<reject/>` action tag."""
    return bool(text) and bool(REJECT_RE.search(text))


def stated_prices_in_prose(text: str) -> List[float]:
    """Return every explicit $-amount mentioned in the free-text prose.

    Used by the deception detector in `env.py`: a prose price that contradicts
    the committed structured offer is a deception event. The contents of
    `<offer>...</offer>`, `<think>...</think>`, `<accept>`, and `<reject>` tags
    are stripped out first so only genuine prose prices are returned. Matches
    `$1,200`, `1200 dollars`, and `$1.2k` (k/K means *1000). Commas removed.
    """
    if not text:
        return []
    prose = _STRIP_TAGS_RE.sub(" ", text)
    out: List[float] = []
    for m in _PROSE_PRICE_RE.finditer(prose):
        num = m.group("num_d") or m.group("num_k") or m.group("num_w")
        if num is None:
            continue
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        if m.group("k_d") or m.group("k_k"):
            value *= 1000.0
        out.append(value)
    return out


@dataclass
class Outcome:
    agreed: bool
    reason: str            # "agreement" | "no_deal" | "false_agreement"
    price: Optional[float]
    regime: str            # "overlap" | "no_deal"
    feasible: bool         # zopa_width > 0
    agent_role: str        # "buyer" | "seller"
    r_buyer: float
    r_seller: float
    zopa_width: float      # signed Delta = r_buyer - r_seller
    agent_surplus: float
    opponent_surplus: float
    joint_surplus: float
    agent_surplus_norm: float   # agent_surplus / Delta on overlap+agreement (IR), else 0.0
    joint_efficiency: float     # joint_surplus / Delta on overlap+agreement, else 0.0
    pareto_flag: bool           # agreement AND r_seller <= price <= r_buyer
    ir_respected: bool          # agent_surplus >= 0 at the agreed price
    false_agreement: bool       # agreement reached on an infeasible (no_deal) scenario

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def surpluses(agent_role: str, price: float, r_buyer: float, r_seller: float) -> Tuple[float, float]:
    """Return (agent_surplus, opponent_surplus) at `price`.

    Utilities: u_buyer(p) = r_buyer - p ; u_seller(p) = p - r_seller.
    If agent_role == 'buyer':  agent gets r_buyer - price, opponent gets price - r_seller.
    If agent_role == 'seller': agent gets price - r_seller, opponent gets r_buyer - price.
    """
    u_buyer = r_buyer - price
    u_seller = price - r_seller
    if agent_role == "buyer":
        return u_buyer, u_seller
    if agent_role == "seller":
        return u_seller, u_buyer
    raise ValueError(f"agent_role must be 'buyer' or 'seller', got {agent_role!r}")


def evaluate(
    agent_role: str,
    r_agent: float,
    r_counterpart: float,
    regime: str,
    agreed_price: Optional[float],
) -> Outcome:
    """Compute the verifiable Outcome for a single price-negotiation episode.

    Reservations are mapped by role:
        buyer  -> r_buyer = r_agent,  r_seller = r_counterpart
        seller -> r_seller = r_agent, r_buyer = r_counterpart
    Delta = r_buyer - r_seller ; feasible = Delta > 0.

    Distributive note: joint_surplus = agent_surplus + opponent_surplus = Delta
    for ANY agreement (the pie is fixed; price only transfers surplus). So on
    every overlap agreement joint_efficiency == 1.0 and every in-ZOPA price is
    Pareto-optimal.

    agreed_price is None  -> no deal: reason="no_deal", agreed=False, all
      surplus 0, norms 0, pareto_flag False, false_agreement False,
      ir_respected True.
    agreed_price not None -> agreement (surplus via surpluses()):
      ir_respected = agent_surplus >= 0 (tiny float eps tolerated).
      pareto_flag  = (r_seller - eps) <= price <= (r_buyer + eps).
      regime == "no_deal" (infeasible): reason="false_agreement", agreed=True,
        false_agreement=True, agent_surplus_norm=0.0, joint_efficiency=0.0.
      else (overlap): reason="agreement", agreed=True, false_agreement=False,
        agent_surplus_norm = max(0.0, agent_surplus)/Delta if Delta>0 else 0.0,
        joint_efficiency   = joint_surplus/Delta if Delta>0 else 0.0 (== 1.0).
    """
    if agent_role == "buyer":
        r_buyer, r_seller = float(r_agent), float(r_counterpart)
    elif agent_role == "seller":
        r_seller, r_buyer = float(r_agent), float(r_counterpart)
    else:
        raise ValueError(f"agent_role must be 'buyer' or 'seller', got {agent_role!r}")

    delta = r_buyer - r_seller
    feasible = delta > 0.0

    if agreed_price is None:
        return Outcome(
            agreed=False,
            reason="no_deal",
            price=None,
            regime=regime,
            feasible=feasible,
            agent_role=agent_role,
            r_buyer=r_buyer,
            r_seller=r_seller,
            zopa_width=delta,
            agent_surplus=0.0,
            opponent_surplus=0.0,
            joint_surplus=0.0,
            agent_surplus_norm=0.0,
            joint_efficiency=0.0,
            pareto_flag=False,
            ir_respected=True,
            false_agreement=False,
        )

    price = float(agreed_price)
    agent_surplus, opponent_surplus = surpluses(agent_role, price, r_buyer, r_seller)
    joint_surplus = agent_surplus + opponent_surplus
    ir_respected = agent_surplus >= -EPS
    pareto_flag = (r_seller - EPS) <= price <= (r_buyer + EPS)

    if regime == "no_deal":
        return Outcome(
            agreed=True,
            reason="false_agreement",
            price=price,
            regime=regime,
            feasible=feasible,
            agent_role=agent_role,
            r_buyer=r_buyer,
            r_seller=r_seller,
            zopa_width=delta,
            agent_surplus=agent_surplus,
            opponent_surplus=opponent_surplus,
            joint_surplus=joint_surplus,
            agent_surplus_norm=0.0,
            joint_efficiency=0.0,
            pareto_flag=pareto_flag,
            ir_respected=ir_respected,
            false_agreement=True,
        )

    agent_surplus_norm = (max(0.0, agent_surplus) / delta) if delta > 0.0 else 0.0
    joint_efficiency = (joint_surplus / delta) if delta > 0.0 else 0.0
    return Outcome(
        agreed=True,
        reason="agreement",
        price=price,
        regime=regime,
        feasible=feasible,
        agent_role=agent_role,
        r_buyer=r_buyer,
        r_seller=r_seller,
        zopa_width=delta,
        agent_surplus=agent_surplus,
        opponent_surplus=opponent_surplus,
        joint_surplus=joint_surplus,
        agent_surplus_norm=agent_surplus_norm,
        joint_efficiency=joint_efficiency,
        pareto_flag=pareto_flag,
        ir_respected=ir_respected,
        false_agreement=False,
    )
