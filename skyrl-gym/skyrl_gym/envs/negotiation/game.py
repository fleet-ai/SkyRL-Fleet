"""Core, dependency-free negotiation game logic.

Shared by the eval harness and the (future) RLVR environment so that the
*verifiable* reward is computed in exactly one place.

A scenario is defined by:
  - item_names: e.g. ["book", "hat", "ball"] or ["food", "water", "firewood"]
  - counts:     how many of each item exist (shared pool), e.g. [1, 4, 1]
  - you_values / them_values: each agent's private per-unit point values

Each agent ends the dialogue by declaring how many of each item *they* take,
parsed from a `<deal>{...}</deal>` tag. The deal is an agreement iff the two
claims exactly partition every pool. Otherwise it is a no-deal (reward 0).
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEAL_RE = re.compile(r"<deal>\s*(\{.*?\})\s*</deal>", re.DOTALL | re.IGNORECASE)
PROPOSE_RE = re.compile(r"<propose>\s*(\{.*?\})\s*</propose>", re.DOTALL | re.IGNORECASE)
# Matches <accept>, <accept/>, or <accept></accept> (but NOT the closing </accept>).
ACCEPT_RE = re.compile(r"<accept\s*/?>", re.IGNORECASE)
_KV_RE = re.compile(r"['\"]?([A-Za-z_]+)['\"]?\s*[:=]\s*(\d+)")


def _extract_counts(blob: str, item_names: List[str]) -> Optional[List[int]]:
    """Parse a `{item: count}` blob into a list aligned to `item_names`."""
    parsed: Dict[str, int] = {}
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    parsed[str(k).strip().lower()] = int(v)
                except (ValueError, TypeError):
                    continue
    except json.JSONDecodeError:
        for k, v in _KV_RE.findall(blob):
            parsed[k.strip().lower()] = int(v)
    if not parsed:
        return None
    return [max(0, parsed.get(name.lower(), 0)) for name in item_names]


def parse_deal(text: str, item_names: List[str]) -> Optional[List[int]]:
    """Extract an agent's claimed take from a `<deal>{...}</deal>` tag.

    Used by the dual-tag protocol. Returns None if no parseable tag is present.
    """
    if not text:
        return None
    m = DEAL_RE.search(text)
    return _extract_counts(m.group(1), item_names) if m else None


def parse_proposal(text: str, item_names: List[str]) -> Optional[List[int]]:
    """Extract a proposer's claimed take from a `<propose>{...}</propose>` tag.

    Used by the single-proposer protocol: the proposer lists how many of each
    item THEY keep; the partner implicitly gets the rest. Returns None if no
    parseable proposal is present.
    """
    if not text:
        return None
    m = PROPOSE_RE.search(text)
    return _extract_counts(m.group(1), item_names) if m else None


def has_accept(text: str) -> bool:
    """True if the message contains an `<accept>` tag (single-proposer protocol)."""
    return bool(text) and bool(ACCEPT_RE.search(text))


def score_of(take: List[int], values: List[int]) -> int:
    return sum(t * v for t, v in zip(take, values))


@dataclass
class Outcome:
    agreed: bool
    reason: str  # "agreement" | "no_deal" | "conflict" | "incomplete"
    you_take: Optional[List[int]]
    them_take: Optional[List[int]]
    you_score: int = 0
    them_score: int = 0
    you_max: int = 0
    them_max: int = 0
    you_norm: float = 0.0          # you_score / you_max  (terminal outcome reward)
    them_norm: float = 0.0
    joint_score: int = 0
    pareto_optimal: bool = False   # is the achieved split on the Pareto frontier?
    max_joint: int = 0             # best achievable joint score for this scenario
    joint_efficiency: float = 0.0  # joint_score / max_joint
    pareto_bonus: float = 0.0      # 1.0 if pareto_optimal else 0.0 (on agreement)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def all_allocations(counts: List[int]):
    """Yield every way to give `take` units of each item to 'you' (0..count)."""
    ranges = [range(c + 1) for c in counts]
    for combo in itertools.product(*ranges):
        yield list(combo)


def pareto_analysis(counts, you_values, them_values, you_take, them_take):
    """Return (is_pareto_optimal, max_joint) for the achieved allocation.

    The feasible set is every full partition of the pools. An allocation is
    Pareto-optimal if no other partition makes one agent better off without
    making the other worse off.
    """
    you_score = score_of(you_take, you_values)
    them_score = score_of(them_take, them_values)
    max_joint = 0
    dominated = False
    for ya in all_allocations(counts):
        tb = [counts[i] - ya[i] for i in range(len(counts))]
        sa = score_of(ya, you_values)
        sb = score_of(tb, them_values)
        if sa + sb > max_joint:
            max_joint = sa + sb
        if (sa >= you_score and sb >= them_score) and (sa > you_score or sb > them_score):
            dominated = True
    return (not dominated), max_joint


def evaluate(
    counts: List[int],
    you_values: List[int],
    them_values: List[int],
    you_take: Optional[List[int]],
    them_take: Optional[List[int]],
) -> Outcome:
    """Compute the verifiable outcome from two parsed claims.

    `you_take` / `them_take` are each agent's claim of what THEY keep.
    """
    you_max = score_of(counts, you_values)
    them_max = score_of(counts, them_values)

    if you_take is None or them_take is None:
        return Outcome(False, "no_deal", you_take, them_take, you_max=you_max, them_max=them_max)

    n = len(counts)
    sums = [you_take[i] + them_take[i] for i in range(n)]
    overclaim = any(sums[i] > counts[i] for i in range(n))
    exact = all(sums[i] == counts[i] for i in range(n))

    if not exact:
        reason = "conflict" if overclaim else "incomplete"
        return Outcome(False, reason, you_take, them_take, you_max=you_max, them_max=them_max)

    you_score = score_of(you_take, you_values)
    them_score = score_of(them_take, them_values)
    is_pareto, max_joint = pareto_analysis(counts, you_values, them_values, you_take, them_take)
    joint = you_score + them_score
    return Outcome(
        agreed=True,
        reason="agreement",
        you_take=you_take,
        them_take=them_take,
        you_score=you_score,
        them_score=them_score,
        you_max=you_max,
        them_max=them_max,
        you_norm=(you_score / you_max) if you_max else 0.0,
        them_norm=(them_score / them_max) if them_max else 0.0,
        joint_score=joint,
        pareto_optimal=is_pareto,
        max_joint=max_joint,
        joint_efficiency=(joint / max_joint) if max_joint else 0.0,
        pareto_bonus=1.0 if is_pareto else 0.0,
    )
