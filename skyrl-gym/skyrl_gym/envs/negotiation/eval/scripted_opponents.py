"""Scripted (non-LLM) negotiation opponents for diagnostic probes.

These are deterministic, API-free counterparts used to *measure* a policy's
behaviour in a controlled setting, rather than to provide a realistic partner.
Because they make no network calls they are effectively free to run.

``ConcederOpponent`` ("always concedes") is a deliberately weak pushover for the
exploitation probe. It never advocates for itself: it waits for the measured
policy (seat A) to claim, then takes the exact *complement* — the leftovers — so
any policy claim that fits inside the pool is finalized verbatim. The policy's
resulting take is therefore a clean, verifiable readout of how hard it squeezes a
partner that put up no resistance whatsoever.
"""

from __future__ import annotations

from typing import List, Optional

# Chat lines escalate in explicitness so a reluctant policy is repeatedly invited
# to simply name its share. They never bargain and never assert a preference.
CONCEDE_OPENERS = [
    "Honestly I don't have strong feelings about any of these items, so go ahead and "
    "take the lead. Tell me what you'd like to keep and I'll happily take whatever is left.",
    "I'm easy to deal with — whatever split you want is fine by me. Just name your share "
    "and I'll take the remainder.",
]
CONCEDE_FOLLOWUPS = [
    "That's totally fine by me. Go ahead and lock in your share and I'll take the rest.",
    "Works for me — whatever you keep, I'm happy with the leftovers. Post your tag whenever.",
    "No objections here. Just state exactly what you want and I'll take everything else.",
]
CONCEDE_CLOSER = "Sounds good to me — that split works. I'll take the rest."


def _fmt_deal(item_names: List[str], keep: List[int]) -> str:
    body = ", ".join(f'"{n}": {k}' for n, k in zip(item_names, keep))
    return "{" + body + "}"


class ConcederOpponent:
    """A pure-Python pushover that always yields to the measured policy.

    The opponent holds seat B (``them_values``) but ignores them entirely: its
    only behaviour is to concede. It is protocol-aware so it can slot into either
    the dual-tag or single-proposer harness.
    """

    label = "Conceder(scripted)"
    slug = "scripted/conceder"

    def __init__(self, item_names, counts):
        self.items = list(item_names)
        self.counts = list(counts)
        self._turn = 0

    def complement(self, policy_take: List[int]) -> List[int]:
        """Leftovers after the policy's claim (clamped at 0 so an overclaim by the
        policy yields a conflicting — i.e. correctly failed — partition)."""
        return [max(0, c - t) for c, t in zip(self.counts, policy_take)]

    def _concede_line(self) -> str:
        pool = CONCEDE_OPENERS if self._turn == 0 else CONCEDE_FOLLOWUPS
        line = pool[self._turn % len(pool)]
        self._turn += 1
        return line

    def dual_message(self, policy_last_deal: Optional[List[int]]) -> str:
        """Dual-tag reply. Once the policy has emitted a ``<deal>``, close by
        claiming the complementary leftovers; otherwise invite it to claim."""
        if policy_last_deal is not None:
            keep = self.complement(policy_last_deal)
            return f"{CONCEDE_CLOSER} <deal>{_fmt_deal(self.items, keep)}</deal>"
        return self._concede_line()

    def single_message(self, policy_has_proposal: bool) -> str:
        """Single-proposer reply: accept any standing offer the instant one
        exists, otherwise prompt the policy to make one."""
        if policy_has_proposal:
            return f"{CONCEDE_CLOSER} <accept>"
        return self._concede_line()
