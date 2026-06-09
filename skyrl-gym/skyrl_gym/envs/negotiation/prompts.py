"""Prompt construction for the negotiation task (shared by eval + RLVR env)."""

from __future__ import annotations

from typing import List

SYSTEM_TEMPLATE = """\
You are playing a multi-issue negotiation game against another player.

There is a shared pool of items to divide between the two of you:
{pool_lines}

These are YOUR private point values (the other player has different, hidden values):
{value_lines}
If you took the entire pool you would score {you_max} points; the other player cannot see this.

How it works:
- Take turns exchanging short messages (1-3 sentences) to agree on who gets what.
- Every unit must go to exactly one of you — together you divide the ENTIRE pool.

Finalizing — READ CAREFULLY, this is where most deals fail:
- When you are ready to lock in, end your message with a single line of the exact form:
  <deal>{deal_example}</deal>
- Your <deal> lists ONLY the items YOU PERSONALLY keep, from your own point of view.
  It is NOT the full split, and it is NOT what the other player keeps.
- The other player submits their OWN separate <deal> listing what THEY keep. Your two
  <deal> tags describe DIFFERENT halves of the pool; added together they must equal the
  pool EXACTLY — no item claimed twice, none left out.
- So do NOT copy the other player's numbers, and do NOT put the whole agreed split in your
  <deal>. Each turn, ask yourself "how many of each item do *I* end up with?" and list only that.
- A deal is finalized only once BOTH players have submitted a <deal>.
- If the two <deal> tags don't sum to exactly the pool — you both claim the same unit
  (overlap), or you both leave something unclaimed — the deal FAILS and you BOTH score 0.
- Confirming an agreement does NOT mean repeating the other player's tag. If the other
  player already posted a <deal>, yours should list the LEFTOVER items (the ones they did
  NOT take) — so your tag will normally have different numbers than theirs.
- Sanity check before you send: add your numbers to the other player's last <deal>. If the
  total exactly equals the pool, you're good. If your tag is identical to theirs, you've
  wrongly claimed their items — list your own leftover share instead.

{worked_example}
- Your goal is to maximize YOUR OWN points. A failed deal (0 points) is worse than a modest
  agreement, so settle the FULL split in words first, then each list only your own keep.

You have at most {max_turns} messages. Be efficient and decisive."""


SYSTEM_TEMPLATE_SINGLE = """\
You are playing a multi-issue negotiation game against another player.

There is a shared pool of items to divide between the two of you:
{pool_lines}

These are YOUR private point values (the other player has different, hidden values):
{value_lines}
If you took the entire pool you would score {you_max} points; the other player cannot see this.

How it works:
- Take turns exchanging short messages (1-3 sentences) to negotiate who gets what.
- To make a concrete OFFER, end your message with a single line of the exact form:
  <propose>{deal_example}</propose>
  listing how many of EACH item YOU would keep. The other player automatically gets ALL the rest.
- To ACCEPT the other player's most recent offer, reply with a line containing exactly:
  <accept>
  The deal is then finalized exactly as they proposed: they keep what they listed and YOU get everything else.
- The negotiation ends the instant an offer is accepted. Only ONE player needs to propose a split;
  the other simply accepts it. Because a single offer divides every item, the claims can never
  "conflict" and items can never be left over.
- If no offer is accepted within the message limit, the deal FAILS and you BOTH score 0.
- Your goal is to maximize YOUR OWN points, but a failed deal (0 points) is worse than a modest
  agreement, so always close: accept a reasonable offer, or make one the other player will accept.

You have at most {max_turns} messages. Be efficient and decisive."""


def _build_worked_example(item_names: List[str]) -> str:
    """A concrete two-perspective example showing that each side's <deal> lists
    only ITS OWN keep, the two tags differ, and together they sum to the pool.

    Uses the real item names so the model anchors on this scenario's vocabulary.
    Hypothetical pool: 2 x item0 (+ 1 x item1 if a second item exists).
    """
    i0 = item_names[0]
    has_second = len(item_names) > 1

    def deal(d: dict) -> str:
        return "{" + ", ".join(f'"{n}": {d.get(n, 0)}' for n in item_names) + "}"

    if has_second:
        i1 = item_names[1]
        pool = f"2 {i0}s and 1 {i1}"
        you_share, their_share = f"1 {i0} + the {i1}", f"1 {i0}"
        you_deal = deal({i0: 1, i1: 1})
        their_deal = deal({i0: 1})
        total = f"2 {i0}s + 1 {i1}"
    else:
        pool = f"2 {i0}s"
        you_share, their_share = f"1 {i0}", f"1 {i0}"
        you_deal = deal({i0: 1})
        their_deal = deal({i0: 1})
        total = f"2 {i0}s"

    return (
        f"Worked example — suppose the pool is {pool}, and you agree that YOU keep "
        f"{you_share} while the OTHER player keeps {their_share}:\n"
        f"  - YOUR  <deal>{you_deal}</deal>   (only your share)\n"
        f"  - THEIR <deal>{their_deal}</deal>   (only their share)\n"
        f"  Added together that is {total} = the whole pool. ✓ Note the two tags are NOT identical — "
        f"each lists a different half."
    )


def build_system_prompt(
    item_names: List[str], counts: List[int], values: List[int], max_turns: int, protocol: str = "single"
) -> str:
    pool_lines = "\n".join(f"  - {c} x {name}" for name, c in zip(item_names, counts))
    value_lines = "\n".join(f"  - {name}: {v} points each" for name, v in zip(item_names, values))
    you_max = sum(c * v for c, v in zip(counts, values))
    deal_example = "{" + ", ".join(f'"{name}": 0' for name in item_names) + "}"
    template = SYSTEM_TEMPLATE_SINGLE if protocol == "single" else SYSTEM_TEMPLATE
    return template.format(
        pool_lines=pool_lines,
        value_lines=value_lines,
        you_max=you_max,
        deal_example=deal_example,
        max_turns=max_turns,
        worked_example=_build_worked_example(item_names),
    )


OPENING_USER_MSG = (
    "You speak first. Open the negotiation with a brief message proposing how to split the items."
)

PARTNER_FIRST_MSG = (
    "The other player will speak first. Wait for their message, then respond."
)
