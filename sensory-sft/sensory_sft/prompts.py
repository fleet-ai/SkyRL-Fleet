"""Exploration-friendly task prompts for trace collection.

The deployed walmart tasks are tightly-specified ("buy exactly X"), which makes a
situationally-blind VL model fixate and doom-loop (e.g. the 920px homepage promo
grid whose carousel auto-advances every 4s — every screenshot looks "new" so a
no-op click looks like progress). For *trace collection* we want broad, diverse
coverage of the action vocabulary and lots of the rare, high-value outcomes
(empty-delta, ok-false), not task success. So prompts are deliberately loose:
"explore and attempt", with an explicit instruction to change tactics when a
click does nothing — which is exactly the behavior the sense-log can teach.
"""

from __future__ import annotations

from typing import List

# Loose goals spanning the main domains (products, cart, search, account,
# wishlists, orders) so the collected deltas cover read / write / route / 501.
EXPLORATORY_GOALS: List[str] = [
    "browse a couple of product departments and open a product you find interesting",
    "search for something you might buy and look at the results",
    "add an item to the cart, then open the cart to see what's there",
    "look at the homepage sections and navigate into one of them",
    "try to find product reviews for any product",
    "open your account or order history and look around",
    "add a product to a wishlist or list",
    "try to start a checkout and see how far you get",
    "look for deals or promotions and try to use one",
    "explore the store freely and interact with whatever looks clickable",
]

_PREAMBLE = (
    "You are exploring an online store in a web browser. Your goal is loose: "
    "{goal}. This is exploration, not a strict task — poke around and try things.\n\n"
    "Important:\n"
    "- After each action, check whether the page actually changed in the way you "
    "expected. If a click did nothing, do NOT click the same spot again — try a "
    "different element, the search bar, or the navigation menu.\n"
    "- Banners and carousels can be decorative or rotate on their own; movement on "
    "screen does not mean your click worked.\n"
    "- Keep actions purposeful; you have a limited number of steps.\n"
    "- When you've explored enough or finished, say <done>."
)


def build_task_prompt(goal: str) -> str:
    """Wrap a loose goal in the exploration preamble."""
    return _PREAMBLE.format(goal=goal)


def exploratory_tasks(n: int | None = None) -> List[dict]:
    """Return up to `n` task dicts ({key, goal, prompt}); cycles if n > len."""
    goals = EXPLORATORY_GOALS
    if n is None:
        n = len(goals)
    out = []
    for i in range(n):
        goal = goals[i % len(goals)]
        out.append({
            "key": f"explore-{i:03d}",
            "goal": goal,
            "prompt": build_task_prompt(goal),
        })
    return out
