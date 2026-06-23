"""Load negotiation *scenarios* (counts + both agents' private values) from the
parsed visualizer JSON. A scenario is the game setup only -- we drop the human
dialogue and final allocation, keeping just what's needed to play a fresh game.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

VIZ_DATA = Path(__file__).resolve().parent / "visualizer" / "public" / "data"

# Vocabulary for synthetic scenarios — distinct single-word items the proposal
# parser (game._extract_counts, matched by name) and prompt builder handle cleanly.
_SYN_ITEMS = ("book", "hat", "ball", "pen", "mug", "lamp", "clock", "plant")


@dataclass(frozen=True)
class Scenario:
    item_names: tuple
    counts: tuple
    you_values: tuple
    them_values: tuple

    @property
    def key(self):
        return (self.counts, self.you_values, self.them_values)


def generate_synthetic_scenarios(n: int = 128, seed: int = 0) -> List[Scenario]:
    """Procedurally generate held-out DnD-style scenarios with structure the
    training set (3 items, values normalized to sum 10) never shows: 4-6 item
    types, larger/asymmetric counts and totals, deliberate zero-value and conflict
    items, across a controlled integrative<->distributive spectrum. Fully
    verifiable by game.evaluate(); deterministic given `seed` so the eval set is
    fixed across runs.
    """
    rng = random.Random(seed)
    out: List[Scenario] = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        k = rng.randint(4, 6)                                   # 4-6 items (train has 3)
        names = tuple(rng.sample(_SYN_ITEMS, k))
        counts = tuple(rng.randint(1, 4) for _ in range(k))     # larger counts
        integ = rng.random()  # 0 = distributive (both want same), 1 = integrative
        you, them = [], []
        for _ in range(k):
            hi = rng.randint(4, 9)
            if rng.random() < integ:
                # integrative: one side values it, the other ~0 (zero-value item)
                if rng.random() < 0.5:
                    you.append(hi)
                    them.append(rng.randint(0, 2))
                else:
                    you.append(rng.randint(0, 2))
                    them.append(hi)
            else:
                # distributive conflict: both value it similarly
                you.append(hi)
                them.append(max(0, hi + rng.randint(-1, 1)))
        you, them = tuple(you), tuple(them)
        # Non-degenerate game; totals deliberately NOT normalized (asymmetric).
        if sum(c * v for c, v in zip(counts, you)) == 0:
            continue
        if sum(c * v for c, v in zip(counts, them)) == 0:
            continue
        key = (counts, you, them)
        if key in seen:
            continue
        seen.add(key)
        out.append(Scenario(item_names=names, counts=counts, you_values=you, them_values=them))
    return out


def load_scenarios(dataset: str = "dnd", split: str = "val", dedupe: bool = True) -> List[Scenario]:
    if dataset == "synthetic":
        # Held-out, procedurally generated eval set (no JSON file). Fixed seed so
        # the set is reproducible across runs; subsampled by --max_extra_val.
        return generate_synthetic_scenarios(n=128, seed=0)
    path = VIZ_DATA / dataset / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No parsed data at {path}. Run visualizer/build.py first."
        )
    data = json.loads(path.read_text())
    out: List[Scenario] = []
    seen = set()
    for g in data["games"]:
        sc = Scenario(
            item_names=tuple(g["item_names"]),
            counts=tuple(g["counts"]),
            you_values=tuple(g["you_values"]),
            them_values=tuple(g["them_values"]),
        )
        if dedupe:
            if sc.key in seen:
                continue
            seen.add(sc.key)
        out.append(sc)
    return out
