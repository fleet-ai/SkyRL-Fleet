"""Load negotiation *scenarios* (counts + both agents' private values) from the
parsed visualizer JSON. A scenario is the game setup only -- we drop the human
dialogue and final allocation, keeping just what's needed to play a fresh game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

VIZ_DATA = Path(__file__).resolve().parent / "visualizer" / "public" / "data"


@dataclass(frozen=True)
class Scenario:
    item_names: tuple
    counts: tuple
    you_values: tuple
    them_values: tuple

    @property
    def key(self):
        return (self.counts, self.you_values, self.them_values)


def load_scenarios(dataset: str = "dnd", split: str = "val", dedupe: bool = True) -> List[Scenario]:
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
