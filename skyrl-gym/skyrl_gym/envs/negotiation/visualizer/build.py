#!/usr/bin/env python3
"""Build all dataset JSON for the visualizer and write the dataset manifest."""

import json
from pathlib import Path

import parse_casino
import parse_data

PUBLIC_DATA = Path(__file__).resolve().parent / "public" / "data"

MANIFEST = {
    "datasets": [
        {
            "id": "dnd",
            "name": "Deal or No Deal",
            "blurb": "FAIR end-to-end negotiator. Split books / hats / balls; private values sum to 10.",
            "splits": ["train", "val", "test"],
            "items": parse_data.ITEM_NAMES,
        },
        {
            "id": "casino",
            "name": "CaSiNo (Campsite)",
            "blurb": "Cornell campsite negotiations. Split Food / Water / Firewood by High/Med/Low priority.",
            "splits": ["all"],
            "items": parse_casino.ITEM_NAMES,
        },
    ]
}


def main():
    parse_data.main()
    parse_casino.main()
    with open(PUBLIC_DATA / "manifest.json", "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"wrote manifest to {PUBLIC_DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()
