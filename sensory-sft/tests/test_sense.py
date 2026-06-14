"""Offline self-test for the outcome taxonomy — no running env required.

Validates classify_delta against synthetic sense records using a tiny inline
registry, covering all four canonical outcomes plus the tricky cases the work
trial flags (501 ok-false, telemetry-only -> empty, dropped attribution gap,
vocabulary miss, route-template matching, ok-false dominance).

Run: `python tests/test_sense.py`  (also works under pytest).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensory_sft.registry import Registry  # noqa: E402
from sensory_sft.sense import Outcome, classify_delta  # noqa: E402

# Minimal registry mirroring schema-registry.json's shape.
REG = Registry({
    "kinds": {
        "rpc": {
            "procedures": {
                "cart.addItem": {"effect": "write", "entity": "cart", "fn": "add"},
                "products.get": {"effect": "read", "entity": "product", "fn": "get"},
                "navigation.track": {
                    "effect": "telemetry-write", "entity": "nav-event", "fn": "track"
                },
                "promos.redeem": {
                    "effect": "write", "entity": "promo", "fn": "redeem",
                    "flags": ["not-implemented"],
                },
            }
        },
        "route": {
            "routeTemplates": {
                "/ip/[id]": {"pageType": "product-detail", "fn": "PDP"},
                "/ip/[slug]/[id]": {"pageType": "product-detail", "fn": "PDP"},
                "/cart": {"pageType": "cart", "fn": "Cart"},
            }
        },
    }
})


def _rpc(proc, ok=True, status=200):
    return {"kind": "rpc", "procedure": proc, "ok": ok, "status": status}


def _route(to, frm=""):
    return {"kind": "route", "from": frm, "to": to}


def test_empty_delta():
    d = classify_delta([], REG)
    assert d.outcome is Outcome.EMPTY_DELTA, d.outcome


def test_new_write():
    d = classify_delta([_rpc("cart.addItem")], REG)
    assert d.outcome is Outcome.NEW_WRITE
    assert "write" in d.effects


def test_new_read_route():
    d = classify_delta([_rpc("products.get")], REG)
    assert d.outcome is Outcome.NEW_READ_OR_ROUTE
    d2 = classify_delta([_route("/ip/491", "/")], REG)
    assert d2.outcome is Outcome.NEW_READ_OR_ROUTE
    assert d2.routes[0]["pageType"] == "product-detail"


def test_route_template_specificity():
    # 3-segment path must match /ip/[slug]/[id], not /ip/[id].
    d = classify_delta([_route("/ip/red-shoe/491")], REG)
    assert d.routes[0]["pageType"] == "product-detail"
    # query string is stripped before matching.
    d2 = classify_delta([_route("/cart?foo=1")], REG)
    assert d2.routes[0]["pageType"] == "cart"


def test_ok_false_501_stub():
    d = classify_delta([_rpc("promos.redeem", ok=False, status=501)], REG)
    assert d.outcome is Outcome.OK_FALSE
    assert "not-implemented" in d.flags
    assert 501 in d.statuses


def test_ok_false_dominates_write():
    # A failed call in the same delta as a write -> the failure is the label.
    d = classify_delta(
        [_rpc("cart.addItem"), _rpc("promos.redeem", ok=False, status=501)], REG
    )
    assert d.outcome is Outcome.OK_FALSE


def test_telemetry_only_folds_to_empty():
    d = classify_delta([_rpc("navigation.track")], REG)
    assert d.outcome is Outcome.TELEMETRY_WRITE
    assert d.coarse_4class() is Outcome.EMPTY_DELTA


def test_dropped_marks_unreliable():
    d = classify_delta([_rpc("products.get")], REG, dropped=3)
    assert d.reliable is False
    assert d.dropped == 3


def test_vocabulary_miss():
    # Unknown procedure: kept, flagged, effect inferred from the verb rule.
    d = classify_delta([_rpc("wishlists.addItem")], REG)  # not in mini-registry
    assert d.has_unknown is True
    assert d.outcome is Outcome.NEW_WRITE  # verb "add" -> write


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
