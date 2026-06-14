"""Sense-log client + outcome attribution.

The env exposes a non-destructive cursor read at `GET /api/sense/log?since=<n>`
returning `{records, next, dropped, text}`:

  - `records`  raw SenseRecord[] (the ground truth we label).
  - `next`     cursor to poll forward from; re-reading is idempotent.
  - `dropped`  records lost because the cursor fell behind the ring buffer
               (>0 means our attribution for this window is unreliable).
  - `text`     the env-rendered agent observation (the gloss) — forwarded
               verbatim into the observation in the *sensory-on* arm.

Attribution is a cursor diff: snapshot `next` BEFORE an action, read
`since=<that next>` AFTER; the returned records are that action's consequence.
An empty record set is a legitimate, learnable outcome ("empty-delta").
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .registry import Registry


class Outcome(str, Enum):
    """The four canonical click outcomes (schema-registry `outcomeClasses`)."""

    NEW_WRITE = "new-write"            # I changed domain state.
    NEW_READ_OR_ROUTE = "new-read-or-route"  # I navigated / fetched data.
    OK_FALSE = "ok-false"             # I tried X and it failed (incl. 501 stubs).
    EMPTY_DELTA = "empty-delta"       # Nothing observable (pure UI / dead button).
    # A fine-grained 5th case: the click only wrote analytics/telemetry
    # (navigation.track, customers.trackView). No user-visible domain change.
    # `coarse_4class()` folds this into EMPTY_DELTA; kept distinct so callers
    # can drop or down-weight navigation.* noise explicitly.
    TELEMETRY_WRITE = "telemetry-write"


@dataclass
class SenseDelta:
    """The classified consequence of a single action (one cursor diff)."""

    outcome: Outcome
    records: List[Dict[str, Any]]
    text: Optional[str]            # env-rendered observation gloss (may be None)
    procedures: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    routes: List[Dict[str, str]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    statuses: List[int] = field(default_factory=list)
    has_unknown: bool = False       # vocabulary miss (record not in registry)
    dropped: int = 0
    reliable: bool = True           # False when dropped>0 (attribution gap)

    def coarse_4class(self) -> Outcome:
        """Map to the canonical 4-class target (telemetry folds to empty)."""
        if self.outcome is Outcome.TELEMETRY_WRITE:
            return Outcome.EMPTY_DELTA
        return self.outcome


def classify_delta(
    records: List[Dict[str, Any]],
    registry: Registry,
    text: Optional[str] = None,
    dropped: int = 0,
) -> SenseDelta:
    """Reduce a list of new sense records to one labeled outcome.

    Priority (most-significant wins): a failed call (ok-false) dominates, then a
    write, then a read/route, then telemetry-only, then nothing (empty-delta).
    """
    procedures: List[str] = []
    effects: List[str] = []
    routes: List[Dict[str, str]] = []
    flags: List[str] = []
    statuses: List[int] = []
    has_unknown = False

    saw_ok_false = False
    saw_write = False
    saw_read_or_route = False
    saw_telemetry = False

    for rec in records:
        kind = rec.get("kind")
        if kind == "rpc":
            proc = rec.get("procedure", "")
            ann = registry.annotate_rpc(proc)
            procedures.append(proc)
            effects.append(ann.effect)
            flags.extend(ann.flags)
            if "status" in rec:
                statuses.append(rec["status"])
            has_unknown = has_unknown or ann.unknown
            # `ok` is coarse; a 501 stub or any 4xx/5xx reads ok:false.
            if rec.get("ok") is False:
                saw_ok_false = True
            elif ann.effect == "write":
                saw_write = True
            elif ann.effect == "telemetry-write":
                saw_telemetry = True
            else:  # read
                saw_read_or_route = True
        elif kind == "route":
            ann = registry.annotate_route(rec.get("to", ""))
            routes.append({
                "from": rec.get("from", ""),
                "to": rec.get("to", ""),
                "pageType": ann.page_type,
            })
            has_unknown = has_unknown or ann.unknown
            saw_read_or_route = True

    if not records:
        outcome = Outcome.EMPTY_DELTA
    elif saw_ok_false:
        outcome = Outcome.OK_FALSE
    elif saw_write:
        outcome = Outcome.NEW_WRITE
    elif saw_read_or_route:
        outcome = Outcome.NEW_READ_OR_ROUTE
    elif saw_telemetry:
        outcome = Outcome.TELEMETRY_WRITE
    else:
        outcome = Outcome.EMPTY_DELTA

    return SenseDelta(
        outcome=outcome,
        records=records,
        text=text,
        procedures=procedures,
        effects=sorted(set(effects)),
        routes=routes,
        flags=sorted(set(flags)),
        statuses=statuses,
        has_unknown=has_unknown,
        dropped=dropped,
        reliable=(dropped == 0),
    )


class SenseClient:
    """Thin cursor-loop client over `/api/sense/log`.

    Stdlib-only (urllib) so the data-engine core has no third-party deps and
    runs anywhere. `base_url` is the env origin, e.g. http://localhost:5173
    (sense routes are dual-mounted at /api/sense/* and /sense/*).
    """

    def __init__(self, base_url: str, registry: Registry, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.registry = registry
        self.timeout = timeout
        self.cursor = 0

    def _read(self, since: int) -> Dict[str, Any]:
        url = (
            f"{self.base_url}/api/sense/log?"
            + urllib.parse.urlencode({"since": since})
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def snapshot(self) -> int:
        """Advance the cursor to 'now' without recording a delta.

        Call once at episode start so the first action is attributed cleanly.
        """
        data = self._read(self.cursor)
        self.cursor = data.get("next", self.cursor)
        return self.cursor

    def read_delta(self) -> SenseDelta:
        """Read everything since the last cursor and classify it as one delta."""
        data = self._read(self.cursor)
        records = data.get("records", []) or []
        self.cursor = data.get("next", self.cursor)
        return classify_delta(
            records,
            self.registry,
            text=data.get("text"),
            dropped=data.get("dropped", 0) or 0,
        )
