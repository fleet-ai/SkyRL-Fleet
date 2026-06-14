"""Annotate sense-log records using falmart's schema-registry.json.

The registry is the env's enforced action vocabulary (112 oRPC procedures + 34
route templates), each annotated with effect / entity / fn / flags. We load it so
the harness labels every observed record with the *same* semantics the env uses
to render its observation text — no drift between the label we train on and the
gloss the agent reads.

For records whose value is not in the registry (a "vocabulary miss" — a new
router, a 404 pathname), we fall back to the verb rule (effect from the leading
verb of the procedure key) and mark the annotation `unknown=True`, never
silently treating it as known.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Rule 1 (SCHEMA_EXPLAINED.md): effect from the procedure key's leading verb.
# Used only as a fallback for vocabulary-miss procedures not in the registry.
_READ_VERBS = {
    "autocomplete", "availability", "batch", "check", "current", "deals",
    "departments", "get", "history", "inventory", "list", "navigation",
    "orders", "payment", "qa", "recently", "redemptions", "related",
    "reorder", "resolve", "reviews", "search", "session", "slots", "stats",
    "status", "tracking", "validate", "variants",
}
_WRITE_VERBS = {
    "add", "cancel", "change", "clear", "create", "delete", "login", "logout",
    "move", "redeem", "register", "remove", "resend", "save", "set", "update",
    "verify",
}
_TELEMETRY_VERBS = {"track"}


def _verb_of(procedure_key: str) -> str:
    """Leading lowercase run of the router key after the dot.

    `addresses.setDefault` -> `set`; `cart.checkoutPreview` -> `checkout`.
    """
    key = procedure_key.split(".", 1)[-1]
    m = re.match(r"^[a-z]+", key)
    return m.group(0) if m else ""


def effect_from_verb(procedure_key: str) -> str:
    """Best-effort effect for an unannotated procedure (verb rule, Rule 1)."""
    verb = _verb_of(procedure_key)
    if verb in _WRITE_VERBS:
        return "write"
    if verb in _TELEMETRY_VERBS:
        return "telemetry-write"
    # `read` is the safe default: the ambiguous `checkout` is read-ish
    # (cart.checkoutPreview) unless the registry says otherwise.
    return "read"


@dataclass
class RpcAnnotation:
    effect: str  # read | write | telemetry-write
    entity: str
    fn: str
    flags: List[str] = field(default_factory=list)
    unknown: bool = False


@dataclass
class RouteAnnotation:
    page_type: str
    fn: str
    template: Optional[str] = None
    unknown: bool = False


def _template_to_regex(template: str) -> re.Pattern:
    """Convert a SvelteKit route template to a matcher.

    `[x]` -> one path segment; `[...x]` -> one or more segments.
    `/ip/[slug]/[id]` matches `/ip/red-shoe/491`.
    """
    parts = []
    for seg in template.strip("/").split("/"):
        if seg.startswith("[...") and seg.endswith("]"):
            parts.append(r".+")
        elif seg.startswith("[") and seg.endswith("]"):
            parts.append(r"[^/]+")
        elif seg == "":
            continue
        else:
            parts.append(re.escape(seg))
    body = "/".join(parts)
    return re.compile(rf"^/{body}/?$")


class Registry:
    """In-memory view of schema-registry.json with annotation lookups."""

    def __init__(self, data: Dict):
        kinds = data.get("kinds", {})
        self._procedures: Dict[str, Dict] = kinds.get("rpc", {}).get(
            "procedures", {}
        )
        self._routes: Dict[str, Dict] = kinds.get("route", {}).get(
            "routeTemplates", {}
        )
        # Pre-compile route matchers, longest (most specific) template first so
        # `/ip/[slug]/[id]` wins over `/ip/[id]` on a 3-segment path.
        self._route_matchers = sorted(
            ((tpl, _template_to_regex(tpl), meta)
             for tpl, meta in self._routes.items()),
            key=lambda t: len(t[0].strip("/").split("/")),
            reverse=True,
        )

    @classmethod
    def load(cls, path: str) -> "Registry":
        with open(os.path.expanduser(path), "r") as f:
            return cls(json.load(f))

    def annotate_rpc(self, procedure: str) -> RpcAnnotation:
        meta = self._procedures.get(procedure)
        if meta is None:
            return RpcAnnotation(
                effect=effect_from_verb(procedure),
                entity="?",
                fn="(unannotated procedure)",
                flags=[],
                unknown=True,
            )
        return RpcAnnotation(
            effect=meta.get("effect", effect_from_verb(procedure)),
            entity=meta.get("entity", "?"),
            fn=meta.get("fn", ""),
            flags=list(meta.get("flags", [])),
        )

    def annotate_route(self, to: str) -> RouteAnnotation:
        # Strip query/hash before matching templates.
        path = (to or "").split("?", 1)[0].split("#", 1)[0]
        # Exact-template fast path.
        if path in self._routes:
            meta = self._routes[path]
            return RouteAnnotation(
                page_type=meta.get("pageType", "?"),
                fn=meta.get("fn", ""),
                template=path,
            )
        for tpl, rx, meta in self._route_matchers:
            if rx.match(path):
                return RouteAnnotation(
                    page_type=meta.get("pageType", "?"),
                    fn=meta.get("fn", ""),
                    template=tpl,
                )
        return RouteAnnotation(
            page_type="?", fn="(unannotated route)", template=None, unknown=True
        )
