"""Value-awareness analysis for negotiation traces.

Question: do agents actually understand and *use* their own private value
function, or do they "vibe-negotiate" (agree on a split that ignores which
items are worth more to them)?

Core metric — VALUE CAPTURE (conditional on quantity)
-----------------------------------------------------
For one side of an agreed deal, let the side end up with T total units.
Given T, the BEST that side could do is grab the T highest-value units in the
pool; the WORST is the T lowest-value units. We score:

    value_capture = (achieved - worst_for_T) / (best_for_T - worst_for_T)

This *controls for how much stuff they won* and asks only: of the units they
took, were they the high-value ones? A value-blind ("vibe") negotiator who
ends up with a random T units scores ~0.5; a value-aware one scores ~1.0.
Undefined (skipped) when best==worst (T=0, T=all, or all unit-values equal).

Supplementary metrics:
- zero_value_concession (dnd): of a side's value-0 item units, fraction it did
  NOT take (correctly gave away). Value-aware -> ~1.0.
- top_item_secured: side captured ALL units of its single strictly-highest-value
  item type.
- first_offer_capture: value_capture of the FIRST concrete offer the "you" side
  put on the table (does the opening bid already chase high-value items?).
- value_talk_rate: fraction of negotiations where a side's prose explicitly
  reasons about item priority/value.
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path
from typing import List, Optional

RESULTS = Path(__file__).parent / "results"

# (label, dataset, protocol, relative path)
FILES = [
    ("gpt-4o-mini", "dnd", "single", "openai_gpt-4o-mini_dnd_val_single_n20.json"),
    ("gpt-4o-mini", "dnd", "dual", "openai_gpt-4o-mini_dnd_val_dual_n12.json"),
    ("gpt-4o-mini", "casino", "single", "openai_gpt-4o-mini_casino_all_single_n20.json"),
    ("gpt-4o-mini", "casino", "dual", "openai_gpt-4o-mini_casino_all_n30.json"),
    ("qwen3.5-35b-a3b", "dnd", "single", "qwen_qwen3.5-35b-a3b_dnd_val_single-nothink_n20.json"),
    ("qwen3.5-35b-a3b", "dnd", "dual", "qwen_qwen3.5-35b-a3b_dnd_val_dual-nothink_n12.json"),
    ("qwen3.5-35b-a3b", "casino", "single", "qwen_qwen3.5-35b-a3b_casino_all_single-nothink_n20.json"),
]

VALUE_TALK_RE = re.compile(
    r"\b(most valuable|least valuable|more valuable|less valuable|high(?:er|est)?[ -]?value|"
    r"low(?:er|est)?[ -]?value|prioriti[sz]e|priority|priorities|worth more|worth less|"
    r"matters? (?:most|more|to me)|important to me|most important|less important|"
    r"i (?:value|care|need|want) |don'?t (?:need|care|value)|care most|"
    r"top priority|my main|i'?m after|really want|biggest value)\b",
    re.IGNORECASE,
)


def unit_values(counts: List[int], values: List[int]) -> List[int]:
    """Expand the pool into a flat list of per-unit point values."""
    out: List[int] = []
    for c, v in zip(counts, values):
        out.extend([v] * c)
    return out


def value_capture(take: List[int], counts: List[int], values: List[int]) -> Optional[float]:
    """(achieved - worst_for_T) / (best_for_T - worst_for_T), conditional on T units."""
    T = sum(take)
    achieved = sum(t * v for t, v in zip(take, values))
    units = sorted(unit_values(counts, values), reverse=True)
    if T <= 0 or T >= len(units):
        return None
    best_for_T = sum(units[:T])
    worst_for_T = sum(units[-T:])
    if best_for_T == worst_for_T:
        return None
    return (achieved - worst_for_T) / (best_for_T - worst_for_T)


def zero_value_concession(take: List[int], counts: List[int], values: List[int]) -> Optional[float]:
    """Of the side's value-0 item units, fraction NOT taken (correctly conceded)."""
    zero_units = sum(c for c, v in zip(counts, values) if v == 0)
    if zero_units == 0:
        return None
    taken_zero = sum(t for t, c, v in zip(take, counts, values) if v == 0)
    return (zero_units - taken_zero) / zero_units


def top_item_secured(take: List[int], counts: List[int], values: List[int]) -> Optional[bool]:
    """Did the side capture ALL units of its single strictly-highest-value item type?"""
    mx = max(values)
    top_idx = [i for i, v in enumerate(values) if v == mx]
    if len(top_idx) != 1:
        return None  # tie for top value -> ambiguous
    i = top_idx[0]
    return take[i] == counts[i]


def first_you_offer(transcript, item_names) -> Optional[List[int]]:
    """Parse the first concrete keep-offer made by the 'you' side.

    single protocol: <propose>{...}</propose> (lists your keep).
    dual protocol:   <deal>{...}</deal>      (lists your keep).
    """
    deal_re = re.compile(r"<(?:propose|deal)>\s*(\{.*?\})\s*</(?:propose|deal)>", re.DOTALL | re.IGNORECASE)
    kv_re = re.compile(r"['\"]?([A-Za-z_]+)['\"]?\s*[:=]\s*(\d+)")
    for msg in transcript:
        if msg.get("speaker") != "you":
            continue
        m = deal_re.search(msg.get("text", ""))
        if not m:
            continue
        blob = m.group(1)
        parsed = {}
        try:
            obj = json.loads(blob)
            for k, v in obj.items():
                parsed[str(k).strip().lower()] = int(v)
        except Exception:
            for k, v in kv_re.findall(blob):
                parsed[k.strip().lower()] = int(v)
        if parsed:
            return [max(0, parsed.get(n.lower(), 0)) for n in item_names]
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def analyze(path: Path):
    d = json.loads(path.read_text())
    results = d["results"]

    vc_you, vc_them, vc_all = [], [], []
    zc_all = []
    top_all = []
    first_off = []
    talk_you, talk_them = [], []
    n_deals = 0

    for r in results:
        sc = r["scenario"]
        counts = sc["counts"]
        yv, tv = sc["you_values"], sc["them_values"]
        item_names = sc["item_names"]
        out = r["outcome"]

        # textual value-talk over the whole negotiation (regardless of deal)
        you_text = " ".join(m["text"] for m in r["transcript"] if m.get("speaker") == "you")
        them_text = " ".join(m["text"] for m in r["transcript"] if m.get("speaker") == "them")
        talk_you.append(1.0 if VALUE_TALK_RE.search(you_text) else 0.0)
        talk_them.append(1.0 if VALUE_TALK_RE.search(them_text) else 0.0)

        if not out.get("agreed"):
            continue
        n_deals += 1
        yt, tt = out["you_take"], out["them_take"]

        cy = value_capture(yt, counts, yv)
        ct = value_capture(tt, counts, tv)
        vc_you.append(cy)
        vc_them.append(ct)
        vc_all.extend([x for x in (cy, ct) if x is not None])

        zc_all.extend(
            [x for x in (zero_value_concession(yt, counts, yv), zero_value_concession(tt, counts, tv)) if x is not None]
        )
        top_all.extend(
            [
                1.0 if x else 0.0
                for x in (top_item_secured(yt, counts, yv), top_item_secured(tt, counts, tv))
                if x is not None
            ]
        )

        fo = first_you_offer(r["transcript"], item_names)
        if fo is not None:
            fc = value_capture(fo, counts, yv)
            if fc is not None:
                first_off.append(fc)

    return {
        "n_total": len(results),
        "n_deals": n_deals,
        "value_capture_you": mean(vc_you),
        "value_capture_them": mean(vc_them),
        "value_capture_both": mean(vc_all),
        "value_capture_n": len([x for x in vc_all if x is not None]),
        "zero_concession": mean(zc_all),
        "zero_concession_n": len(zc_all),
        "top_item_secured": mean(top_all),
        "top_item_n": len(top_all),
        "first_offer_capture": mean(first_off),
        "first_offer_n": len(first_off),
        "value_talk_you": mean(talk_you),
        "value_talk_them": mean(talk_them),
    }


def fmt(x, pct=True):
    if x is None:
        return "  -  "
    return f"{x*100:4.0f}%" if pct else f"{x:.2f}"


def main():
    rows = []
    for label, dataset, protocol, fname in FILES:
        path = RESULTS / fname
        if not path.exists():
            print(f"!! missing {fname}")
            continue
        m = analyze(path)
        m.update(model=label, dataset=dataset, protocol=protocol, file=fname)
        rows.append(m)

    hdr = (
        f"{'model':16} {'data':7} {'proto':7} {'deals':>5} "
        f"{'VC both':>8} {'VC prop':>8} {'VC acc':>8} {'0-conc':>7} {'top':>5} "
        f"{'1stoff':>7} {'talk(y/t)':>11}"
    )
    print("\n" + "=" * len(hdr))
    print("VALUE-AWARENESS  (VC = value-capture conditional on quantity; higher = uses value fn)")
    print("  VC prop = proposer/you side, VC acc = accepter/them side")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        talk = f"{fmt(m['value_talk_you'])}/{fmt(m['value_talk_them'])}"
        print(
            f"{m['model']:16} {m['dataset']:7} {m['protocol']:7} {m['n_deals']:>5} "
            f"{fmt(m['value_capture_both']):>8} {fmt(m['value_capture_you']):>8} "
            f"{fmt(m['value_capture_them']):>8} {fmt(m['zero_concession']):>7} "
            f"{fmt(m['top_item_secured']):>5} {fmt(m['first_offer_capture']):>7} {talk:>11}"
        )

    paired_dnd()

    # dump machine-readable
    out = RESULTS.parent / "value_alignment_metrics.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


def _per_scenario_capture(path: Path):
    """Map scenario-index -> mean(both-sides value_capture) for agreed deals."""
    d = json.loads(path.read_text())
    cap = {}
    for i, r in enumerate(d["results"]):
        out = r["outcome"]
        if not out.get("agreed"):
            cap[i] = None
            continue
        sc = r["scenario"]
        c, yv, tv = sc["counts"], sc["you_values"], sc["them_values"]
        cy = value_capture(out["you_take"], c, yv)
        ct = value_capture(out["them_take"], c, tv)
        vals = [x for x in (cy, ct) if x is not None]
        cap[i] = st.mean(vals) if vals else None
    return cap


def paired_dnd():
    """Same-scenario, both-protocols-agreed paired comparison on dnd."""
    pairs = [
        ("gpt-4o-mini", "openai_gpt-4o-mini_dnd_val_single_n20.json", "openai_gpt-4o-mini_dnd_val_dual_n12.json"),
        (
            "qwen3.5-35b-a3b",
            "qwen_qwen3.5-35b-a3b_dnd_val_single-nothink_n20.json",
            "qwen_qwen3.5-35b-a3b_dnd_val_dual-nothink_n12.json",
        ),
    ]
    print("\n" + "=" * 70)
    print("PAIRED dnd: same scenarios, agreed under BOTH protocols")
    print("  value_capture (both sides), single vs dual, mean over shared deals")
    print("=" * 70)
    print(f"{'model':16} {'#shared':>7} {'single':>8} {'dual':>8} {'Δ(dual-single)':>15}")
    print("-" * 70)
    for label, sfile, dfile in pairs:
        cs = _per_scenario_capture(RESULTS / sfile)
        cd = _per_scenario_capture(RESULTS / dfile)
        shared = [i for i in cs if i in cd and cs[i] is not None and cd[i] is not None]
        if not shared:
            print(f"{label:16}  no shared agreed scenarios")
            continue
        s_mean = st.mean(cs[i] for i in shared)
        d_mean = st.mean(cd[i] for i in shared)
        print(f"{label:16} {len(shared):>7} {s_mean*100:7.0f}% {d_mean*100:7.0f}% " f"{(d_mean-s_mean)*100:>+14.0f}%")


if __name__ == "__main__":
    main()
