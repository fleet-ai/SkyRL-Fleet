#!/usr/bin/env python3
"""Render the curated 'Pareto case studies' run to a self-contained PDF.

Reads visualizer/public/data/eval/pareto_cases__dnd__curated.json (built by
build_pareto_cases.py), emits a print-styled HTML, and prints it to PDF via
headless Chrome. Output: results/pareto_case_studies.{html,pdf}

Usage:  python3 export_pareto_pdf.py
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CURATED = HERE.parent / "visualizer" / "public" / "data" / "eval" / "pareto_cases__dnd__curated.json"
HTML_OUT = RESULTS / "pareto_case_studies.html"
PDF_OUT = RESULTS / "pareto_case_studies.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def esc(s):
    return html.escape(str(s))


def alloc_str(items, alloc):
    if not alloc:
        return "—"
    return ", ".join(f"{a}×{n}" for n, a in zip(items, alloc) if True)


def values_str(items, vals):
    return ", ".join(f"{n} {v}" for n, v in zip(items, vals))


def transcript_html(turns):
    rows = []
    for t in turns:
        who = "You" if t.get("speaker") == "you" else "Them"
        body = esc(t.get("text", ""))
        body = body.replace("&lt;deal&gt;", '<b class="tag">&lt;deal&gt;</b>').replace(
            "&lt;/deal&gt;", '<b class="tag">&lt;/deal&gt;</b>'
        )
        rows.append(f'<div class="msg {t.get("speaker","you")}"><span class="sp">{who}:</span> {body}</div>')
    return '<div class="transcript">' + "".join(rows) + "</div>"


def card_html(g):
    items = g["item_names"]
    pareto = g.get("pareto_optimal", False)
    joint = g.get("joint_score", (g.get("you_score") or 0) + (g.get("them_score") or 0))
    mj = g.get("max_joint") or joint
    eff = round(joint / mj * 100) if mj else 100
    if pareto:
        badge = f'<div class="pareto ok">✓ Pareto-optimal · joint {joint}/{mj} · {eff}% efficient</div>'
    else:
        gap = mj - joint
        eff_line = ""
        if g.get("efficient_you") and g.get("efficient_them"):
            eff_line = (
                f'<div class="effsplit">efficient split → You [{alloc_str(items, g["efficient_you"])}]'
                f' · Them [{alloc_str(items, g["efficient_them"])}] (joint {mj})</div>'
            )
        badge = (
            f'<div class="pareto bad">✗ off the Pareto frontier · joint {joint}/{mj} · {eff}% efficient'
            f" · left {gap} joint pts on the table</div>{eff_line}"
        )
    alloc = (
        (
            f'<div class="alloc">You take <b>{alloc_str(items, g.get("you_alloc"))}</b> '
            f'({g.get("you_score",0)} pts) · Them take <b>{alloc_str(items, g.get("them_alloc"))}</b> '
            f'({g.get("them_score",0)} pts)</div>'
        )
        if g.get("agreed")
        else '<div class="alloc nodeal">No deal — both score 0</div>'
    )
    return (
        f'<div class="card">'
        f'<div class="mlabel">{esc(g.get("model_label",""))}</div>'
        f"{alloc}{badge}"
        f'<div class="tmeta">{g.get("num_turns","?")} turns</div>'
        f'{transcript_html(g.get("turns", []))}'
        f"</div>"
    )


def build_html(payload):
    games = payload["games"]
    s = payload["stats"]
    # group by case_no (falls back to fixed chunks for older runs)
    if all("case_no" in g for g in games):
        order, groups = [], {}
        for g in games:
            c = g["case_no"]
            if c not in groups:
                groups[c] = []
                order.append(c)
            groups[c].append(g)
        cases = [groups[c] for c in order]
    else:
        cases = [games[i : i + 2] for i in range(0, len(games), 2)]

    sections = []
    for ci, pair in enumerate(cases, 1):
        g0 = pair[0]
        items = g0["item_names"]
        counts = g0["counts"]
        pool = ", ".join(f"{c}×{n}" for n, c in zip(items, counts))
        header = (
            f'<div class="scn">'
            f'<div class="pool">Pool: <b>{pool}</b></div>'
            f'<div class="vals">Your values: {values_str(items, g0["you_values"])} '
            f'&nbsp;|&nbsp; Their values: {values_str(items, g0["them_values"])}</div>'
            f'<div class="vals muted">(both agents see only their own values; the split below is '
            f"evaluated against the true joint frontier)</div>"
            f"</div>"
        )
        cards = "".join(card_html(g) for g in pair)
        sections.append(
            f'<section class="case"><h2>Case {ci}</h2>{header}' f'<div class="cards">{cards}</div></section>'
        )

    intro = (
        '<p class="lead">Each case is the <b>same negotiation scenario</b> shown as two <b>matched '
        "same-protocol pairs</b>: a <b>frontier model vs Qwen3.5-9B on dual-tag</b>, then the <b>same two "
        "on single-proposer</b>. Under either protocol the frontier trades items to whoever values them "
        "most and lands on the <b>Pareto frontier</b>, while Qwen3.5-9B either <b>fails to close</b> "
        "(no-deal) or <b>agrees off-frontier</b>, leaving joint value on the table. Crucially, on the "
        "deals it does close its <b>own-score can look fine</b> — so neither agreement-rate nor a "
        "self-outcome reward flags the failure. Only a <b>Pareto / joint-efficiency reward</b> sees it. "
        "That is the case for adding it as an RLVR target.</p>"
        f'<p class="lead muted">{len(cases)} cases · dnd/val · matched same-protocol pairs (dual + single) '
        f'· Pareto rate across all cards {round(s["pareto_rate"]*100)}% (the gap is the trainable signal).</p>'
    )

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Pareto case studies</title>
<style>
@page {{ size: A4; margin: 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color: #1a1a1a; font-size: 11px; line-height: 1.4; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }}
.sub {{ color: #666; font-size: 12px; margin-bottom: 10px; }}
.lead {{ font-size: 11.5px; max-width: 100%; margin: 6px 0; }}
.muted {{ color: #777; }}
.case {{ page-break-inside: avoid; break-inside: avoid; border-top: 2px solid #ddd; padding-top: 8px; margin-top: 14px; }}
.case h2 {{ font-size: 14px; margin: 0 0 6px; color: #5a3fc0; }}
.scn {{ background: #f6f6fb; border: 1px solid #e4e4ef; border-radius: 6px; padding: 6px 9px; margin-bottom: 8px; }}
.scn .pool {{ font-size: 12px; }}
.scn .vals {{ font-size: 10.5px; }}
.cards {{ display: flex; gap: 10px; align-items: flex-start; }}
.card {{ flex: 1 1 0; border: 1px solid #e0e0e0; border-radius: 7px; padding: 8px; min-width: 0; }}
.mlabel {{ font-weight: 800; color: #5a3fc0; margin-bottom: 5px; font-size: 11px; }}
.alloc {{ font-size: 10.5px; margin-bottom: 5px; }}
.alloc.nodeal {{ color: #b00; font-weight: 600; }}
.pareto {{ font-size: 10px; font-weight: 700; padding: 4px 7px; border-radius: 6px; margin-bottom: 3px; }}
.pareto.ok {{ background: #e6f7ec; border: 1px solid #9bdcb2; color: #1a7a3f; }}
.pareto.bad {{ background: #fdf3df; border: 1px solid #e8cd86; color: #8a6d16; }}
.effsplit {{ font-size: 9.5px; color: #666; margin-bottom: 4px; }}
.tmeta {{ font-size: 9px; color: #999; margin: 3px 0; }}
.transcript {{ border-top: 1px dashed #e0e0e0; padding-top: 5px; }}
.msg {{ font-size: 9.5px; margin: 2px 0; }}
.msg .sp {{ font-weight: 700; }}
.msg.you {{ color: #1f4fb0; }}
.msg.them {{ color: #444; }}
.tag {{ color: #b05; font-family: ui-monospace, Menlo, monospace; }}
</style></head><body>
<h1>Pareto Efficiency as an RLVR Target — Case Studies</h1>
<div class="sub">Frontier models vs Qwen3.5-9B · same scenarios · negotiation (Deal-or-No-Deal)</div>
{intro}
{''.join(sections)}
</body></html>"""


def main():
    payload = json.loads(CURATED.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(build_html(payload))
    print(f"wrote {HTML_OUT}")

    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None) or shutil.which("chromium")
    if not chrome:
        print("No Chrome/Chromium found — open the HTML and Print → Save as PDF.")
        return
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-sandbox",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if PDF_OUT.exists():
        print(f"wrote {PDF_OUT}")
    else:
        print("PDF render failed:\n", r.stderr[-1500:])


if __name__ == "__main__":
    main()
