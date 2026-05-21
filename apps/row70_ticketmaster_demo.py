#!/usr/bin/env python3
"""Streamlit browser for downloaded Ticketmaster eval trajectories."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    name: str
    short_name: str
    jsonl: Path
    image_dir: Path
    success_threshold: float
    color: str
    note: str


RUNS = {
    "Taste-shaped GRPO": RunConfig(
        name="Taste-shaped GRPO",
        short_name="taste",
        jsonl=ROOT / "local_runs/tm-grpo-taste-s42-v2/global_step_4/ticketmaster.jsonl",
        image_dir=ROOT / "local_runs/tm-grpo-taste-s42-v2/global_step_4/images",
        success_threshold=0.5,
        color="#2A9D8F",
        note="Success proxy is reward >= 0.5 for the shaped reward run.",
    ),
    "Verifier-only GRPO": RunConfig(
        name="Verifier-only GRPO",
        short_name="verifier",
        jsonl=ROOT / "local_runs/tm-grpo-verifier-s42/global_step_4/ticketmaster.jsonl",
        image_dir=ROOT / "local_runs/tm-grpo-verifier-s42/global_step_4/images",
        success_threshold=1.0,
        color="#457B9D",
        note="Success is binary verifier reward == 1.",
    ),
    "Qwen 9B baseline": RunConfig(
        name="Qwen 9B baseline",
        short_name="baseline",
        jsonl=ROOT / "local_runs/tm-qwen35-9b-baseline-s42/global_step_0/ticketmaster.jsonl",
        image_dir=ROOT / "local_runs/tm-qwen35-9b-baseline-s42/global_step_0/images",
        success_threshold=1.0,
        color="#8D99AE",
        note="Out-of-the-box Qwen eval-only baseline.",
    ),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.15rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: 0 !important; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 11px 13px;
        }
        .run-card {
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 13px 15px;
            background: #ffffff;
        }
        .pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.75rem;
            color: white;
            margin-right: 6px;
        }
        .muted { color: #566070; font-size: 0.92rem; }
        .mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def summed_score(row: dict[str, Any]) -> float:
    raw = row.get("score", 0.0)
    if isinstance(raw, list):
        return float(sum(float(x) for x in raw))
    return float(raw)


def extract_task(prompt: str) -> str:
    prompt = str(prompt or "")
    match = re.search(r"<\|im_start\|>user\n(.*?)(?:Here is feedback|<\|vision_start\|>|<\|im_end\|>)", prompt, re.S)
    if not match:
        return re.sub(r"\s+", " ", prompt[:900]).strip() or "(task prompt unavailable)"
    return re.sub(r"\s+", " ", match.group(1)).strip()


def parse_tool_calls(response: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for turn, match in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response or "", re.S), start=1):
        try:
            payload = json.loads(match.group(1))
            args = payload.get("arguments", {})
        except json.JSONDecodeError:
            calls.append({"turn": turn, "action": "malformed", "detail": match.group(1)[:120]})
            continue

        action = args.get("action") or payload.get("name") or "unknown"
        if action == "left_click":
            detail = str(args.get("coordinate"))
            normalized = f"click {detail}"
        elif action == "type":
            detail = repr(args.get("text", ""))
            normalized = f"type {detail}"
        elif action == "scroll":
            detail = f"{args.get('scroll_direction')} {args.get('scroll_amount')}"
            normalized = f"scroll {detail}"
        elif action == "wait":
            detail = f"{args.get('duration')}s"
            normalized = f"wait {detail}"
        else:
            detail = json.dumps(args, ensure_ascii=False)[:140]
            normalized = str(action)
        calls.append({"turn": turn, "action": str(action), "detail": detail, "normalized": normalized})
    return calls


def clean_response_text(response: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", response or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("user [image]", " ")
    return re.sub(r"\s+", " ", text).strip()


def trajectory_markers(response: str) -> dict[str, bool]:
    text = (response or "").lower()
    checks = {
        "Tool error": ["failed after 8 attempts", "error: call_tool"],
        "Get Tickets": ["get tickets"],
        "Ticket terms": ["what you need to know", "i agree"],
        "Ticket selection": ["ticket selection", "select 2 tickets", "section", "row"],
        "Checkout": ["checkout", "order summary"],
        "Complete Purchase": ["complete purchase"],
        "Confirmation": ["confirmation", "order number", "order confirmation"],
    }
    return {name: any(term in text for term in terms) for name, terms in checks.items()}


@st.cache_data(show_spinner=False)
def load_rows(run_name: str) -> list[dict[str, Any]]:
    cfg = RUNS[run_name]
    rows: list[dict[str, Any]] = []
    with cfg.jsonl.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            response = row.get("output_response") or ""
            calls = parse_tool_calls(response)
            score = summed_score(row)
            task = extract_task(row.get("input_prompt", ""))
            rows.append(
                {
                    "idx": idx,
                    "row": row,
                    "score": score,
                    "success": score >= cfg.success_threshold,
                    "task": task,
                    "calls": calls,
                    "markers": trajectory_markers(response),
                    "screenshots": int(row.get("num_screenshots") or len(row.get("image_paths") or [])),
                    "response_text": clean_response_text(response),
                }
            )
    return rows


def image_path(cfg: RunConfig, row_idx: int, image_idx: int) -> Path:
    return cfg.image_dir / f"eval_{row_idx:04d}_img_{image_idx:03d}.jpg"


def trajectory_option(item: dict[str, Any]) -> str:
    outcome = "PASS" if item["success"] else "FAIL"
    task = item["task"]
    if len(task) > 90:
        task = task[:87] + "..."
    return f"row {item['idx']:03d} | {outcome} | score={item['score']:.3f} | {task}"


def repeated_actions(calls: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return Counter(call.get("normalized", "") for call in calls).most_common(8)


def image_indices(n: int, mode: str) -> list[int]:
    if n <= 0:
        return []
    if mode == "First/middle/last":
        raw = [0, 1, 2, n // 4, n // 2, (3 * n) // 4, max(0, n - 2), n - 1]
    else:
        step = max(1, n // 12)
        raw = list(range(0, n, step))[:12]
        if raw[-1] != n - 1:
            raw.append(n - 1)
    deduped: list[int] = []
    for idx in raw:
        if 0 <= idx < n and idx not in deduped:
            deduped.append(idx)
    return deduped


def render_header(cfg: RunConfig, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    successes = sum(1 for row in rows if row["success"])
    st.markdown(
        f"""
        <div class="run-card">
          <span class="pill" style="background:{cfg.color}">{cfg.name}</span>
          <span class="muted">{cfg.note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajectories", n)
    c2.metric("Successes", successes)
    c3.metric("Pass rate", f"{successes / n:.0%}" if n else "n/a")
    c4.metric("Avg screenshots", f"{sum(r['screenshots'] for r in rows) / n:.1f}" if n else "n/a")


def render_metrics(selected: dict[str, Any]) -> None:
    calls = selected["calls"]
    repeated = repeated_actions(calls)
    top_repeat = repeated[0][1] if repeated else 0
    unique_actions = len({call.get("normalized", "") for call in calls})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Row", f"{selected['idx']:03d}")
    c2.metric("Score", f"{selected['score']:.3f}")
    c3.metric("Outcome", "Success" if selected["success"] else "Fail")
    c4.metric("Actions", len(calls))
    c5.metric("Top repeat", f"{top_repeat}x")
    st.caption(f"Unique normalized actions: {unique_actions}")


def render_markers(selected: dict[str, Any]) -> None:
    markers = selected["markers"]
    cols = st.columns(len(markers))
    for col, (name, present) in zip(cols, markers.items()):
        col.metric(name, "yes" if present else "no")


def render_screenshots(cfg: RunConfig, selected: dict[str, Any]) -> None:
    n = selected["screenshots"]
    if n <= 0:
        st.warning("No screenshots found for this trajectory.")
        return

    view = st.radio("Screenshot view", ["Step slider", "First/middle/last", "Timeline thumbnails"], horizontal=True)
    if view == "Step slider":
        idx = st.slider("Screenshot index", 0, n - 1, min(n - 1, 0))
        path = image_path(cfg, selected["idx"], idx)
        st.image(str(path), caption=f"row {selected['idx']:03d}, screenshot {idx:03d}", use_container_width=True)
        return

    indices = image_indices(n, view)
    for start in range(0, len(indices), 4):
        cols = st.columns(4)
        for col, idx in zip(cols, indices[start : start + 4]):
            path = image_path(cfg, selected["idx"], idx)
            if path.exists():
                col.image(str(path), caption=f"step {idx:03d}", use_container_width=True)
            else:
                col.warning(f"Missing {idx:03d}")


def render_actions(selected: dict[str, Any]) -> None:
    calls = selected["calls"]
    st.markdown("**Most repeated actions**")
    for action, count in repeated_actions(calls):
        st.markdown(f"- `{action}`: **{count}x**")

    with st.expander("Full parsed action list"):
        st.dataframe(
            [{"turn": c["turn"], "action": c["action"], "detail": c["detail"]} for c in calls],
            use_container_width=True,
            height=360,
        )


def render_transcript(selected: dict[str, Any]) -> None:
    text = selected["response_text"]
    st.text_area("Cleaned transcript text", value=text, height=360)


def main() -> None:
    st.set_page_config(page_title="Ticketmaster Trajectory Browser", layout="wide")
    inject_css()

    st.title("Ticketmaster Trajectory Browser")
    st.caption("Browse downloaded S3 eval rollouts. Use this as a live fallback to inspect successful or failed trajectories.")

    run_name = st.sidebar.selectbox("Run", list(RUNS.keys()))
    cfg = RUNS[run_name]
    rows = load_rows(run_name)

    outcome_filter = st.sidebar.radio("Outcome filter", ["Successes", "Failures", "All"], horizontal=False)
    if outcome_filter == "Successes":
        candidates = [row for row in rows if row["success"]]
    elif outcome_filter == "Failures":
        candidates = [row for row in rows if not row["success"]]
    else:
        candidates = rows

    marker_filter = st.sidebar.multiselect(
        "Require marker",
        ["Get Tickets", "Ticket terms", "Ticket selection", "Checkout", "Complete Purchase", "Confirmation", "Tool error"],
        default=[],
    )
    if marker_filter:
        candidates = [row for row in candidates if all(row["markers"].get(marker, False) for marker in marker_filter)]

    if not candidates:
        st.error("No trajectories match the selected filters.")
        return

    default_idx = 0
    selected_label = st.sidebar.selectbox(
        "Trajectory",
        [trajectory_option(row) for row in candidates],
        index=default_idx,
    )
    selected = candidates[[trajectory_option(row) for row in candidates].index(selected_label)]

    render_header(cfg, rows)

    st.subheader("Selected Trajectory")
    with st.expander("Task prompt", expanded=True):
        st.write(selected["task"])

    render_metrics(selected)

    st.subheader("Progress Markers")
    render_markers(selected)

    left, right = st.columns([1.25, 0.75])
    with left:
        st.subheader("Screenshots")
        render_screenshots(cfg, selected)
    with right:
        st.subheader("Actions")
        render_actions(selected)

    st.subheader("Transcript")
    render_transcript(selected)

    with st.expander("Source files"):
        st.markdown(
            f"""
            - JSONL: `{cfg.jsonl.relative_to(ROOT)}`
            - Images: `{cfg.image_dir.relative_to(ROOT)}`
            - Row index: `{selected['idx']}`
            - Success threshold: `{cfg.success_threshold}`
            """
        )


if __name__ == "__main__":
    main()
