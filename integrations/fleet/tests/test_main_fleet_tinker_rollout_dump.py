"""Contract tests for the rollout-dump streaming feature.

The streaming-write design exists because the alternative (collect-all-then-
flush) silently throws away N-1 rollouts when the eval crashes mid-batch.
mcpbench OOM'd at 71/104 in a prior run and we had to manually salvage the
trajectory files — these tests lock the contract so that exact regression
can't reappear.

AST-only (matches test_main_fleet_tinker_eval_only.py style) so the tests
run in seconds without pulling Tinker / SkyRL-train / Ray, which aren't
installed in the dev test env.

Run:
    uv run --extra dev --extra tinker pytest \
        integrations/fleet/tests/test_main_fleet_tinker_rollout_dump.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SRC = (
    Path(__file__).resolve().parents[3]
    / "integrations/fleet/entrypoints/main_fleet_tinker.py"
)


def _parse() -> ast.Module:
    return ast.parse(SRC.read_text())


def _find_func(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _func_kwarg_defaults(fn) -> dict:
    """Return {kwarg_name: default_node} for an AST function def."""
    out = {}
    # positional + keyword-only, paired with defaults from the right
    args = list(fn.args.args) + list(fn.args.kwonlyargs)
    defaults = list(fn.args.defaults) + list(fn.args.kw_defaults)
    # Align right-to-left: only the last len(defaults) positional args have defaults
    # but kwonlyargs always pair 1:1 with kw_defaults.
    # Simplest: walk both lists in parallel with explicit None for "no default".
    pos_defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
    kw_defaults = list(fn.args.kw_defaults)
    for arg, dflt in zip(fn.args.args, pos_defaults):
        out[arg.arg] = dflt
    for arg, dflt in zip(fn.args.kwonlyargs, kw_defaults):
        out[arg.arg] = dflt
    return out


# ---------------------------------------------------------------------------
# RolloutOutput exposes messages + sample fields
# ---------------------------------------------------------------------------


def test_rollout_output_exposes_messages_and_sample_fields():
    """messages enables offline analysis; sample makes per-rollout filenames
    deterministic without per-task counters."""
    tree = _parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RolloutOutput":
            field_names = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
            assert "messages" in field_names, (
                "RolloutOutput must expose `messages` for transcript dumps"
            )
            assert "sample" in field_names, (
                "RolloutOutput must expose `sample` so dumpers can write "
                "deterministic filenames (task_key__sN.json) without tracking "
                "per-task counters."
            )
            return
    raise AssertionError("RolloutOutput class not found")


# ---------------------------------------------------------------------------
# Public signatures: new kwargs surfaced everywhere they need to be
# ---------------------------------------------------------------------------


def test_collect_fleet_rollout_has_capture_messages_default_false():
    """Default must be False or training rollouts pay gigabytes of unnecessary
    RAM (200 steps × 16 batch × 8 samples)."""
    fn = _find_func(_parse(), "collect_fleet_rollout")
    defaults = _func_kwarg_defaults(fn)
    assert "capture_messages" in defaults, (
        "collect_fleet_rollout must accept capture_messages"
    )
    d = defaults["capture_messages"]
    assert isinstance(d, ast.Constant) and d.value is False, (
        "capture_messages default must be False — eval phases opt in explicitly"
    )


def test_collect_batch_rollouts_has_callback_and_capture():
    fn = _find_func(_parse(), "collect_batch_rollouts")
    defaults = _func_kwarg_defaults(fn)
    assert "capture_messages" in defaults
    d_cap = defaults["capture_messages"]
    assert isinstance(d_cap, ast.Constant) and d_cap.value is False, (
        "capture_messages default must be False"
    )
    assert "on_rollout_complete" in defaults, (
        "collect_batch_rollouts must accept on_rollout_complete — without it, "
        "dumpers can only run after the entire batch completes and a mid-eval "
        "crash loses every rollout."
    )
    d_cb = defaults["on_rollout_complete"]
    assert isinstance(d_cb, ast.Constant) and d_cb.value is None


def test_main_signature_has_rollout_dump_dir():
    fn = _find_func(_parse(), "main")
    defaults = _func_kwarg_defaults(fn)
    assert "rollout_dump_dir" in defaults


# ---------------------------------------------------------------------------
# CLI surfaces it (or no operator can drive it from the shell wrapper)
# ---------------------------------------------------------------------------


def test_argparse_defines_rollout_dump_dir():
    src = SRC.read_text()
    assert '"--rollout-dump-dir"' in src


# ---------------------------------------------------------------------------
# Wiring: every _run_eval call site passes a phase label so phases don't
# clobber each other in the dump tree
# ---------------------------------------------------------------------------


def test_every_run_eval_call_passes_dump_label():
    """Without per-phase labels, all phases would write into the same subdir
    and overwrite each other (pre vs step_N vs final vs eval_only)."""
    tree = _parse()
    main_fn = _find_func(tree, "main")
    sites = []
    for node in ast.walk(main_fn):
        if not (isinstance(node, ast.Await) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        fn = call.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "_run_eval":
            continue
        kwargs = {kw.arg for kw in call.keywords}
        sites.append(kwargs)
    assert len(sites) >= 4, (
        f"expected ≥4 _run_eval call sites (pre, periodic, final, eval_only), "
        f"found {len(sites)}"
    )
    for kwargs in sites:
        assert "dump_label" in kwargs, (
            f"every _run_eval call must pass dump_label; site keywords: {sorted(kwargs)}"
        )


# ---------------------------------------------------------------------------
# _run_eval body: dumper is wired into collect_batch_rollouts AND
# capture_messages is set (otherwise dumps have empty transcripts)
# ---------------------------------------------------------------------------


def test_run_eval_passes_callback_and_capture_to_collect_batch_rollouts():
    tree = _parse()
    main_fn = _find_func(tree, "main")
    run_eval = None
    for node in ast.walk(main_fn):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_eval":
            run_eval = node
            break
    assert run_eval is not None, "_run_eval closure must exist inside main()"

    for node in ast.walk(run_eval):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "collect_batch_rollouts":
                kw_names = {kw.arg for kw in node.keywords}
                assert "on_rollout_complete" in kw_names, (
                    "_run_eval must thread on_rollout_complete into collect_batch_rollouts"
                )
                assert "capture_messages" in kw_names, (
                    "_run_eval must pass capture_messages — dumps without "
                    "transcripts are useless for offline analysis"
                )
                return
    raise AssertionError("no collect_batch_rollouts call found inside _run_eval")
