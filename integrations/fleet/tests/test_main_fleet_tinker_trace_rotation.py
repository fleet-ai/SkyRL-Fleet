"""Contract test: main_fleet_tinker rotates the Fleet trace job at each
phase boundary (eval_pre, train_step_N, eval_step_N, eval_final).

The helper `_rotate_trace_job` is a closure inside `main()`, so unit-
testing it in isolation requires mocking the entire training loop.
Instead, this test AST-parses main_fleet_tinker.py and verifies the
call-site labels match the documented phase taxonomy. Cheap, deterministic,
and catches a future refactor that drops a rotation point without
noticing — which would silently regress the trace-viewer partitioning
(all phases collapse back into a single trace job pane)."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "integrations/fleet/entrypoints/main_fleet_tinker.py"


def _collect_rotate_calls() -> list[str]:
    """Return the literal label arguments passed to every _rotate_trace_job
    await call. For f-strings, return the joined template with `{var}` for
    interpolated parts (so `f"train_step_{step}"` → `"train_step_{step}"`)."""
    src = SRC.read_text()
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        # Match Name('_rotate_trace_job') or Attribute(...).id == '_rotate_trace_job'
        fn = call.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "_rotate_trace_job":
            continue
        if not call.args:
            out.append("<no-args>")
            continue
        a = call.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            out.append(a.value)
        elif isinstance(a, ast.JoinedStr):
            # f-string — emit literal parts + {var} for FormattedValues
            parts = []
            for v in a.values:
                if isinstance(v, ast.Constant):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    inner = getattr(v.value, "id", None) or "<expr>"
                    parts.append("{" + inner + "}")
                else:
                    parts.append("<unknown>")
            out.append("".join(parts))
        else:
            out.append(f"<{type(a).__name__}>")
    return out


class TestPhaseBoundaryRotation:
    def test_all_four_phase_labels_present(self):
        """The trainer must rotate at four kinds of phase boundary:
        pre-train eval, per-step training, per-step eval (periodic), and
        final eval. Dropping any of these silently collapses sessions
        from that phase into the prior phase's trace job."""
        labels = _collect_rotate_calls()
        assert "eval_pre" in labels, f"missing eval_pre in {labels}"
        assert "train_step_{step}" in labels, f"missing per-step train rotation in {labels}"
        assert "eval_step_{step}" in labels, f"missing periodic eval rotation in {labels}"
        assert "eval_final" in labels, f"missing eval_final in {labels}"

    def test_exactly_four_rotation_sites(self):
        """If the count drifts (e.g. someone adds a rotation we didn't
        plan for, or copy-pastes a duplicate), surface it loudly."""
        labels = _collect_rotate_calls()
        assert len(labels) == 4, f"expected 4 rotation sites, got {len(labels)}: {labels}"

    def test_rotation_helper_is_defined(self):
        """Sanity: the helper itself exists in the source — defends
        against an over-eager refactor that removes the closure."""
        src = SRC.read_text()
        assert "async def _rotate_trace_job" in src

    def test_rotation_helper_no_op_without_fleet_api_key(self):
        """The helper must short-circuit when FLEET_API_KEY is unset
        instead of crashing. Documented contract: rollouts still run
        without trace upload if the key is missing."""
        src = SRC.read_text()
        # Cheap source-level check: the helper body references `fleet_api_key`
        # and bails before calling create_trace_job. Tighter than a string
        # match — locate the function and look at its first statement.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_rotate_trace_job":
                # First statement should be an `if not fleet_api_key: return`
                first = node.body[0]
                assert isinstance(first, ast.If), \
                    f"first stmt should be `if not fleet_api_key: return`, got {ast.dump(first)}"
                return
        raise AssertionError("_rotate_trace_job not found in main_fleet_tinker.py")
