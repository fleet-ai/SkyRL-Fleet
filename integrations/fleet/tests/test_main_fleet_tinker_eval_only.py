"""Contract tests for --eval-only mode of main_fleet_tinker.

The eval-only path skips the entire training loop. If a future refactor
silently re-introduces forward_backward or optim_step into that branch,
the eval will start mutating weights and produce noisy, non-reproducible
metrics. These tests AST-walk the file to lock the contract.

Style mirrors test_main_fleet_tinker_trace_rotation.py — AST checks instead
of mocking the full training stack, so they run in seconds without Tinker /
Fleet / WandB credentials.

Run:
    uv run --extra dev --extra tinker pytest \
        integrations/fleet/tests/test_main_fleet_tinker_eval_only.py
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


SRC = (
    Path(__file__).resolve().parents[3]
    / "integrations/fleet/entrypoints/main_fleet_tinker.py"
)


def _parse() -> ast.Module:
    return ast.parse(SRC.read_text())


def _find_func(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in main_fleet_tinker.py")


# ---------------------------------------------------------------------------
# Public signature: eval_only + from_checkpoint exposed on main()
# ---------------------------------------------------------------------------


def test_main_signature_has_eval_only_and_from_checkpoint():
    tree = _parse()
    main = _find_func(tree, "main")
    kw_only = {a.arg for a in main.args.kwonlyargs}
    pos = {a.arg for a in main.args.args}
    params = kw_only | pos
    assert "eval_only" in params, (
        "main() must expose `eval_only` parameter so the CLI flag can drive the "
        "control-flow branch. Without it the entrypoint cannot run eval without "
        "also running the training loop."
    )
    assert "from_checkpoint" in params, (
        "main() must expose `from_checkpoint` parameter — required to bind the "
        "sampling_client to a specific Tinker checkpoint URI instead of the "
        "training_client's evolving weights."
    )


# ---------------------------------------------------------------------------
# Argparse exposes both flags + validation
# ---------------------------------------------------------------------------


def test_argparse_defines_eval_only_and_from_checkpoint_flags():
    """The CLI must accept --eval-only and --from-checkpoint, or operators
    can't drive the eval-only mode from a shell script."""
    src = SRC.read_text()
    assert '"--eval-only"' in src, "argparse missing --eval-only"
    assert '"--from-checkpoint"' in src, "argparse missing --from-checkpoint"


def test_argparse_validates_eval_only_requires_from_checkpoint():
    """Without this guard, `--eval-only` without a checkpoint silently
    falls through to training-client init or fails deep in the SDK with a
    confusing error. The parser.error() call should surface 'requires
    --from-checkpoint' early."""
    src = SRC.read_text()
    assert "--eval-only requires --from-checkpoint" in src, (
        "Argparse must error out with a clear message when --eval-only is "
        "passed without --from-checkpoint"
    )
    assert "--eval-only requires --eval-dataset-file" in src, (
        "Argparse must also require --eval-dataset-file in eval-only mode; "
        "otherwise main() crashes deep in dataset loading with a less "
        "actionable error."
    )


# ---------------------------------------------------------------------------
# Body: eval-only branch skips the training loop and never calls
# forward_backward / optim_step. This is the core safety contract.
# ---------------------------------------------------------------------------


def _eval_only_branch(tree: ast.Module) -> ast.If:
    """Locate the top-level `if eval_only:` (or `if not eval_only:` negation)
    inside main(). Returns the If node so subsequent tests can walk its body."""
    main = _find_func(tree, "main")
    for node in ast.walk(main):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "eval_only":
            return node
    raise AssertionError(
        "main() body must contain `if eval_only:` branch that short-circuits "
        "the training loop. None found — the refactor likely regressed."
    )


def test_eval_only_branch_exists_and_returns():
    """The branch must end with `return` (or `raise`) — otherwise execution
    falls through to the training_client creation and training loop, which
    defeats the purpose of the flag."""
    tree = _parse()
    branch = _eval_only_branch(tree)
    # Walk the branch body for a Return statement at any nesting level.
    returns = [n for n in ast.walk(branch) if isinstance(n, ast.Return)]
    assert returns, (
        "The `if eval_only:` branch must `return` before the training loop. "
        "Without it the branch's eval call runs AND then training proceeds."
    )


def test_eval_only_branch_calls_run_eval():
    """The branch must actually invoke _run_eval — that's the whole point."""
    tree = _parse()
    branch = _eval_only_branch(tree)
    calls = [
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(branch)
        if isinstance(n, ast.Call)
    ]
    assert "_run_eval" in calls, (
        "Eval-only branch must call _run_eval to produce rollouts + metrics. "
        f"Found calls: {sorted(set(calls))[:20]}"
    )


def test_eval_only_branch_never_calls_forward_backward_or_optim_step():
    """Safety contract: the eval-only path is read-only against the
    checkpoint. forward_backward mutates LoRA weights; optim_step applies
    them. Either in the eval branch turns "eval" into "continued training"
    and the reported metrics drift from what you asked to measure."""
    tree = _parse()
    branch = _eval_only_branch(tree)
    forbidden = {"forward_backward", "optim_step", "save_state", "save_weights_for_sampler"}
    seen: list[str] = []
    for n in ast.walk(branch):
        if isinstance(n, ast.Call):
            fn = n.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in forbidden:
                seen.append(name)
    assert not seen, (
        f"Eval-only branch must not call training/checkpoint-writing methods. "
        f"Found: {seen}. Remove these calls — the path must be pure inference."
    )


# ---------------------------------------------------------------------------
# Training-client init is gated on `not eval_only`
# ---------------------------------------------------------------------------


def test_training_client_creation_gated_on_not_eval_only():
    """create_lora_training_client_async is expensive (provisions LoRA
    shards). The eval-only path doesn't need it. The guard ensures we don't
    burn Tinker quota on init for an eval that never trains."""
    src = SRC.read_text()
    # Find the training_client = await service_client.create_lora_training_client_async(...) line
    assert "create_lora_training_client_async" in src
    # And confirm it's gated. Either explicit `if not eval_only:` or an
    # initialization to None + conditional assignment. Accept either.
    tree = _parse()
    main = _find_func(tree, "main")
    # Look for any assignment of `training_client` inside an `if not eval_only` block
    # OR a top-level `training_client = None` followed by conditional assignment.
    found_guard = False
    for node in ast.walk(main):
        if isinstance(node, ast.If):
            test = node.test
            # `if not eval_only:` → UnaryOp(Not, Name(eval_only))
            if (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name)
                and test.operand.id == "eval_only"
            ):
                # body must reassign training_client
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if isinstance(tgt, ast.Name) and tgt.id == "training_client":
                                found_guard = True
    assert found_guard, (
        "Expected `if not eval_only:` block that reassigns `training_client`. "
        "Without it the eval-only path still pays for a LoRA training-client "
        "init it never uses."
    )
