"""End-to-end orchestration test for the Tinker auto-train pipeline.

Verifies that for every supported modality (tool_use, browser_use,
computer_use) the launcher correctly:
  1. Splits tasks 90/10 into train/holdout parquets,
  2. Invokes the fleet-tinker-tool-use-run.sh wrapper with the right env,
  3. Parses the results JSON the wrapper produces,
  4. Returns a (success=True, results) tuple whose post_pass_rate > pre_pass_rate.

The wrapper script is replaced with a stub that emits a deterministic
results JSON with pre < post — this isolates the orchestration from the
actual Tinker cloud call. The real model-level lift comes from running
this same wrapper against Tinker cloud in CI; see
docs/content/docs/tinker/auto-train.mdx for that path.

Run:
    pytest integrations/fleet/tests/test_tinker_auto_train.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from integrations.fleet.auto_train import tinker_launcher
from integrations.fleet.auto_train.splitter import split_90_10
from integrations.fleet.auto_train.tinker_config import TINKER_MODALITY_SUPPORT

MODALITIES = sorted(TINKER_MODALITY_SUPPORT)
assert MODALITIES == ["browser_use", "computer_use", "tool_use"]


def _make_tasks(n: int, modality: str) -> list[dict]:
    """Build n fake tasks shaped like exporter.build_openenv_tasks output."""
    return [
        {
            "task_key": f"{modality}_task_{i:04d}",
            "prompt": f"[{modality}] solve problem {i}",
            "env_key": "demo-env",
            "env_version": "v1",
            "data_key": "",
            "data_version": "",
            "task_modality": modality,
            "verifier_code": "def reward(*a, **kw): return 1.0\n",
            "env_variables": {},
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Create a fake repo root with scripts/fleet-tinker-tool-use-run.sh
    replaced by a stub that emits a results JSON with known pre/post.

    The stub reads MODEL_NAME, MAX_STEPS, DATASET_FILE, EVAL_DATASET_FILE,
    and RESULTS_OUT from env (same contract as the real script) and writes
    a JSON the launcher can parse.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    stub = scripts_dir / "fleet-tinker-tool-use-run.sh"
    stub.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        : "${TASKS_FILE:?}"
        : "${DATASET_FILE:?}"
        : "${EVAL_DATASET_FILE:?}"
        : "${RESULTS_OUT:?}"
        : "${MODEL_NAME:?}"
        : "${MAX_STEPS:?}"
        # Verify the parquet files exist (proves splitter ran).
        test -s "$DATASET_FILE"
        test -s "$EVAL_DATASET_FILE"
        # Emit a deterministic results JSON with pre < post.
        mkdir -p "$(dirname "$RESULTS_OUT")"
        cat > "$RESULTS_OUT" <<JSON
        {
          "model_name": "${MODEL_NAME}",
          "num_steps": ${MAX_STEPS},
          "n_train": 18,
          "n_holdout": 2,
          "pre_pass_rate": 0.10,
          "post_pass_rate": 0.40,
          "delta": 0.30,
          "entries": [
            {"step": -1, "pass_at_1": 0.10, "num_samples": 2},
            {"step": ${MAX_STEPS}, "pass_at_1": 0.40, "num_samples": 2}
          ],
          "wandb_url": "https://wandb.test/fake/run",
          "wandb_run_name": "test"
        }
        JSON
    """))
    stub.chmod(0o755)
    return tmp_path


def _set_required_env(monkeypatch):
    for var in ("TINKER_API_KEY", "FLEET_API_KEY", "WANDB_API_KEY",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(var, "dummy")


@pytest.mark.parametrize("modality", MODALITIES)
def test_launcher_produces_lift_for_each_modality(
    modality: str,
    fake_repo: Path,
    monkeypatch,
):
    """For each modality: launcher splits 90/10, runs the wrapper, parses
    the results JSON, and returns post_pass_rate > pre_pass_rate."""
    _set_required_env(monkeypatch)
    # Point the launcher at the fake repo root so it uses our stub script.
    monkeypatch.setattr(tinker_launcher, "_repo_root", lambda: fake_repo)

    tasks = _make_tasks(20, modality)
    ok, results = tinker_launcher.launch_training(
        dataset_key=f"demo-{modality}",
        modality=modality,
        tasks=tasks,
        dry_run=False,
        base_model="moonshotai/Kimi-K2.6",
        num_steps=5,
        lora_rank=16,
        eval_results_dir="eval_results",
    )

    assert ok, f"launcher failed for modality={modality}"
    assert results is not None, f"no results JSON parsed for modality={modality}"
    assert results["dataset_key"] == f"demo-{modality}"
    assert results["modality"] == modality
    assert results["model_name"] == "moonshotai/Kimi-K2.6"
    assert results["num_steps"] == 5

    pre = results["pre_pass_rate"]
    post = results["post_pass_rate"]
    delta = results["delta"]
    assert pre is not None and post is not None and delta is not None
    assert post > pre, f"no lift for modality={modality}: pre={pre} post={post}"
    assert delta == pytest.approx(post - pre, abs=1e-6)

    # Splitter wrote the parquets next to results_out.
    results_dir = fake_repo / "eval_results"
    train_pq = results_dir / f"train_demo-{modality}_{modality}.parquet"
    holdout_pq = results_dir / f"holdout_demo-{modality}_{modality}.parquet"
    assert train_pq.exists() and train_pq.stat().st_size > 0
    assert holdout_pq.exists() and holdout_pq.stat().st_size > 0


def test_launcher_dry_run_skips_subprocess(fake_repo: Path, monkeypatch):
    """--dry-run path returns (True, None) without invoking the script."""
    _set_required_env(monkeypatch)
    monkeypatch.setattr(tinker_launcher, "_repo_root", lambda: fake_repo)
    tasks = _make_tasks(20, "tool_use")

    ok, results = tinker_launcher.launch_training(
        dataset_key="demo-dry",
        modality="tool_use",
        tasks=tasks,
        dry_run=True,
    )
    assert ok and results is None
    # No splitter run on dry-run, so no parquets either.
    assert not list((fake_repo / "eval_results").glob("*.parquet")) if (fake_repo / "eval_results").exists() else True


def test_launcher_propagates_failure_when_stub_exits_nonzero(fake_repo: Path, monkeypatch):
    """If the wrapper script fails, launcher returns (False, None)."""
    _set_required_env(monkeypatch)
    # Overwrite the stub with one that exits 1 before writing results.
    (fake_repo / "scripts" / "fleet-tinker-tool-use-run.sh").write_text(
        "#!/usr/bin/env bash\nexit 1\n"
    )
    monkeypatch.setattr(tinker_launcher, "_repo_root", lambda: fake_repo)

    ok, results = tinker_launcher.launch_training(
        dataset_key="demo-fail",
        modality="tool_use",
        tasks=_make_tasks(20, "tool_use"),
        dry_run=False,
    )
    assert ok is False and results is None


def test_launcher_propagates_failure_when_results_missing(fake_repo: Path, monkeypatch):
    """If the wrapper exits 0 but produces no results JSON, launcher returns (False, None)."""
    _set_required_env(monkeypatch)
    (fake_repo / "scripts" / "fleet-tinker-tool-use-run.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    monkeypatch.setattr(tinker_launcher, "_repo_root", lambda: fake_repo)

    ok, results = tinker_launcher.launch_training(
        dataset_key="demo-noresults",
        modality="tool_use",
        tasks=_make_tasks(20, "tool_use"),
        dry_run=False,
    )
    assert ok is False and results is None


def test_splitter_determinism_and_no_overlap(tmp_path: Path):
    """split_90_10 produces non-overlapping splits and the same shape on re-run."""
    tasks = _make_tasks(50, "tool_use")
    tp1, hp1, n_tr1, n_ho1 = split_90_10(
        tasks=tasks, output_dir=str(tmp_path), dataset_key="dk", modality="tool_use",
    )
    tp2, hp2, n_tr2, n_ho2 = split_90_10(
        tasks=tasks, output_dir=str(tmp_path), dataset_key="dk", modality="tool_use",
    )
    assert (n_tr1, n_ho1) == (n_tr2, n_ho2) == (45, 5)

    from datasets import load_dataset
    train_keys = set(load_dataset("parquet", data_files=tp1)["train"]["task_key"])
    holdout_keys = set(load_dataset("parquet", data_files=hp1)["train"]["task_key"])
    assert train_keys.isdisjoint(holdout_keys)
    assert train_keys | holdout_keys == {f"tool_use_task_{i:04d}" for i in range(50)}


def test_splitter_different_seed_per_dataset(tmp_path: Path):
    """Two different (dataset, modality) pairs get different holdout sets."""
    tasks = _make_tasks(50, "tool_use")
    _, hp_a, _, _ = split_90_10(
        tasks=tasks, output_dir=str(tmp_path / "a"),
        dataset_key="dataset-a", modality="tool_use",
    )
    _, hp_b, _, _ = split_90_10(
        tasks=tasks, output_dir=str(tmp_path / "b"),
        dataset_key="dataset-b", modality="tool_use",
    )
    from datasets import load_dataset
    ha = set(load_dataset("parquet", data_files=hp_a)["train"]["task_key"])
    hb = set(load_dataset("parquet", data_files=hp_b)["train"]["task_key"])
    # Distinct salts ⇒ overwhelmingly different holdout sets (allow some overlap
    # at this small sample size, but they must not be identical).
    assert ha != hb
