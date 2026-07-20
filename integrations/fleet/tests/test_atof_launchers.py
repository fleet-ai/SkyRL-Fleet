"""Launcher defaults for NeMo ATOF event emission."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
DEFAULT_ON = 'export SKYRL_ATOF_ENABLED="${SKYRL_ATOF_ENABLED:-1}"'


@pytest.mark.parametrize(
    "script",
    [
        "scripts/fleet-common-setup.sh",
        "scripts/fleet-common-run.sh",
        "scripts/fleet-tinker-tool-use-run.sh",
        "scripts/fleet-tinker-eval-run.sh",
    ],
)
def test_launchers_enable_atof_by_default(script):
    assert DEFAULT_ON in (REPO_ROOT / script).read_text()


@pytest.mark.parametrize(
    "task",
    [
        "tasks/openenv-fleet-grpo-minimax-m27.yaml",
        "tasks/openenv-fleet-grpo-qwen3_5-35b-vl.yaml",
        "tasks/openenv-fleet-grpo-qwen3_5-35b.yaml",
        "tasks/openenv-fleet-grpo-qwen3_6-35b-vl.yaml",
        "tasks/openenv-fleet-grpo-vl.yaml",
    ],
)
def test_production_tasks_enable_atof(task):
    assert 'SKYRL_ATOF_ENABLED: "1"' in (REPO_ROOT / task).read_text()
