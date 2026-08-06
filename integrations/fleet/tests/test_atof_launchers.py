"""Launcher defaults for NeMo ATOF event emission."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
DEFAULT_ON = "export NEMO_RELAY_ENABLED=1"
RELAY_INSTALLERS = [
    "scripts/fleet-common-setup.sh",
    "scripts/fleet-negotiation-setup.sh",
    "scripts/fleet-tinker-tool-use-run.sh",
    "scripts/fleet-tinker-eval-run.sh",
]
RUNTIME_LAUNCHERS = [
    "scripts/fleet-common-run.sh",
    "scripts/fleet-tinker-tool-use-run.sh",
    "scripts/fleet-tinker-eval-run.sh",
]


@pytest.mark.parametrize(
    "script",
    RUNTIME_LAUNCHERS,
)
def test_launchers_enable_atof_by_default(script):
    assert DEFAULT_ON in (REPO_ROOT / script).read_text()


@pytest.mark.parametrize("script", RELAY_INSTALLERS)
def test_setup_paths_install_nemo_relay(script):
    source = (REPO_ROOT / script).read_text()
    assert "fleet-nemo-relay-artifacts/wheels/latest" in source
    assert 'NEMO_RELAY_ARCH="$(uname -m)"' in source
    assert '"$NEMO_WHEEL_DIR"/nemo_relay-*-"$NEMO_RELAY_ARCH".whl' in source
    assert '"$NEMO_WHEEL_DIR"/nemo_relay_runtime-*.whl' in source


@pytest.mark.parametrize("script", RUNTIME_LAUNCHERS)
def test_runtime_launchers_export_shared_nemo_config(script):
    source = (REPO_ROOT / script).read_text()
    assert "export NEMO_RELAY_ENABLED=1" in source
    assert "SKYRL_ATOF_ENABLED" not in source
    assert "THESEUS_ATOF_ENABLED" not in source
    assert "THESEUS_ATOF_MSK_BROKERS" in source
    assert 'THESEUS_ATOF_TENANT_ID="${THESEUS_ATOF_TENANT_ID:-skyrl}"' in source


def test_task_configs_inherit_shared_atof_default():
    tasks_root = REPO_ROOT / "tasks"
    task_files = [*tasks_root.rglob("*.yaml"), *tasks_root.rglob("*.yml")]
    overrides = [str(path.relative_to(REPO_ROOT)) for path in task_files if "NEMO_RELAY_ENABLED" in path.read_text()]
    assert overrides == []


def test_standard_auto_train_installs_datasets():
    source = (REPO_ROOT / ".github/workflows/auto-train.yaml").read_text()
    assert '"datasets>=2.20"' in source
