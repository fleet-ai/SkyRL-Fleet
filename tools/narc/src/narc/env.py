from __future__ import annotations

import contextlib
import io
import os
import platform
import socket
import subprocess
import sys
from typing import Any

from narc.checksum import stable_hash


DETERMINISTIC_ENV_DEFAULTS = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def prepare_deterministic_environment() -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in DETERMINISTIC_ENV_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied


def slurm_context() -> dict[str, str | None]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_JOB_NODELIST",
        "SLURM_STEP_ID",
        "SLURM_GPUS_ON_NODE",
    ]
    return {key.lower(): os.environ.get(key) for key in keys}


def _run_command(args: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return output.strip()


def _nvidia_smi_for_device(device_identifier: str) -> dict[str, str] | None:
    output = _run_command(
        [
            "nvidia-smi",
            f"--id={device_identifier}",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 5:
        return {"raw": first_line}
    return {
        "index": parts[0],
        "uuid": parts[1],
        "pci_bus_id": parts[2],
        "name": parts[3],
        "driver_version": parts[4],
    }


def _visible_device_identifier(logical_device: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(logical_device)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if logical_device < len(entries):
        return entries[logical_device]
    return str(logical_device)


def _torch_config_hash(torch: Any) -> tuple[str | None, str | None]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        returned = torch.__config__.show()
    config = returned if isinstance(returned, str) else stream.getvalue()
    if not config:
        return None, None
    return stable_hash(config), config


def collect_fingerprint(
    torch: Any,
    *,
    device_type: str,
    logical_device: int,
) -> dict[str, Any]:
    config_hash, config_text = _torch_config_hash(torch)
    fingerprint: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hostname": socket.gethostname(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "git_version": getattr(torch.version, "git_version", None),
            "cudnn_version": (
                torch.backends.cudnn.version()
                if hasattr(torch.backends, "cudnn")
                else None
            ),
            "config_hash": config_hash,
        },
        "device": {
            "type": device_type,
            "logical_index": logical_device,
        },
    }
    if config_text is not None:
        fingerprint["torch"]["config"] = config_text

    if device_type == "cuda":
        cuda = torch.cuda
        props = cuda.get_device_properties(logical_device)
        device_identifier = _visible_device_identifier(logical_device)
        sm_count = getattr(props, "multi_processor_count", None)
        fingerprint["device"].update(
            {
                "visible_identifier": device_identifier,
                "name": cuda.get_device_name(logical_device),
                "capability": list(cuda.get_device_capability(logical_device)),
                "total_memory": props.total_memory,
                "multi_processor_count": sm_count,
                "nvidia_smi": _nvidia_smi_for_device(device_identifier),
            }
        )

    return fingerprint
