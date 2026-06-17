from __future__ import annotations

import ctypes
import ctypes.util
import contextlib
import io
import os
import platform
import socket
import subprocess
import sys
import uuid
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


def _nvidia_smi_query(device_identifier: str | None = None) -> dict[str, str] | None:
    args = ["nvidia-smi"]
    if device_identifier:
        args.append(f"--id={device_identifier}")
    args.extend(
        [
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    output = _run_command(args)
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


def _load_cuda_driver() -> Any:
    library_path = ctypes.util.find_library("cuda")
    candidates = [library_path, "libcuda.so.1", "libcuda.so", "nvcuda.dll"]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("; ".join(errors) or "CUDA driver library not found")


def _check_cuda(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA driver error {result}")


def _format_cuda_uuid(raw: bytes) -> str:
    return f"GPU-{uuid.UUID(bytes=raw)}"


def _cuda_driver_identity(logical_device: int) -> dict[str, Any]:
    try:
        driver = _load_cuda_driver()

        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetPCIBusId.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
        driver.cuDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuDeviceGetName.restype = ctypes.c_int

        uuid_function = getattr(driver, "cuDeviceGetUuid_v2", None) or getattr(
            driver,
            "cuDeviceGetUuid",
        )
        uuid_function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        uuid_function.restype = ctypes.c_int

        _check_cuda(driver.cuInit(0), "cuInit")
        device = ctypes.c_int()
        _check_cuda(
            driver.cuDeviceGet(ctypes.byref(device), logical_device),
            "cuDeviceGet",
        )

        uuid_buffer = (ctypes.c_ubyte * 16)()
        _check_cuda(
            uuid_function(ctypes.byref(uuid_buffer), device.value),
            "cuDeviceGetUuid",
        )

        pci_buffer = ctypes.create_string_buffer(64)
        _check_cuda(
            driver.cuDeviceGetPCIBusId(pci_buffer, len(pci_buffer), device.value),
            "cuDeviceGetPCIBusId",
        )

        name_buffer = ctypes.create_string_buffer(256)
        _check_cuda(
            driver.cuDeviceGetName(name_buffer, len(name_buffer), device.value),
            "cuDeviceGetName",
        )

        gpu_uuid = _format_cuda_uuid(bytes(uuid_buffer))
        pci_bus_id = pci_buffer.value.decode("ascii", errors="replace")
        return {
            "source": "cuda_driver_api",
            "available": True,
            "cuda_ordinal": logical_device,
            "driver_device": device.value,
            "uuid": gpu_uuid,
            "pci_bus_id": pci_bus_id,
            "name": name_buffer.value.decode("utf-8", errors="replace"),
            "accelerator_id": gpu_uuid,
        }
    except Exception as error:
        return {
            "source": "cuda_driver_api",
            "available": False,
            "cuda_ordinal": logical_device,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def _fallback_accelerator_id(
    *,
    logical_device: int,
    driver_identity: dict[str, Any],
    nvidia_smi: dict[str, str] | None,
) -> str:
    hostname = socket.gethostname()
    if driver_identity.get("uuid"):
        return str(driver_identity["uuid"])
    if nvidia_smi and nvidia_smi.get("uuid"):
        return nvidia_smi["uuid"]
    if driver_identity.get("pci_bus_id"):
        return f"{hostname}/{driver_identity['pci_bus_id']}"
    if nvidia_smi and nvidia_smi.get("pci_bus_id"):
        return f"{hostname}/{nvidia_smi['pci_bus_id']}"
    return f"{hostname}/cuda:{logical_device}"


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
        driver_identity = _cuda_driver_identity(logical_device)
        driver_uuid = driver_identity.get("uuid")
        nvidia_smi = _nvidia_smi_query(driver_uuid) if driver_uuid else None
        sm_count = getattr(props, "multi_processor_count", None)
        accelerator_id = _fallback_accelerator_id(
            logical_device=logical_device,
            driver_identity=driver_identity,
            nvidia_smi=nvidia_smi,
        )
        fingerprint["device"].update(
            {
                "accelerator_id": accelerator_id,
                "cuda_driver": driver_identity,
                "name": cuda.get_device_name(logical_device),
                "capability": list(cuda.get_device_capability(logical_device)),
                "total_memory": props.total_memory,
                "multi_processor_count": sm_count,
                "nvidia_smi": nvidia_smi,
            }
        )

    return fingerprint
