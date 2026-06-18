from narc.env import (
    fallback_accelerator_id,
    format_cuda_uuid,
    normalize_gpu_uuid,
    torch_device_identity,
    visible_device_identifier,
)


class TorchUuid:
    bytes = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]


class DeviceProperties:
    uuid = TorchUuid()
    pci_bus_id = "00000000:3b:00.0"


def test_format_cuda_uuid_uses_nvidia_gpu_prefix():
    raw = bytes.fromhex("00112233445566778899aabbccddeeff")

    assert format_cuda_uuid(raw) == "GPU-00112233-4455-6677-8899-aabbccddeeff"


def test_normalize_gpu_uuid_accepts_torch_uuid_bytes():
    assert normalize_gpu_uuid(TorchUuid()) == "GPU-00112233-4455-6677-8899-aabbccddeeff"


def test_normalize_gpu_uuid_adds_prefix_to_bare_uuid_string():
    assert (
        normalize_gpu_uuid("00112233-4455-6677-8899-aabbccddeeff")
        == "GPU-00112233-4455-6677-8899-aabbccddeeff"
    )


def test_torch_device_identity_is_json_serializable():
    assert torch_device_identity(DeviceProperties()) == {
        "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff",
        "pci_bus_id": "00000000:3b:00.0",
    }


def test_accelerator_id_prefers_driver_uuid_without_hostname():
    accelerator_id = fallback_accelerator_id(
        logical_device=0,
        driver_identity={"uuid": "GPU-driver"},
        torch_identity={"uuid": "GPU-torch"},
        nvidia_smi={"uuid": "GPU-smi", "pci_bus_id": "00000000:00:00.0"},
    )

    assert accelerator_id == "GPU-driver"


def test_accelerator_id_uses_host_for_pci_fallback(monkeypatch):
    monkeypatch.setattr("narc.env.socket.gethostname", lambda: "node-a")

    accelerator_id = fallback_accelerator_id(
        logical_device=0,
        driver_identity={},
        torch_identity={},
        nvidia_smi={"pci_bus_id": "00000000:3b:00.0"},
    )

    assert accelerator_id == "node-a/00000000:3b:00.0"


def test_visible_device_identifier_prefers_assigned_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    assert visible_device_identifier(1) == "GPU-b"
