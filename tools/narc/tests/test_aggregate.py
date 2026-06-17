import json

from narc.aggregate import aggregate_path


def test_aggregate_counts_results_and_failures(tmp_path):
    good = {
        "status": "pass",
        "hostname": "node-a",
        "run_id": "run-a",
        "slurm": {"slurm_procid": "0", "slurm_localid": "0"},
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {
            "device": {
                "accelerator_id": "GPU-a",
                "logical_index": 0,
                "cuda_driver": {
                    "uuid": "GPU-a",
                    "pci_bus_id": "00000000:3b:00.0",
                },
            }
        },
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {"timing": {"tokens_per_second": 10.0}},
        "errors": [],
    }
    bad = {
        "status": "fail",
        "hostname": "node-b",
        "run_id": "run-b",
        "slurm": {"slurm_procid": "1", "slurm_localid": "1"},
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {
            "device": {
                "accelerator_id": "GPU-b",
                "logical_index": 0,
                "cuda_driver": {
                    "uuid": "GPU-b",
                    "pci_bus_id": "00000000:4c:00.0",
                },
            }
        },
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-b"},
        "measurements": {"timing": {"tokens_per_second": 5.0}},
        "errors": [{"type": "RuntimeError", "message": "boom"}],
    }
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["loaded_results"] == 2
    assert summary["status_counts"] == {
        "pass": 1,
        "warn": 0,
        "fail": 1,
        "unknown": 0,
    }
    assert summary["fingerprint_hashes"] == {"fingerprint-a": 2}
    assert summary["accelerator_ids"] == {"GPU-a": 1, "GPU-b": 1}
    assert summary["devices"][0]["accelerator_id"] == "GPU-b"
    assert summary["performance"]["tokens_per_second"]["median"] == 7.5
    assert not summary["pass"]
    assert summary["failures"][0]["run_id"] == "run-b"
    assert summary["failures"][0]["accelerator_id"] == "GPU-b"


def test_aggregate_reports_corrupt_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["loaded_results"] == 0
    assert summary["load_errors"][0]["type"] == "JSONDecodeError"
    assert not summary["pass"]
