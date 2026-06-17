import json

from narc.aggregate import aggregate_path


def test_aggregate_counts_results_and_failures(tmp_path):
    good = {
        "status": "pass",
        "hostname": "node-a",
        "run_id": "run-a",
        "fingerprint_hash": "fingerprint-a",
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {"timing": {"tokens_per_second": 10.0}},
        "errors": [],
    }
    bad = {
        "status": "fail",
        "hostname": "node-b",
        "run_id": "run-b",
        "fingerprint_hash": "fingerprint-a",
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
    assert summary["performance"]["tokens_per_second"]["median"] == 7.5
    assert not summary["pass"]
    assert summary["failures"][0]["run_id"] == "run-b"


def test_aggregate_reports_corrupt_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["loaded_results"] == 0
    assert summary["load_errors"][0]["type"] == "JSONDecodeError"
    assert not summary["pass"]
