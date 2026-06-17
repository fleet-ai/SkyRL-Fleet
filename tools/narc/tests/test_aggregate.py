import json

import pytest

from narc.aggregate import aggregate_path, generate_aggregate_parser, handle_aggregate


def test_aggregate_counts_results_and_failures(tmp_path):
    good = {
        "schema_version": 1,
        "status": "pass",
        "profile": "correctness",
        "hostname": "node-a",
        "run_id": "run-a",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 100,
        "slurm": {"slurm_procid": "0", "slurm_localid": "0"},
        "command": {},
        "probe_config": {},
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
        "schema_version": 1,
        "status": "fail",
        "profile": "correctness",
        "hostname": "node-b",
        "run_id": "run-b",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 101,
        "slurm": {"slurm_procid": "1", "slurm_localid": "1"},
        "command": {},
        "probe_config": {},
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
    assert summary["output_hash_failures"]
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


def test_aggregate_fails_on_divergent_output_hashes_for_same_config(tmp_path):
    for name, output_hash in (("a.json", "hash-a"), ("b.json", "hash-b")):
        result = {
            "schema_version": 1,
            "status": "pass",
            "profile": "correctness",
            "hostname": name,
            "run_id": name,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "pid": 100,
            "slurm": {},
            "command": {},
            "probe_config": {},
            "fingerprint_hash": "fingerprint-a",
            "fingerprint": {"device": {"accelerator_id": name}},
            "probe_config_hash": "config-a",
            "checks": {"output_hash": output_hash},
            "measurements": {},
            "errors": [],
        }
        (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["output_hash_failures"] == [
        {
            "group": "correctness:config-a",
            "profile": "correctness",
            "probe_config_hash": "config-a",
            "output_hashes": {"hash-a": 1, "hash-b": 1},
            "missing_output_hash": [],
        }
    ]


def test_aggregate_fails_on_duplicate_accelerator_for_same_config(tmp_path):
    for name, procid in (("a.json", "0"), ("b.json", "1")):
        result = {
            "schema_version": 1,
            "status": "pass",
            "profile": "correctness",
            "hostname": "node-a",
            "run_id": name,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "pid": 100,
            "slurm": {"slurm_procid": procid, "slurm_localid": procid},
            "command": {},
            "probe_config": {},
            "fingerprint_hash": "fingerprint-a",
            "fingerprint": {
                "device": {
                    "accelerator_id": "GPU-same",
                    "logical_index": 0,
                }
            },
            "probe_config_hash": "config-a",
            "checks": {"output_hash": "hash-a"},
            "measurements": {},
            "errors": [],
        }
        (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["duplicate_accelerator_failures"] == [
        {
            "group": "correctness:config-a",
            "profile": "correctness",
            "probe_config_hash": "config-a",
            "duplicates": {
                "GPU-same": [
                    {
                        "path": str(tmp_path / "a.json"),
                        "hostname": "node-a",
                        "run_id": "a.json",
                        "logical_index": 0,
                        "slurm_procid": "0",
                        "slurm_localid": "0",
                    },
                    {
                        "path": str(tmp_path / "b.json"),
                        "hostname": "node-a",
                        "run_id": "b.json",
                        "logical_index": 0,
                        "slurm_procid": "1",
                        "slurm_localid": "1",
                    },
                ]
            },
        }
    ]


def test_aggregate_allows_same_accelerator_for_different_configs(tmp_path):
    for name, profile, config_hash in (
        ("correctness.json", "correctness", "config-a"),
        ("performance.json", "performance", "config-b"),
    ):
        result = {
            "schema_version": 1,
            "status": "pass",
            "profile": profile,
            "hostname": "node-a",
            "run_id": name,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "pid": 100,
            "slurm": {},
            "command": {},
            "probe_config": {},
            "fingerprint_hash": "fingerprint-a",
            "fingerprint": {"device": {"accelerator_id": "GPU-same"}},
            "probe_config_hash": config_hash,
            "checks": {"output_hash": "hash-a"},
            "measurements": {},
            "errors": [],
        }
        (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["pass"]
    assert not summary["duplicate_accelerator_failures"]


def test_aggregate_outfile_inside_input_directory_is_not_ingested(tmp_path):
    result = {
        "schema_version": 1,
        "status": "pass",
        "profile": "correctness",
        "hostname": "node-a",
        "run_id": "run-a",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 100,
        "slurm": {},
        "command": {},
        "probe_config": {},
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {"device": {"accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    outfile = tmp_path / "summary.json"
    parser = generate_aggregate_parser()
    args = parser.parse_args([str(tmp_path), "--fail-on-fail", "-o", str(outfile)])

    handle_aggregate(args)

    summary = json.loads(outfile.read_text(encoding="utf-8"))
    assert summary["pass"]
    assert summary["total_files"] == 1
    assert not summary["load_errors"]


def test_aggregate_ignores_previous_summary_json(tmp_path):
    result = {
        "schema_version": 1,
        "status": "pass",
        "profile": "correctness",
        "hostname": "node-a",
        "run_id": "run-a",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 100,
        "slurm": {},
        "command": {},
        "probe_config": {},
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {"device": {"accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    previous_summary = {
        "input": str(tmp_path),
        "pass": False,
        "status_counts": {"unknown": 1},
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "summary-old.json").write_text(
        json.dumps(previous_summary),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert summary["pass"]
    assert summary["loaded_results"] == 1
    assert summary["ignored_files"] == [str(tmp_path / "summary-old.json")]


def test_aggregate_fails_on_schema_versioned_non_result_json(tmp_path):
    result = {
        "schema_version": 1,
        "status": "pass",
    }
    (tmp_path / "partial-result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"][0]["path"] == str(tmp_path / "partial-result.json")
    assert summary["schema_errors"][0]["type"] == "SchemaError"
    assert summary["schema_errors"][0]["message"].startswith(
        "missing required result keys: "
    )
    assert "command" in summary["schema_errors"][0]["message"]
    assert "probe_config" in summary["schema_errors"][0]["message"]
    assert not summary["ignored_files"]


def test_aggregate_fails_on_unsupported_schema_version(tmp_path):
    result = {
        "schema_version": 999,
        "status": "pass",
        "profile": "correctness",
        "run_id": "run-a",
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {"device": {"accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
    }
    (tmp_path / "future-result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "future-result.json"),
            "type": "SchemaError",
            "message": "unsupported schema_version 999; expected 1",
        }
    ]


def test_aggregate_fails_when_expected_result_count_is_missing(tmp_path):
    result = {
        "schema_version": 1,
        "status": "pass",
        "profile": "correctness",
        "hostname": "node-a",
        "run_id": "run-a",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 100,
        "slurm": {},
        "command": {},
        "probe_config": {},
        "fingerprint_hash": "fingerprint-a",
        "fingerprint": {"device": {"accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path, expected_results=2)

    assert not summary["pass"]
    assert summary["expected_results"] == 2
    assert summary["result_count_failure"] == {
        "expected_results": 2,
        "loaded_results": 1,
        "message": "expected 2 result file(s), loaded 1",
    }


def test_aggregate_parser_rejects_negative_expected_results():
    parser = generate_aggregate_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([".", "--expected-results", "-1"])


def test_aggregate_ignores_valid_non_object_json(tmp_path):
    (tmp_path / "array.json").write_text("[1, 2, 3]", encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["loaded_results"] == 0
    assert summary["ignored_files"] == [str(tmp_path / "array.json")]
    assert not summary["load_errors"]
    assert not summary["pass"]
