import json

import pytest

import narc.files as files_module
from narc.aggregate import aggregate_path, generate_aggregate_parser, handle_aggregate


class FakeS3Body:
    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text.encode("utf-8")


class FakeS3Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": key}
                for bucket, key in sorted(self.objects)
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }


def s3_client_error(code, message="simulated S3 error"):
    return files_module.ClientError(
        {"Error": {"Code": code, "Message": message}},
        "HeadObject",
    )


class FakeS3Client:
    def __init__(self, objects, head_errors=None):
        self.objects = objects
        self.head_errors = head_errors or {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) in self.head_errors:
            raise self.head_errors[(Bucket, Key)]
        if (Bucket, Key) not in self.objects:
            raise s3_client_error("404", "not found")
        return {}

    def get_object(self, Bucket, Key):
        return {"Body": FakeS3Body(self.objects[(Bucket, Key)])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakeS3Paginator(self.objects)

    def put_object(self, Bucket, Key, Body, ContentType):
        assert ContentType == "application/json"
        self.objects[(Bucket, Key)] = Body.decode("utf-8")
        return {}


def install_fake_s3(monkeypatch, objects, head_errors=None):
    client = FakeS3Client(objects, head_errors=head_errors)
    monkeypatch.setattr(files_module.boto3, "client", lambda service: client)
    return client


def valid_result(*, output_hash="hash-a", accelerator_id="GPU-a"):
    return {
        "schema_version": 1,
        "status": "pass",
        "profile": "correctness",
        "hostname": "node-a",
        "run_id": accelerator_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "pid": 100,
        "slurm": {},
        "command": {},
        "probe_config": {},
        "fingerprint_hash": f"fingerprint-{accelerator_id}",
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": accelerator_id}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": output_hash},
        "measurements": {},
        "errors": [],
    }


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
                "type": "cuda",
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
                "type": "cuda",
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
            "fingerprint": {"device": {"type": "cuda", "accelerator_id": name}},
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
                    "type": "cuda",
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
            "fingerprint": {
                "device": {"type": "cuda", "accelerator_id": "GPU-same"}
            },
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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
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


def test_aggregate_writes_s3_outfile(tmp_path, monkeypatch):
    result = valid_result()
    write_path = tmp_path / "result.json"
    write_path.write_text(json.dumps(result), encoding="utf-8")
    objects = {}
    install_fake_s3(monkeypatch, objects)
    parser = generate_aggregate_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--fail-on-fail",
            "-o",
            "s3://fleet-research/narc/summary.json",
        ]
    )

    handle_aggregate(args)

    summary = json.loads(objects[("fleet-research", "narc/summary.json")])
    assert summary["pass"]
    assert summary["loaded_results"] == 1


def test_aggregate_refuses_to_overwrite_s3_probe_result(tmp_path, monkeypatch):
    result = valid_result()
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/result.json"): json.dumps(result)},
    )
    parser = generate_aggregate_parser()
    args = parser.parse_args(
        [str(tmp_path), "-o", "s3://fleet-research/narc/result.json"]
    )

    with pytest.raises(ValueError, match="must not overwrite a probe result file"):
        handle_aggregate(args)


def test_aggregate_refuses_s3_outfile_when_existence_check_fails(
    tmp_path,
    monkeypatch,
):
    result = valid_result()
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    existing = json.dumps(valid_result(output_hash="old-hash"))
    objects = {("fleet-research", "narc/result.json"): existing}
    install_fake_s3(
        monkeypatch,
        objects,
        head_errors={
            ("fleet-research", "narc/result.json"): s3_client_error("AccessDenied")
        },
    )
    parser = generate_aggregate_parser()
    args = parser.parse_args(
        [str(tmp_path), "-o", "s3://fleet-research/narc/result.json"]
    )

    with pytest.raises(files_module.ClientError):
        handle_aggregate(args)

    assert objects[("fleet-research", "narc/result.json")] == existing


def test_aggregate_refuses_to_overwrite_input_result_file(tmp_path):
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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    parser = generate_aggregate_parser()
    args = parser.parse_args([str(result_path), "-o", str(result_path)])

    with pytest.raises(ValueError, match="must not overwrite the input result file"):
        handle_aggregate(args)

    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_aggregate_refuses_to_overwrite_probe_result_in_input_directory(tmp_path):
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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    parser = generate_aggregate_parser()
    args = parser.parse_args([str(tmp_path), "-o", str(result_path)])

    with pytest.raises(ValueError, match="must not overwrite a probe result file"):
        handle_aggregate(args)

    assert json.loads(result_path.read_text(encoding="utf-8")) == result


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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
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


def test_aggregate_rejects_result_without_device_fingerprint(tmp_path):
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
        "fingerprint": {},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "missing-device.json").write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "missing-device.json"),
            "type": "SchemaError",
            "message": "fingerprint.device must be an object",
        }
    ]


def test_aggregate_rejects_cuda_result_without_accelerator_id(tmp_path):
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
        "fingerprint": {"device": {"type": "cuda", "logical_index": 0}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "missing-accelerator.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "missing-accelerator.json"),
            "type": "SchemaError",
            "message": (
                "fingerprint.device.accelerator_id must be a non-empty string "
                "for cuda results"
            ),
        }
    ]


def test_aggregate_rejects_bad_nested_result_types_without_crashing(tmp_path):
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
        "fingerprint": {
            "device": {
                "type": "cuda",
                "accelerator_id": "GPU-a",
                "logical_index": 0,
            }
        },
        "probe_config_hash": "config-a",
        "checks": [],
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "bad-checks.json").write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "bad-checks.json"),
            "type": "SchemaError",
            "message": "checks must be an object",
        }
    ]


def test_aggregate_rejects_bad_output_hash_type_without_crashing(tmp_path):
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
        "fingerprint": {
            "device": {
                "type": "cuda",
                "accelerator_id": "GPU-a",
                "logical_index": 0,
            }
        },
        "probe_config_hash": "config-a",
        "checks": {"output_hash": []},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "bad-output-hash.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "bad-output-hash.json"),
            "type": "SchemaError",
            "message": "checks.output_hash must be a string when present",
        }
    ]


def test_aggregate_rejects_bad_cuda_driver_type_without_crashing(tmp_path):
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
        "fingerprint": {
            "device": {
                "type": "cuda",
                "accelerator_id": "GPU-a",
                "logical_index": 0,
                "cuda_driver": [],
            }
        },
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "bad-cuda-driver.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "bad-cuda-driver.json"),
            "type": "SchemaError",
            "message": "fingerprint.device.cuda_driver must be an object when present",
        }
    ]


def test_aggregate_rejects_bad_cpu_accelerator_id_type_without_crashing(tmp_path):
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
        "fingerprint": {
            "device": {"type": "cpu", "logical_index": 0, "accelerator_id": []}
        },
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "bad-cpu-accelerator.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "bad-cpu-accelerator.json"),
            "type": "SchemaError",
            "message": (
                "fingerprint.device.accelerator_id must be a non-empty string "
                "when present"
            ),
        }
    ]


def test_aggregate_rejects_pass_result_with_errors(tmp_path):
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
        "fingerprint": {"device": {"type": "cpu", "logical_index": 0}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [{"type": "RuntimeError", "message": "boom"}],
    }
    (tmp_path / "pass-with-errors.json").write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(tmp_path / "pass-with-errors.json"),
            "type": "SchemaError",
            "message": "errors must be empty when status is pass",
        }
    ]


def test_aggregate_allows_cpu_result_without_accelerator_id(tmp_path):
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
        "fingerprint": {"device": {"type": "cpu", "logical_index": 0}},
        "probe_config_hash": "config-a",
        "checks": {"output_hash": "hash-a"},
        "measurements": {},
        "errors": [],
    }
    (tmp_path / "cpu-result.json").write_text(json.dumps(result), encoding="utf-8")

    summary = aggregate_path(tmp_path)

    assert summary["pass"]
    assert summary["loaded_results"] == 1
    assert not summary["schema_errors"]
    assert summary["accelerator_ids"] == {"missing": 1}


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
        "fingerprint": {"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
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


def test_aggregate_fails_on_explicit_valid_non_object_json(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    summary = aggregate_path(path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(path),
            "type": "SchemaError",
            "message": "result JSON must be an object",
        }
    ]


def test_aggregate_fails_on_explicit_schema_less_object_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"pass": True}), encoding="utf-8")

    summary = aggregate_path(path)

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["schema_errors"] == [
        {
            "path": str(path),
            "type": "SchemaError",
            "message": "unsupported schema_version None; expected 1",
        }
    ]


def test_load_result_reads_s3_object(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/a.json"): json.dumps(result)},
    )

    loaded, error = files_module.load_result("s3://fleet-research/narc/a.json")

    assert error is None
    assert loaded == result


def test_load_result_preserves_s3_key_reserved_characters(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/result#1?.json"): json.dumps(result)},
    )

    loaded, error = files_module.load_result(
        "s3://fleet-research/narc/result#1?.json"
    )

    assert error is None
    assert loaded == result


def test_load_result_accepts_uppercase_s3_scheme(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/a.json"): json.dumps(result)},
    )

    loaded, error = files_module.load_result("S3://fleet-research/narc/a.json")

    assert error is None
    assert loaded == result


def test_aggregate_path_lists_s3_prefix(monkeypatch):
    first = valid_result(output_hash="hash-a", accelerator_id="GPU-a")
    second = valid_result(output_hash="hash-a", accelerator_id="GPU-b")
    install_fake_s3(
        monkeypatch,
        {
            ("fleet-research", "narc/a.json"): json.dumps(first),
            ("fleet-research", "narc/b.json"): json.dumps(second),
            ("fleet-research", "narc/c.json.tmp"): json.dumps(second),
            ("fleet-research", "narc/readme.txt"): "not json",
        },
    )

    summary = aggregate_path("s3://fleet-research/narc/")

    assert summary["pass"]
    assert summary["input"] == "s3://fleet-research/narc/"
    assert summary["total_files"] == 2
    assert summary["loaded_results"] == 2
    assert summary["devices"] == [
        {
            "path": "s3://fleet-research/narc/a.json",
            "status": "pass",
            "hostname": "node-a",
            "accelerator_id": "GPU-a",
            "logical_index": None,
            "cuda_uuid": None,
            "pci_bus_id": None,
            "slurm_procid": None,
            "slurm_localid": None,
        },
        {
            "path": "s3://fleet-research/narc/b.json",
            "status": "pass",
            "hostname": "node-a",
            "accelerator_id": "GPU-b",
            "logical_index": None,
            "cuda_uuid": None,
            "pci_bus_id": None,
            "slurm_procid": None,
            "slurm_localid": None,
        },
    ]


def test_aggregate_path_s3_prefix_without_slash_excludes_siblings(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {
            ("fleet-research", "narc/a.json"): json.dumps(result),
            ("fleet-research", "narc-extra.json"): json.dumps(result),
        },
    )

    summary = aggregate_path("s3://fleet-research/narc")

    assert summary["pass"]
    assert summary["total_files"] == 1
    assert summary["devices"][0]["path"] == "s3://fleet-research/narc/a.json"


def test_aggregate_path_reads_explicit_s3_object_without_json_suffix(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/result"): json.dumps(result)},
    )

    summary = aggregate_path("s3://fleet-research/narc/result")

    assert summary["pass"]
    assert summary["total_files"] == 1
    assert summary["loaded_results"] == 1
    assert summary["devices"][0]["path"] == "s3://fleet-research/narc/result"


def test_aggregate_path_lists_s3_prefix_that_ends_with_json(monkeypatch):
    result = valid_result()
    install_fake_s3(
        monkeypatch,
        {("fleet-research", "narc/job.json/a.json"): json.dumps(result)},
    )

    summary = aggregate_path("s3://fleet-research/narc/job.json")

    assert summary["pass"]
    assert summary["total_files"] == 1
    assert summary["devices"][0]["path"] == "s3://fleet-research/narc/job.json/a.json"


def test_aggregate_reports_s3_load_error(monkeypatch):
    install_fake_s3(monkeypatch, {})

    summary = aggregate_path("s3://fleet-research/missing.json")

    assert not summary["pass"]
    assert summary["loaded_results"] == 0
    assert summary["load_errors"][0]["path"] == "s3://fleet-research/missing.json"
    assert summary["load_errors"][0]["type"] == "KeyError"
