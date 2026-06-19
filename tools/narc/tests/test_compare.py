import copy
import json

import pytest

import narc.files as files_module
from narc.cli import generate_cli
from narc.compare import compare, compare_paths, handle_compare
from narc.schema import SCHEMA_VERSION


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


def result_payload(
    *,
    status: str = "pass",
    output_hash: str | None = "hash-a",
    input_hash: str | None = "input-a",
    errors: list[dict[str, str]] | None = None,
    checks: dict[str, object] | None = None,
    accelerator_id: str = "GPU-a",
) -> dict[str, object]:
    result_checks = {"input_hash": input_hash, "output_hash": output_hash}
    if checks:
        result_checks.update(checks)
    if output_hash is None:
        result_checks.pop("output_hash")
    if input_hash is None:
        result_checks.pop("input_hash")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
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
        "fingerprint": {
            "device": {
                "type": "cuda",
                "accelerator_id": accelerator_id,
                "logical_index": 0,
            }
        },
        "probe_config_hash": "config-a",
        "checks": result_checks,
        "measurements": {},
        "errors": errors or [],
    }


def write_result(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_groups_matching_pass_results():
    first = result_payload(accelerator_id="GPU-a")
    second = result_payload(accelerator_id="GPU-b")

    partitions = compare([first, second])

    assert partitions == [[first, second]]


def test_compare_splits_pass_results_by_output_hash():
    first = result_payload(output_hash="hash-a", accelerator_id="GPU-a")
    second = result_payload(output_hash="hash-b", accelerator_id="GPU-b")

    partitions = compare([first, second])

    assert partitions == [[first], [second]]


def test_compare_splits_pass_results_by_input_hash():
    first = result_payload(input_hash="input-a", accelerator_id="GPU-a")
    second = result_payload(input_hash="input-b", accelerator_id="GPU-b")

    partitions = compare([first, second])

    assert partitions == [[first], [second]]


def test_compare_groups_failed_results_by_error_shape():
    first = result_payload(
        status="fail",
        output_hash=None,
        errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
        accelerator_id="GPU-a",
    )
    second = result_payload(
        status="fail",
        output_hash=None,
        errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
        accelerator_id="GPU-b",
    )
    third = result_payload(
        status="fail",
        output_hash=None,
        errors=[{"type": "RuntimeError", "message": "illegal memory access"}],
        accelerator_id="GPU-c",
    )

    partitions = compare([first, second, third])

    assert partitions == [[first, second], [third]]


def test_compare_splits_failed_results_by_check_shape():
    first = result_payload(
        status="fail",
        errors=[],
        checks={"repeat_match": False},
        accelerator_id="GPU-a",
    )
    second = copy.deepcopy(first)
    second["run_id"] = "GPU-b"
    second["fingerprint"]["device"]["accelerator_id"] = "GPU-b"
    second["checks"]["repeat_match"] = True
    second["checks"]["expected_hash_match"] = False

    partitions = compare([first, second])

    assert partitions == [[first], [second]]


def test_compare_paths_reports_partitions(tmp_path):
    first = result_payload(accelerator_id="GPU-a")
    second = result_payload(accelerator_id="GPU-b")
    write_result(tmp_path / "a.json", first)
    write_result(tmp_path / "b.json", second)

    report = compare_paths([tmp_path])

    assert report["pass"]
    assert report["loaded_results"] == 2
    assert report["status_counts"] == {"pass": 2, "warn": 0, "fail": 0}
    assert report["partition_count"] == 1
    assert report["partitions"][0]["size"] == 2
    assert report["partitions"][0]["equivalence"] == {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "profile": "correctness",
        "probe_config_hash": "config-a",
        "input_hash": "input-a",
        "output_hash": "hash-a",
    }


def test_compare_paths_deduplicates_equivalent_local_paths(tmp_path, monkeypatch):
    write_result(tmp_path / "a.json", result_payload())
    monkeypatch.chdir(tmp_path)

    report = compare_paths([tmp_path / "a.json", "a.json"])

    assert report["pass"]
    assert report["total_files"] == 1
    assert report["loaded_results"] == 1


def test_compare_paths_reports_schema_errors(tmp_path):
    write_result(tmp_path / "bad.json", {"schema_version": 1, "status": "pass"})

    report = compare_paths([tmp_path / "bad.json"])

    assert not report["pass"]
    assert report["loaded_results"] == 0
    assert report["schema_errors"][0]["type"] == "SchemaError"


def test_compare_paths_fails_on_explicit_schema_less_object_json(tmp_path):
    first = result_payload(accelerator_id="GPU-a")
    second = result_payload(accelerator_id="GPU-b")
    write_result(tmp_path / "a.json", first)
    write_result(tmp_path / "b.json", second)
    malformed = tmp_path / "not-a-result.json"
    write_result(malformed, {"pass": True})

    report = compare_paths([tmp_path / "a.json", tmp_path / "b.json", malformed])

    assert not report["pass"]
    assert report["loaded_results"] == 2
    assert report["schema_errors"] == [
        {
            "path": str(malformed),
            "type": "SchemaError",
            "message": f"unsupported schema_version None; expected {SCHEMA_VERSION}",
        }
    ]


def test_compare_paths_ignores_scanned_schema_less_object_json(tmp_path):
    write_result(tmp_path / "a.json", result_payload(accelerator_id="GPU-a"))
    write_result(tmp_path / "b.json", result_payload(accelerator_id="GPU-b"))
    write_result(tmp_path / "compare-old.json", {"compare_schema_version": 1})

    report = compare_paths([tmp_path])

    assert report["pass"]
    assert report["ignored_files"] == [str(tmp_path / "compare-old.json")]


def test_compare_paths_fails_when_equivalent_results_are_failed(tmp_path):
    first = result_payload(
        status="fail",
        output_hash=None,
        errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
        accelerator_id="GPU-a",
    )
    second = result_payload(
        status="fail",
        output_hash=None,
        errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
        accelerator_id="GPU-b",
    )
    write_result(tmp_path / "a.json", first)
    write_result(tmp_path / "b.json", second)

    report = compare_paths([tmp_path])

    assert not report["pass"]
    assert not report["split"]
    assert report["status_counts"] == {"pass": 0, "warn": 0, "fail": 2}
    assert report["partition_count"] == 1


def test_compare_paths_fails_when_pass_result_has_no_output_hash(tmp_path):
    write_result(tmp_path / "a.json", result_payload(output_hash=None))
    write_result(tmp_path / "b.json", result_payload(output_hash=None))

    report = compare_paths([tmp_path])

    assert not report["pass"]
    assert not report["split"]
    assert report["missing_output_hash"] == [
        str(tmp_path / "a.json"),
        str(tmp_path / "b.json"),
    ]


def test_compare_paths_fails_when_pass_result_has_empty_output_hash(tmp_path):
    write_result(tmp_path / "a.json", result_payload(output_hash=""))

    report = compare_paths([tmp_path])

    assert not report["pass"]
    assert not report["split"]
    assert report["missing_output_hash"] == [str(tmp_path / "a.json")]


def test_compare_paths_fails_when_pass_result_has_no_input_hash(tmp_path):
    write_result(tmp_path / "a.json", result_payload(input_hash=None))

    report = compare_paths([tmp_path])

    assert not report["pass"]
    assert not report["split"]
    assert report["missing_input_hash"] == [str(tmp_path / "a.json")]


def test_compare_paths_rejects_bad_input_hash_type(tmp_path):
    result = result_payload()
    result["checks"]["input_hash"] = []
    write_result(tmp_path / "bad-input-hash.json", result)

    report = compare_paths([tmp_path])

    assert not report["pass"]
    assert report["loaded_results"] == 0
    assert report["schema_errors"] == [
        {
            "path": str(tmp_path / "bad-input-hash.json"),
            "type": "SchemaError",
            "message": "checks.input_hash must be a string when present",
        }
    ]


def test_compare_cli_fails_on_split_by_default(tmp_path):
    write_result(tmp_path / "a.json", result_payload(output_hash="hash-a"))
    write_result(tmp_path / "b.json", result_payload(output_hash="hash-b"))
    outfile = tmp_path / "compare.json"
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "-o",
            str(outfile),
        ]
    )

    with pytest.raises(SystemExit) as exc:
        handle_compare(args)

    assert exc.value.code == 1
    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert report["split"]
    assert not report["pass"]
    assert report["partition_count"] == 2


def test_compare_cli_fails_on_equivalent_failed_results_by_default(tmp_path):
    write_result(
        tmp_path / "a.json",
        result_payload(
            status="fail",
            output_hash=None,
            errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
            accelerator_id="GPU-a",
        ),
    )
    write_result(
        tmp_path / "b.json",
        result_payload(
            status="fail",
            output_hash=None,
            errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
            accelerator_id="GPU-b",
        ),
    )
    outfile = tmp_path / "compare.json"
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "-o",
            str(outfile),
        ]
    )

    with pytest.raises(SystemExit) as exc:
        handle_compare(args)

    assert exc.value.code == 1
    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert not report["split"]
    assert not report["pass"]


def test_compare_cli_fails_on_missing_output_hash_by_default(tmp_path):
    write_result(tmp_path / "a.json", result_payload(output_hash=None))
    outfile = tmp_path / "compare.json"
    parser = generate_cli()
    args = parser.parse_args(["compare", str(tmp_path / "a.json"), "-o", str(outfile)])

    with pytest.raises(SystemExit) as exc:
        handle_compare(args)

    assert exc.value.code == 1
    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert not report["pass"]
    assert report["missing_output_hash"] == [str(tmp_path / "a.json")]


def test_compare_cli_nofail_allows_split(tmp_path):
    write_result(tmp_path / "a.json", result_payload(output_hash="hash-a"))
    write_result(tmp_path / "b.json", result_payload(output_hash="hash-b"))
    outfile = tmp_path / "compare.json"
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            "--nofail",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "-o",
            str(outfile),
        ]
    )

    handle_compare(args)

    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert report["split"]
    assert not report["pass"]


def test_compare_cli_nofail_allows_equivalent_failed_results(tmp_path):
    write_result(
        tmp_path / "a.json",
        result_payload(
            status="fail",
            output_hash=None,
            errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
            accelerator_id="GPU-a",
        ),
    )
    write_result(
        tmp_path / "b.json",
        result_payload(
            status="fail",
            output_hash=None,
            errors=[{"type": "RuntimeError", "message": "CUDA out of memory"}],
            accelerator_id="GPU-b",
        ),
    )
    outfile = tmp_path / "compare.json"
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            "--nofail",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "-o",
            str(outfile),
        ]
    )

    handle_compare(args)

    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert not report["split"]
    assert not report["pass"]


def test_compare_cli_refuses_to_overwrite_corrupt_input_directory_json(tmp_path):
    write_result(tmp_path / "good.json", result_payload())
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{", encoding="utf-8")
    parser = generate_cli()
    args = parser.parse_args(["compare", str(tmp_path), "-o", str(corrupt)])

    with pytest.raises(ValueError, match="compare inputs"):
        handle_compare(args)

    assert corrupt.read_text(encoding="utf-8") == "{"


def test_compare_cli_allows_overwriting_previous_compare_report(tmp_path):
    write_result(tmp_path / "good.json", result_payload())
    outfile = tmp_path / "compare.json"
    write_result(outfile, {"compare_schema_version": 1})
    parser = generate_cli()
    args = parser.parse_args(["compare", str(tmp_path), "-o", str(outfile)])

    handle_compare(args)

    report = json.loads(outfile.read_text(encoding="utf-8"))
    assert report["pass"]
    assert report["loaded_results"] == 1


def test_compare_cli_writes_s3_outfile(tmp_path, monkeypatch):
    write_result(tmp_path / "a.json", result_payload(accelerator_id="GPU-a"))
    write_result(tmp_path / "b.json", result_payload(accelerator_id="GPU-b"))
    objects = {}
    install_fake_s3(monkeypatch, objects)
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            str(tmp_path),
            "-o",
            "s3://fleet-research/narc/compare.json",
        ]
    )

    handle_compare(args)

    report = json.loads(objects[("fleet-research", "narc/compare.json")])
    assert report["pass"]
    assert report["loaded_results"] == 2


def test_compare_cli_allows_overwriting_previous_s3_compare_report(monkeypatch):
    result = result_payload()
    objects = {
        ("fleet-research", "narc/result.json"): json.dumps(result),
        ("fleet-research", "narc/compare.json"): json.dumps(
            {"compare_schema_version": 1}
        ),
    }
    install_fake_s3(monkeypatch, objects)
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            "s3://fleet-research/narc/",
            "-o",
            "s3://fleet-research/narc/compare.json",
        ]
    )

    handle_compare(args)

    report = json.loads(objects[("fleet-research", "narc/compare.json")])
    assert report["pass"]
    assert report["loaded_results"] == 1
    assert report["total_files"] == 1


def test_compare_cli_refuses_to_overwrite_s3_input_result(monkeypatch):
    result = result_payload()
    objects = {("fleet-research", "narc/result.json"): json.dumps(result)}
    install_fake_s3(monkeypatch, objects)
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            "s3://fleet-research/narc/",
            "-o",
            "s3://fleet-research/narc/result.json",
        ]
    )

    with pytest.raises(ValueError, match="probe result file"):
        handle_compare(args)


def test_compare_refuses_s3_outfile_when_existence_check_fails(
    tmp_path,
    monkeypatch,
):
    write_result(tmp_path / "a.json", result_payload(accelerator_id="GPU-a"))
    existing = json.dumps(result_payload(output_hash="old-hash"))
    objects = {("fleet-research", "narc/result.json"): existing}
    install_fake_s3(
        monkeypatch,
        objects,
        head_errors={
            ("fleet-research", "narc/result.json"): s3_client_error("AccessDenied")
        },
    )
    parser = generate_cli()
    args = parser.parse_args(
        [
            "compare",
            str(tmp_path),
            "-o",
            "s3://fleet-research/narc/result.json",
        ]
    )

    with pytest.raises(files_module.ClientError):
        handle_compare(args)

    assert objects[("fleet-research", "narc/result.json")] == existing


def test_compare_paths_reads_s3_prefix(monkeypatch):
    first = result_payload(accelerator_id="GPU-a")
    second = result_payload(accelerator_id="GPU-b")
    install_fake_s3(
        monkeypatch,
        {
            ("fleet-research", "narc/a.json"): json.dumps(first),
            ("fleet-research", "narc/b.json"): json.dumps(second),
            ("fleet-research", "narc/ignored.json.tmp"): json.dumps(first),
        },
    )

    report = compare_paths(["s3://fleet-research/narc/"])

    assert report["pass"]
    assert report["inputs"] == ["s3://fleet-research/narc/"]
    assert report["total_files"] == 2
    assert report["loaded_results"] == 2
    assert report["partitions"][0]["results"][0]["path"] == (
        "s3://fleet-research/narc/a.json"
    )
    assert report["partitions"][0]["results"][1]["path"] == (
        "s3://fleet-research/narc/b.json"
    )
