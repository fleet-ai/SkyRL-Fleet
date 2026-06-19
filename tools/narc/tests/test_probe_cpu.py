import pytest

import narc.probe as probe_module
from narc.cli import generate_cli
from narc.probe import (
    default_output_location,
    default_output_path,
    output_device_id,
    run_probe,
)
from narc.schema import ProbeResult


def test_cpu_correctness_probe_is_repeatable():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run",
            "--device",
            "cpu",
            "--profile",
            "correctness",
            "--dtype",
            "fp32",
            "--repeat",
            "2",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--sequence-length",
            "8",
            "--vocab-size",
            "64",
            "--d-model",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--mlp-ratio",
            "2",
        ]
    )

    result = run_probe(args)
    payload = result.to_dict()

    assert payload["status"] == "pass"
    assert payload["checks"]["repeat_match"] is True
    assert payload["checks"]["output_hash"]
    assert payload["fingerprint_hash"]


def test_cpu_probe_rejects_sequence_length_without_targets():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run",
            "--device",
            "cpu",
            "--sequence-length",
            "1",
        ]
    )

    with pytest.raises(ValueError, match="sequence_length must be at least 2"):
        run_probe(args)


def test_cpu_probe_rejects_zero_overrides_instead_of_defaulting():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run",
            "--device",
            "cpu",
            "--batch-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        run_probe(args)


def test_default_output_path_sanitizes_user_controlled_components(tmp_path):
    result = ProbeResult(
        schema_version=1,
        status="pass",
        profile="correctness",
        run_id="../../escape",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        hostname="node/../../x",
        pid=123,
        slurm={"slurm_procid": "../rank", "slurm_localid": "0/1"},
        command={},
        probe_config={},
        probe_config_hash="config-a",
        fingerprint={"device": {"accelerator_id": "GPU-a"}},
        fingerprint_hash="fingerprint-a",
        checks={"output_hash": "hash-a"},
        measurements={},
        errors=[],
    )

    output_path = default_output_path(result, tmp_path)

    assert output_path.parent == tmp_path
    assert output_path.name.startswith("GPU-a-")
    assert ".." not in output_path.name
    assert "/" not in output_path.name
    assert output_path.name.endswith(".json")


def test_output_device_id_falls_back_for_cpu_result(tmp_path):
    result = ProbeResult(
        schema_version=1,
        status="pass",
        profile="correctness",
        run_id="run-a",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        hostname="node-a",
        pid=123,
        slurm={},
        command={},
        probe_config={},
        probe_config_hash="config-a",
        fingerprint={"device": {"type": "cpu", "logical_index": 0}},
        fingerprint_hash="fingerprint-a",
        checks={"output_hash": "hash-a"},
        measurements={},
        errors=[],
    )

    assert output_device_id(result) == "cpu-0"
    assert default_output_path(result, tmp_path).name.startswith("cpu-0-")


def test_default_output_location_supports_s3_prefix():
    result = ProbeResult(
        schema_version=1,
        status="pass",
        profile="correctness",
        run_id="run-a",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        hostname="node-a",
        pid=123,
        slurm={"slurm_procid": "7", "slurm_localid": "3"},
        command={},
        probe_config={},
        probe_config_hash="config-a",
        fingerprint={"device": {"type": "cuda", "accelerator_id": "GPU-a"}},
        fingerprint_hash="fingerprint-a",
        checks={"output_hash": "hash-a"},
        measurements={},
        errors=[],
    )

    output_location = default_output_location(
        result,
        "s3://fleet-research/narc/job-1/",
    )

    assert output_location == (
        "s3://fleet-research/narc/job-1/GPU-a-rank7-local3-pid123-run-a.json"
    )


def test_run_rejects_directory_outfile_before_probe(tmp_path, monkeypatch):
    parser = generate_cli()
    args = parser.parse_args(["run", "--device", "cpu", "-o", str(tmp_path)])
    monkeypatch.setattr(
        probe_module,
        "run_probe",
        lambda parsed: pytest.fail("run_probe should not be called"),
    )

    with pytest.raises(ValueError, match="outfile must not be an existing directory"):
        args.func(args)


def test_run_rejects_s3_bucket_outfile_before_probe(monkeypatch):
    parser = generate_cli()
    args = parser.parse_args(["run", "--device", "cpu", "-o", "s3://fleet-research"])
    monkeypatch.setattr(
        probe_module,
        "run_probe",
        lambda parsed: pytest.fail("run_probe should not be called"),
    )

    with pytest.raises(ValueError, match="S3 URI must include an object key"):
        args.func(args)
