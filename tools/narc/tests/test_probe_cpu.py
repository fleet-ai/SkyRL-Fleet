import pytest

from narc.cli import generate_cli
from narc.probe import run_probe


def test_cpu_correctness_probe_is_repeatable():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run-local",
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
            "run-local",
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
            "run-local",
            "--device",
            "cpu",
            "--batch-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        run_probe(args)
