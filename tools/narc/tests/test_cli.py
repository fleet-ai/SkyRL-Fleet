import pytest

from narc.cli import generate_cli


def test_root_parser_defaults_to_help_function():
    parser = generate_cli()
    args = parser.parse_args([])

    assert callable(args.func)


def test_run_parser_shape():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run",
            "--device",
            "cpu",
            "--profile",
            "correctness",
            "--steps",
            "1",
            "--input-seed",
            "5678",
        ]
    )

    assert args.device == "cpu"
    assert args.profile == "correctness"
    assert args.steps == 1
    assert args.input_seed == 5678
    assert callable(args.func)


def test_run_rejects_negative_logical_device():
    parser = generate_cli()

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--logical-device", "-1"])
