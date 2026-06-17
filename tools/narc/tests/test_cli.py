from narc.cli import generate_cli


def test_root_parser_defaults_to_help_function():
    parser = generate_cli()
    args = parser.parse_args([])

    assert callable(args.func)


def test_run_local_parser_shape():
    parser = generate_cli()
    args = parser.parse_args(
        [
            "run-local",
            "--device",
            "cpu",
            "--profile",
            "correctness",
            "--steps",
            "1",
        ]
    )

    assert args.device == "cpu"
    assert args.profile == "correctness"
    assert args.steps == 1
    assert callable(args.func)
