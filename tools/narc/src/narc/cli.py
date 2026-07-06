import argparse
from importlib.metadata import PackageNotFoundError, version

from narc.compare import generate_compare_parser
from narc.probe import generate_run_parser


def package_version() -> str:
    try:
        return version("narc")
    except PackageNotFoundError:
        return "0.0.0+local"


def generate_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("narc: deterministic accelerator probes for compute clusters"))
    parser.add_argument(
        "--version",
        action="version",
        version=package_version(),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show tracebacks instead of compact CLI errors.",
    )
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    run_parser = generate_run_parser()
    subparsers.add_parser(
        "run",
        parents=[run_parser],
        add_help=False,
        help=run_parser.description,
    )

    compare_parser = generate_compare_parser()
    subparsers.add_parser(
        "compare",
        parents=[compare_parser],
        add_help=False,
        help=compare_parser.description,
    )

    return parser
