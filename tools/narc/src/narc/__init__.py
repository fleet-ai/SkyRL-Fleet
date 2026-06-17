import argparse
import sys
from collections.abc import Sequence

from narc.cli import generate_cli


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = generate_cli()
    args = parser.parse_args(args_list)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        if getattr(args, "debug", False):
            raise
        parser.exit(1, f"narc: error: {error}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = generate_cli()
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


__all__ = ["main", "parse_args"]
