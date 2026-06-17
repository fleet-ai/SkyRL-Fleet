from __future__ import annotations

import argparse
import json
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _iter_json_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*.json")
        if not candidate.name.endswith(".tmp")
    )


def _load_result(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as error:
        return None, {
            "path": str(path),
            "type": type(error).__name__,
            "message": str(error),
        }


def _performance_values(results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for result in results:
        timing = result.get("measurements", {}).get("timing", {})
        value = timing.get("tokens_per_second")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _summarize_performance(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def aggregate_path(path: Path) -> dict[str, Any]:
    json_paths = _iter_json_paths(path)
    loaded: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for json_path in json_paths:
        result, error = _load_result(json_path)
        if result is None:
            if error is not None:
                load_errors.append(error)
            continue
        result["_source_path"] = str(json_path)
        loaded.append(result)

    status_counts = {"pass": 0, "warn": 0, "fail": 0, "unknown": 0}
    for result in loaded:
        status = result.get("status", "unknown")
        if status not in status_counts:
            status = "unknown"
        status_counts[status] += 1

    by_fingerprint: dict[str, int] = {}
    by_config: dict[str, int] = {}
    output_hashes: dict[str, int] = {}
    for result in loaded:
        fingerprint_hash = result.get("fingerprint_hash") or "missing"
        config_hash = result.get("probe_config_hash") or "missing"
        output_hash = result.get("checks", {}).get("output_hash") or "missing"
        by_fingerprint[fingerprint_hash] = by_fingerprint.get(fingerprint_hash, 0) + 1
        by_config[config_hash] = by_config.get(config_hash, 0) + 1
        output_hashes[output_hash] = output_hashes.get(output_hash, 0) + 1

    failures = [
        {
            "path": result.get("_source_path"),
            "hostname": result.get("hostname"),
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "errors": result.get("errors", []),
            "checks": result.get("checks", {}),
        }
        for result in loaded
        if result.get("status") != "pass"
    ]

    return {
        "input": str(path),
        "total_files": len(json_paths),
        "loaded_results": len(loaded),
        "load_errors": load_errors,
        "status_counts": status_counts,
        "fingerprint_hashes": by_fingerprint,
        "probe_config_hashes": by_config,
        "output_hashes": output_hashes,
        "performance": {
            "tokens_per_second": _summarize_performance(_performance_values(loaded)),
        },
        "failures": failures,
        "pass": (
            status_counts["fail"] == 0
            and status_counts["unknown"] == 0
            and not load_errors
            and len(loaded) > 0
        ),
    }


def handle_aggregate(args: argparse.Namespace) -> None:
    summary = aggregate_path(Path(args.path))
    context = nullcontext(args.outfile) if args.outfile is sys.stdout else args.outfile
    with context as outfile:
        json.dump(summary, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    if args.fail_on_fail and not summary["pass"]:
        raise SystemExit(1)


def generate_aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate narc per-device JSON results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Result JSON file or directory.")
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit non-zero when aggregation finds failures or load errors.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file for aggregate JSON.",
    )
    parser.set_defaults(func=handle_aggregate)
    return parser
