from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from narc.schema import SCHEMA_VERSION


REQUIRED_RESULT_KEYS = {
    "schema_version",
    "status",
    "profile",
    "run_id",
    "probe_config_hash",
    "fingerprint_hash",
    "fingerprint",
    "checks",
    "measurements",
}


def iter_json_paths(path: Path, exclude_paths: set[Path] | None = None) -> list[Path]:
    excluded = exclude_paths or set()
    if path.is_file():
        if path.resolve() in excluded:
            return []
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*.json")
        if not candidate.name.endswith(".tmp")
        and candidate.resolve() not in excluded
    )


def load_result(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception as error:
        return None, {
            "path": str(path),
            "type": type(error).__name__,
            "message": str(error),
        }
    if not isinstance(document, dict):
        return None, None
    return document, None


def is_probe_result(document: dict[str, Any]) -> bool:
    if document.get("schema_version") != SCHEMA_VERSION:
        return False
    return REQUIRED_RESULT_KEYS.issubset(document.keys())


def performance_values(results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for result in results:
        timing = result.get("measurements", {}).get("timing", {})
        value = timing.get("tokens_per_second")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize_performance(values: list[float]) -> dict[str, float | int | None]:
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


def device_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    device = result.get("fingerprint", {}).get("device", {})
    cuda_driver = device.get("cuda_driver", {})
    slurm = result.get("slurm", {})
    return {
        "path": result.get("source_path"),
        "status": result.get("status"),
        "hostname": result.get("hostname"),
        "accelerator_id": device.get("accelerator_id"),
        "logical_index": device.get("logical_index"),
        "cuda_uuid": cuda_driver.get("uuid"),
        "pci_bus_id": cuda_driver.get("pci_bus_id"),
        "slurm_procid": slurm.get("slurm_procid"),
        "slurm_localid": slurm.get("slurm_localid"),
    }


def comparable_output_groups(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for result in results:
        config_hash = result.get("probe_config_hash") or "missing"
        profile = result.get("profile") or "missing"
        key = f"{profile}:{config_hash}"
        output_hash = result.get("checks", {}).get("output_hash")
        source_path = result.get("source_path")
        group = groups.setdefault(
            key,
            {
                "profile": profile,
                "probe_config_hash": config_hash,
                "output_hashes": {},
                "missing_output_hash": [],
            },
        )
        if output_hash:
            hashes = group["output_hashes"]
            hashes[output_hash] = hashes.get(output_hash, 0) + 1
        else:
            group["missing_output_hash"].append(source_path)
    return groups


def output_hash_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key, group in comparable_output_groups(results).items():
        output_hashes = group["output_hashes"]
        missing = group["missing_output_hash"]
        if len(output_hashes) > 1 or missing:
            failures.append(
                {
                    "group": key,
                    "profile": group["profile"],
                    "probe_config_hash": group["probe_config_hash"],
                    "output_hashes": output_hashes,
                    "missing_output_hash": missing,
                }
            )
    return failures


def accelerator_entry(result: dict[str, Any]) -> dict[str, Any]:
    device = result.get("fingerprint", {}).get("device", {})
    slurm = result.get("slurm", {})
    return {
        "path": result.get("source_path"),
        "hostname": result.get("hostname"),
        "run_id": result.get("run_id"),
        "logical_index": device.get("logical_index"),
        "slurm_procid": slurm.get("slurm_procid"),
        "slurm_localid": slurm.get("slurm_localid"),
    }


def comparable_accelerator_groups(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for result in results:
        config_hash = result.get("probe_config_hash") or "missing"
        profile = result.get("profile") or "missing"
        key = f"{profile}:{config_hash}"
        accelerator_id = (
            result.get("fingerprint", {}).get("device", {}).get("accelerator_id")
        )
        group = groups.setdefault(
            key,
            {
                "profile": profile,
                "probe_config_hash": config_hash,
                "accelerators": {},
            },
        )
        if accelerator_id:
            accelerators = group["accelerators"]
            entries = accelerators.setdefault(accelerator_id, [])
            entries.append(accelerator_entry(result))
    return groups


def duplicate_accelerator_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key, group in comparable_accelerator_groups(results).items():
        duplicates = {
            accelerator_id: entries
            for accelerator_id, entries in group["accelerators"].items()
            if len(entries) > 1
        }
        if duplicates:
            failures.append(
                {
                    "group": key,
                    "profile": group["profile"],
                    "probe_config_hash": group["probe_config_hash"],
                    "duplicates": duplicates,
                }
            )
    return failures


def aggregate_path(
    path: Path,
    *,
    exclude_paths: set[Path] | None = None,
) -> dict[str, Any]:
    json_paths = iter_json_paths(path, exclude_paths=exclude_paths)
    loaded: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    ignored_files: list[str] = []
    for json_path in json_paths:
        result, error = load_result(json_path)
        if result is None:
            if error is not None:
                load_errors.append(error)
            else:
                ignored_files.append(str(json_path))
            continue
        if not is_probe_result(result):
            ignored_files.append(str(json_path))
            continue
        result["source_path"] = str(json_path)
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
    accelerator_ids: dict[str, int] = {}
    for result in loaded:
        fingerprint_hash = result.get("fingerprint_hash") or "missing"
        config_hash = result.get("probe_config_hash") or "missing"
        output_hash = result.get("checks", {}).get("output_hash") or "missing"
        accelerator_id = (
            result.get("fingerprint", {}).get("device", {}).get("accelerator_id")
            or "missing"
        )
        by_fingerprint[fingerprint_hash] = by_fingerprint.get(fingerprint_hash, 0) + 1
        by_config[config_hash] = by_config.get(config_hash, 0) + 1
        output_hashes[output_hash] = output_hashes.get(output_hash, 0) + 1
        accelerator_ids[accelerator_id] = accelerator_ids.get(accelerator_id, 0) + 1

    failures = [
        {
            "path": result.get("source_path"),
            "hostname": result.get("hostname"),
            "accelerator_id": device_fingerprint(result).get("accelerator_id"),
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "errors": result.get("errors", []),
            "checks": result.get("checks", {}),
        }
        for result in loaded
        if result.get("status") != "pass"
    ]
    hash_failures = output_hash_failures(loaded)
    accelerator_failures = duplicate_accelerator_failures(loaded)

    return {
        "input": str(path),
        "total_files": len(json_paths),
        "loaded_results": len(loaded),
        "load_errors": load_errors,
        "ignored_files": ignored_files,
        "status_counts": status_counts,
        "fingerprint_hashes": by_fingerprint,
        "probe_config_hashes": by_config,
        "output_hashes": output_hashes,
        "output_hash_failures": hash_failures,
        "accelerator_ids": accelerator_ids,
        "duplicate_accelerator_failures": accelerator_failures,
        "devices": [device_fingerprint(result) for result in loaded],
        "performance": {
            "tokens_per_second": summarize_performance(performance_values(loaded)),
        },
        "failures": failures,
        "pass": (
            status_counts["fail"] == 0
            and status_counts["warn"] == 0
            and status_counts["unknown"] == 0
            and not load_errors
            and not hash_failures
            and not accelerator_failures
            and len(loaded) > 0
        ),
    }


def handle_aggregate(args: argparse.Namespace) -> None:
    output_path = Path(args.outfile).resolve() if args.outfile else None
    exclude_paths = {output_path} if output_path else None
    summary = aggregate_path(Path(args.path), exclude_paths=exclude_paths)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True)
            outfile.write("\n")
    else:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
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
        type=str,
        help="Output file for aggregate JSON.",
    )
    parser.set_defaults(func=handle_aggregate)
    return parser
