from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any

from narc.files import (
    ResultLocation,
    explicit_input_identities,
    iter_json_paths,
    json_report_text,
    load_result,
    location_identity,
    location_text,
    validate_outfile,
    validate_output_path,
    write_json_report,
)
from narc.schema import SCHEMA_VERSION


REQUIRED_RESULT_KEYS = {
    "checks",
    "command",
    "errors",
    "fingerprint",
    "fingerprint_hash",
    "finished_at",
    "hostname",
    "measurements",
    "pid",
    "probe_config",
    "probe_config_hash",
    "profile",
    "run_id",
    "schema_version",
    "slurm",
    "started_at",
    "status",
}

RESULT_FIELD_TYPES = {
    "schema_version": int,
    "narc_data_version": int,
    "status": str,
    "profile": str,
    "run_id": str,
    "started_at": str,
    "finished_at": str,
    "hostname": str,
    "pid": int,
    "slurm": dict,
    "command": dict,
    "probe_config": dict,
    "probe_config_hash": str,
    "fingerprint": dict,
    "fingerprint_hash": str,
    "checks": dict,
    "measurements": dict,
    "errors": list,
}

VALID_PROFILES = {"correctness", "performance"}
VALID_STATUSES = {"pass", "warn", "fail"}


def type_description(expected_type: type[Any]) -> str:
    if expected_type is dict:
        return "an object"
    if expected_type is list:
        return "an array"
    if expected_type is str:
        return "a string"
    if expected_type is int:
        return "an integer"
    return expected_type.__name__


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def matches_expected_type(value: Any, expected_type: type[Any]) -> bool:
    if expected_type is int:
        return is_integer(value)
    return isinstance(value, expected_type)


def probe_schema_messages(document: dict[str, Any]) -> list[str]:
    if document.get("schema_version") != SCHEMA_VERSION:
        return [
            (
                f"unsupported schema_version {document.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        ]

    messages: list[str] = []
    missing = sorted(REQUIRED_RESULT_KEYS.difference(document.keys()))
    if missing:
        messages.append(f"missing required result keys: {', '.join(missing)}")

    for field, expected_type in RESULT_FIELD_TYPES.items():
        if field not in document:
            continue
        value = document[field]
        if not matches_expected_type(value, expected_type):
            messages.append(f"{field} must be {type_description(expected_type)}")

    status = document.get("status")
    if isinstance(status, str) and status not in VALID_STATUSES:
        messages.append(f"status must be one of {', '.join(sorted(VALID_STATUSES))}")

    profile = document.get("profile")
    if isinstance(profile, str) and profile not in VALID_PROFILES:
        messages.append(f"profile must be one of {', '.join(sorted(VALID_PROFILES))}")

    data_version = document.get("narc_data_version")
    if isinstance(data_version, int) and not isinstance(data_version, bool):
        if data_version < 0:
            messages.append("narc_data_version must be at least 0")

    errors = document.get("errors")
    if isinstance(errors, list):
        for index, entry in enumerate(errors):
            if not isinstance(entry, dict):
                messages.append(f"errors[{index}] must be an object")
    if document.get("status") == "pass" and errors:
        messages.append("errors must be empty when status is pass")

    checks = document.get("checks")
    if isinstance(checks, dict):
        output_hash = checks.get("output_hash")
        if output_hash is not None and not isinstance(output_hash, str):
            messages.append("checks.output_hash must be a string when present")
        input_hash = checks.get("input_hash")
        if input_hash is not None and not isinstance(input_hash, str):
            messages.append("checks.input_hash must be a string when present")

    measurements = document.get("measurements")
    if isinstance(measurements, dict):
        timing = measurements.get("timing")
        if timing is not None and not isinstance(timing, dict):
            messages.append("measurements.timing must be an object when present")

    fingerprint = document.get("fingerprint")
    if isinstance(fingerprint, dict):
        device = fingerprint.get("device")
        if not isinstance(device, dict):
            messages.append("fingerprint.device must be an object")
        else:
            device_type = device.get("type")
            if not isinstance(device_type, str) or not device_type:
                messages.append("fingerprint.device.type must be a non-empty string")
            elif device_type not in {"cpu", "cuda"}:
                messages.append("fingerprint.device.type must be cpu or cuda")
            accelerator_id = device.get("accelerator_id")
            valid_accelerator_id = (
                isinstance(accelerator_id, str) and bool(accelerator_id)
            )
            if accelerator_id is not None and not valid_accelerator_id:
                messages.append(
                    "fingerprint.device.accelerator_id must be a non-empty string "
                    "when present"
                )
            if device_type == "cuda" and accelerator_id is None:
                messages.append(
                    "fingerprint.device.accelerator_id must be a non-empty string "
                    "for cuda results"
                )
            cuda_driver = device.get("cuda_driver")
            if cuda_driver is not None and not isinstance(cuda_driver, dict):
                messages.append(
                    "fingerprint.device.cuda_driver must be an object when present"
                )
    return messages


def is_probe_result(document: dict[str, Any]) -> bool:
    return not probe_schema_messages(document)


def probe_schema_error(path: ResultLocation, document: dict[str, Any]) -> dict[str, Any]:
    message = "; ".join(probe_schema_messages(document))
    return {
        "path": location_text(path),
        "type": "SchemaError",
        "message": message,
    }


def invalid_result_error(path: ResultLocation, message: str) -> dict[str, Any]:
    return {
        "path": location_text(path),
        "type": "SchemaError",
        "message": message,
    }


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


def result_count_failure(
    loaded_count: int,
    expected_results: int | None,
) -> dict[str, Any] | None:
    if expected_results is None or loaded_count == expected_results:
        return None
    return {
        "expected_results": expected_results,
        "loaded_results": loaded_count,
        "message": (
            f"expected {expected_results} result file(s), "
            f"loaded {loaded_count}"
        ),
    }


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def aggregate_path(
    path: ResultLocation,
    *,
    exclude_paths: set[ResultLocation] | None = None,
    expected_results: int | None = None,
) -> dict[str, Any]:
    json_paths = iter_json_paths(path, exclude_paths=exclude_paths)
    explicit_inputs = explicit_input_identities([path])
    loaded: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    ignored_files: list[str] = []
    for json_path in json_paths:
        result, error = load_result(json_path)
        if result is None:
            if error is not None:
                load_errors.append(error)
            elif location_identity(json_path) in explicit_inputs:
                schema_errors.append(
                    invalid_result_error(json_path, "result JSON must be an object")
                )
            else:
                ignored_files.append(location_text(json_path))
            continue
        if not is_probe_result(result):
            if (
                "schema_version" in result
                or location_identity(json_path) in explicit_inputs
            ):
                schema_errors.append(probe_schema_error(json_path, result))
            else:
                ignored_files.append(location_text(json_path))
            continue
        result["source_path"] = location_text(json_path)
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
    count_failure = result_count_failure(len(loaded), expected_results)

    return {
        "input": location_text(path),
        "expected_results": expected_results,
        "total_files": len(json_paths),
        "loaded_results": len(loaded),
        "load_errors": load_errors,
        "schema_errors": schema_errors,
        "ignored_files": ignored_files,
        "status_counts": status_counts,
        "fingerprint_hashes": by_fingerprint,
        "probe_config_hashes": by_config,
        "output_hashes": output_hashes,
        "output_hash_failures": hash_failures,
        "accelerator_ids": accelerator_ids,
        "duplicate_accelerator_failures": accelerator_failures,
        "result_count_failure": count_failure,
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
            and not schema_errors
            and not hash_failures
            and not accelerator_failures
            and count_failure is None
            and len(loaded) > 0
        ),
    }


def handle_aggregate(args: argparse.Namespace) -> None:
    output_path = validate_outfile(args.outfile) if args.outfile else None
    input_path = args.path
    if output_path:
        validate_output_path(input_path, output_path)
    exclude_paths = {output_path} if output_path else None
    summary = aggregate_path(
        input_path,
        exclude_paths=exclude_paths,
        expected_results=args.expected_results,
    )
    if output_path:
        write_json_report(output_path, summary)
    else:
        sys.stdout.write(json_report_text(summary))
    if args.fail_on_fail and not summary["pass"]:
        raise SystemExit(1)


def generate_aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate narc per-device JSON results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Result JSON file, directory, or S3 URI.")
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit non-zero when aggregation finds failures or load errors.",
    )
    parser.add_argument(
        "--expected-results",
        type=non_negative_int,
        help="Fail unless exactly this many probe result files are loaded.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Output file or S3 URI for aggregate JSON.",
    )
    parser.set_defaults(func=handle_aggregate)
    return parser
