from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from narc.files import (
    ResultLocation,
    explicit_input_identities,
    is_s3_uri,
    iter_json_paths,
    json_report_text,
    load_result,
    location_identity,
    location_text,
    parse_s3_uri,
    result_location_exists,
    validate_outfile,
    validate_output_path,
    write_json_report,
)
from narc.schema import SCHEMA_VERSION

COMPARE_SCHEMA_VERSION = 1
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
    "run_id",
    "schema_version",
    "slurm",
    "started_at",
    "status",
}
RESULT_FIELD_TYPES = {
    "schema_version": int,
    "status": str,
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


def matches_expected_type(value: Any, expected_type: type[Any]) -> bool:
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def probe_schema_messages(document: dict[str, Any]) -> list[str]:
    if document.get("schema_version") != SCHEMA_VERSION:
        return [(f"unsupported schema_version {document.get('schema_version')!r}; " f"expected {SCHEMA_VERSION}")]

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
            valid_accelerator_id = isinstance(accelerator_id, str) and bool(accelerator_id)
            if accelerator_id is not None and not valid_accelerator_id:
                messages.append("fingerprint.device.accelerator_id must be a non-empty string " "when present")
            if device_type == "cuda" and accelerator_id is None:
                messages.append("fingerprint.device.accelerator_id must be a non-empty string " "for cuda results")
            cuda_driver = device.get("cuda_driver")
            if cuda_driver is not None and not isinstance(cuda_driver, dict):
                messages.append("fingerprint.device.cuda_driver must be an object when present")
    return messages


def is_probe_result(document: dict[str, Any]) -> bool:
    return not probe_schema_messages(document)


def probe_schema_error(
    path: ResultLocation,
    document: dict[str, Any],
) -> dict[str, Any]:
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalized_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalized_value(value[key]) for key in sorted(value) if isinstance(key, str)}
    if isinstance(value, list):
        return [normalized_value(item) for item in value]
    return value


def equivalence(result: dict[str, Any]) -> dict[str, Any]:
    checks = result["checks"]
    status = result["status"]
    value: dict[str, Any] = {
        "schema_version": result["schema_version"],
        "status": status,
        "probe_config_hash": result["probe_config_hash"],
        "input_hash": checks.get("input_hash"),
        "output_hash": checks.get("output_hash"),
    }
    if status != "pass":
        value["checks"] = normalized_value(checks)
        value["errors"] = normalized_value(result["errors"])
    return value


def equivalence_key(result: dict[str, Any]) -> str:
    return canonical_json(equivalence(result))


def compare(results: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = equivalence_key(result)
        group = groups.setdefault(key, [])
        group.append(result)
    return list(groups.values())


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    device = device_fingerprint(result)
    return {
        "path": result.get("source_path"),
        "status": result.get("status"),
        "hostname": result.get("hostname"),
        "accelerator_id": device.get("accelerator_id"),
        "run_id": result.get("run_id"),
        "probe_config_hash": result.get("probe_config_hash"),
        "input_hash": result.get("checks", {}).get("input_hash"),
        "output_hash": result.get("checks", {}).get("output_hash"),
        "fingerprint_hash": result.get("fingerprint_hash"),
    }


def compare_paths(
    paths: list[ResultLocation],
    *,
    exclude_paths: set[ResultLocation] | None = None,
) -> dict[str, Any]:
    json_paths_by_identity: dict[str, ResultLocation] = {}
    for path in paths:
        for json_path in iter_json_paths(path, exclude_paths=exclude_paths):
            json_paths_by_identity.setdefault(location_identity(json_path), json_path)
    json_paths = sorted(json_paths_by_identity.values(), key=location_text)
    explicit_inputs = explicit_input_identities(paths)

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
                schema_errors.append(invalid_result_error(json_path, "result JSON must be an object"))
            else:
                ignored_files.append(location_text(json_path))
            continue
        if not is_probe_result(result):
            if "schema_version" in result or location_identity(json_path) in explicit_inputs:
                schema_errors.append(probe_schema_error(json_path, result))
            else:
                ignored_files.append(location_text(json_path))
            continue
        result["source_path"] = location_text(json_path)
        loaded.append(result)

    partitions = compare(loaded)
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for result in loaded:
        status_counts[result["status"]] += 1
    missing_output_hash = [
        result.get("source_path")
        for result in loaded
        if result["status"] == "pass" and not result["checks"].get("output_hash")
    ]
    missing_input_hash = [
        result.get("source_path")
        for result in loaded
        if result["status"] == "pass" and not result["checks"].get("input_hash")
    ]
    partition_reports = [
        {
            "id": index,
            "size": len(partition),
            "equivalence": equivalence(partition[0]),
            "results": [result_summary(result) for result in partition],
        }
        for index, partition in enumerate(partitions)
    ]
    split = len(partitions) != 1
    return {
        "compare_schema_version": COMPARE_SCHEMA_VERSION,
        "inputs": [location_text(path) for path in paths],
        "input_count": len(paths),
        "total_files": len(json_paths),
        "loaded_results": len(loaded),
        "load_errors": load_errors,
        "schema_errors": schema_errors,
        "ignored_files": ignored_files,
        "status_counts": status_counts,
        "missing_output_hash": missing_output_hash,
        "missing_input_hash": missing_input_hash,
        "partition_count": len(partitions),
        "split": split,
        "partitions": partition_reports,
        "pass": (
            bool(loaded)
            and not load_errors
            and not schema_errors
            and not missing_output_hash
            and not missing_input_hash
            and len(partitions) == 1
            and status_counts["warn"] == 0
            and status_counts["fail"] == 0
        ),
    }


def is_compare_report(document: dict[str, Any] | None) -> bool:
    return isinstance(document, dict) and document.get("compare_schema_version") == COMPARE_SCHEMA_VERSION


def path_contains(parent: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def output_s3_path_is_input_json(
    path: ResultLocation,
    output_path: ResultLocation,
) -> bool:
    if not is_s3_uri(path) or not is_s3_uri(output_path):
        return False
    input_bucket, input_key = parse_s3_uri(str(path))
    output_bucket, output_key = parse_s3_uri(str(output_path))
    if input_bucket != output_bucket:
        return False
    if input_key == output_key:
        return True
    prefix = input_key
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return output_key.endswith(".json") and not output_key.endswith(".tmp") and output_key.startswith(prefix)


def output_path_is_input_json(
    path: ResultLocation,
    output_path: ResultLocation,
) -> bool:
    if is_s3_uri(path) or is_s3_uri(output_path):
        return output_s3_path_is_input_json(path, output_path)
    resolved_output = Path(output_path).resolve()
    local_path = Path(path)
    if local_path.is_file():
        return local_path.resolve() == resolved_output
    return (
        output_path.suffix == ".json"
        and not output_path.name.endswith(".tmp")
        and path_contains(local_path.resolve(), resolved_output)
    )


def validate_compare_output_path(
    paths: list[ResultLocation],
    output_path: ResultLocation,
) -> None:
    for path in paths:
        validate_output_path(path, output_path)
        if not result_location_exists(output_path) or not output_path_is_input_json(
            path,
            output_path,
        ):
            continue
        document, error = load_result(output_path)
        if error is None and is_compare_report(document):
            continue
        raise ValueError("outfile must not overwrite an existing JSON file from the compare inputs")


def handle_compare(args: argparse.Namespace) -> None:
    output_path = validate_outfile(args.outfile) if args.outfile else None
    input_paths = args.paths
    if output_path:
        validate_compare_output_path(input_paths, output_path)
    report = compare_paths(
        input_paths,
        exclude_paths={output_path} if output_path else None,
    )
    if output_path:
        write_json_report(output_path, report)
    else:
        sys.stdout.write(json_report_text(report))
    if report["load_errors"] or report["schema_errors"] or not report["loaded_results"]:
        raise SystemExit(1)
    if not report["pass"] and not args.nofail:
        raise SystemExit(1)


def generate_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare narc result JSON files by equivalence class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result JSON files, directories, or S3 URIs to compare.",
    )
    parser.add_argument(
        "--nofail",
        action="store_true",
        help=(
            "Exit zero for valid non-passing compare reports; load errors, "
            "schema errors, and empty comparisons still fail."
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Output file or S3 URI for compare JSON.",
    )
    parser.set_defaults(func=handle_compare)
    return parser
