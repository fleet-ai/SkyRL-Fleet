from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from narc.aggregate import (
    device_fingerprint,
    invalid_result_error,
    is_probe_result,
    probe_schema_error,
)
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


COMPARE_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalized_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized_value(value[key])
            for key in sorted(value)
            if isinstance(key, str)
        }
    if isinstance(value, list):
        return [normalized_value(item) for item in value]
    return value


def equivalence(result: dict[str, Any]) -> dict[str, Any]:
    checks = result["checks"]
    status = result["status"]
    value: dict[str, Any] = {
        "schema_version": result["schema_version"],
        "narc_data_version": result.get("narc_data_version"),
        "status": status,
        "profile": result["profile"],
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
        "profile": result.get("profile"),
        "hostname": result.get("hostname"),
        "accelerator_id": device.get("accelerator_id"),
        "run_id": result.get("run_id"),
        "narc_data_version": result.get("narc_data_version"),
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

    partitions = compare(loaded)
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for result in loaded:
        status_counts[result["status"]] += 1
    missing_output_hash = [
        result.get("source_path")
        for result in loaded
        if result["status"] == "pass" and not result["checks"].get("output_hash")
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
        "partition_count": len(partitions),
        "split": split,
        "partitions": partition_reports,
        "pass": (
            bool(loaded)
            and not load_errors
            and not schema_errors
            and not missing_output_hash
            and len(partitions) == 1
            and status_counts["warn"] == 0
            and status_counts["fail"] == 0
        ),
    }


def is_compare_report(document: dict[str, Any] | None) -> bool:
    return (
        isinstance(document, dict)
        and document.get("compare_schema_version") == COMPARE_SCHEMA_VERSION
    )


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
    return (
        output_key.endswith(".json")
        and not output_key.endswith(".tmp")
        and output_key.startswith(prefix)
    )


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
        raise ValueError(
            "outfile must not overwrite an existing JSON file from the compare inputs"
        )


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
