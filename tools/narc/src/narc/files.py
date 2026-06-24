from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

ResultLocation = Path | str


def is_s3_uri(location: ResultLocation) -> bool:
    return isinstance(location, str) and location.lower().startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.lower().startswith("s3://"):
        raise ValueError(f"invalid S3 URI: {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"invalid S3 URI: {uri}")
    return bucket, key if separator else ""


def s3_uri(bucket: str, key: str) -> str:
    if key:
        return f"s3://{bucket}/{key}"
    return f"s3://{bucket}"


def location_identity(location: ResultLocation) -> str:
    if is_s3_uri(location):
        bucket, key = parse_s3_uri(str(location))
        return s3_uri(bucket, key)
    return str(Path(location).resolve())


def location_text(location: ResultLocation) -> str:
    return str(location)


def s3_client() -> Any:
    return boto3.client("s3")


def s3_error(uri: str, error: Exception) -> dict[str, Any]:
    return {
        "path": uri,
        "type": type(error).__name__,
        "message": str(error),
    }


def require_s3_object_uri(uri: str) -> tuple[str, str]:
    bucket, key = parse_s3_uri(uri)
    if not key:
        raise ValueError("S3 URI must include an object key")
    return bucket, key


def load_s3_result(uri: str) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        bucket, key = require_s3_object_uri(uri)
        response = s3_client().get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        if isinstance(body, bytes):
            text = body.decode("utf-8")
        else:
            text = str(body)
        return json.loads(text), None
    except Exception as error:
        return None, s3_error(uri, error)


def write_s3_text(uri: str, text: str) -> None:
    bucket, key = require_s3_object_uri(uri)
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="application/json",
    )


def s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    if not key:
        return False
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def list_s3_json_paths(
    uri: str,
    *,
    exclude_paths: set[ResultLocation] | None = None,
) -> list[ResultLocation]:
    excluded = {location_identity(path) for path in exclude_paths or set()}
    bucket, prefix = parse_s3_uri(uri)
    client = s3_client()
    if (
        not prefix.endswith("/")
        and s3_object_exists(client, bucket, prefix)
        and location_identity(uri) not in excluded
    ):
        return [uri]

    list_prefix = prefix
    if list_prefix and not list_prefix.endswith("/"):
        list_prefix = f"{list_prefix}/"

    locations: list[ResultLocation] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=list_prefix)
        for page in pages:
            contents = page.get("Contents", [])
            if not isinstance(contents, list):
                continue
            for entry in contents:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("Key")
                if not isinstance(key, str):
                    continue
                entry_uri = s3_uri(bucket, key)
                if (
                    key.endswith(".json")
                    and not key.endswith(".tmp")
                    and location_identity(entry_uri) not in excluded
                ):
                    locations.append(entry_uri)
    except Exception:
        return [uri]
    if (
        not locations
        and uri.endswith(".json")
        and not uri.endswith(".tmp")
        and location_identity(uri) not in excluded
    ):
        return [uri]
    return sorted(locations, key=location_text)


def iter_json_paths(
    path: ResultLocation,
    exclude_paths: set[ResultLocation] | None = None,
) -> list[ResultLocation]:
    if is_s3_uri(path):
        return list_s3_json_paths(str(path), exclude_paths=exclude_paths)

    excluded = {location_identity(path) for path in exclude_paths or set()}
    local_path = Path(path)
    if local_path.is_file():
        if location_identity(local_path) in excluded:
            return []
        return [local_path]
    return sorted(
        candidate
        for candidate in local_path.rglob("*.json")
        if not candidate.name.endswith(".tmp")
        and location_identity(candidate) not in excluded
    )


def input_location_is_explicit(path: ResultLocation) -> bool:
    if is_s3_uri(path):
        bucket, key = parse_s3_uri(str(path))
        return bool(key) and (
            key.endswith(".json")
            or key.endswith(".tmp")
            or s3_object_exists(s3_client(), bucket, key)
        )
    return Path(path).is_file()


def explicit_input_identities(paths: list[ResultLocation]) -> set[str]:
    return {
        location_identity(path)
        for path in paths
        if input_location_is_explicit(path)
    }


def load_result(
    path: ResultLocation,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if is_s3_uri(path):
        document, error = load_s3_result(str(path))
        if error is not None:
            return None, error
        if not isinstance(document, dict):
            return None, None
        return document, None

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception as error:
        return None, {
            "path": location_text(path),
            "type": type(error).__name__,
            "message": str(error),
        }
    if not isinstance(document, dict):
        return None, None
    return document, None


def result_location_exists(location: ResultLocation) -> bool:
    if is_s3_uri(location):
        bucket, key = parse_s3_uri(str(location))
        return s3_object_exists(s3_client(), bucket, key)
    return Path(location).exists()


def validate_outfile(outfile: str) -> ResultLocation:
    if is_s3_uri(outfile):
        require_s3_object_uri(outfile)
        return outfile
    return Path(outfile).resolve()


def json_report_text(document: dict[str, Any]) -> str:
    return f"{json.dumps(document, indent=2, sort_keys=True)}\n"


def write_json_report(location: ResultLocation, document: dict[str, Any]) -> None:
    text = json_report_text(document)
    if is_s3_uri(location):
        write_s3_text(str(location), text)
        return
    output_path = Path(location)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def validate_output_path(
    input_path: ResultLocation,
    output_path: ResultLocation,
) -> None:
    if location_identity(input_path) == location_identity(output_path):
        raise ValueError("outfile must not overwrite the input result file")
    if not is_s3_uri(input_path):
        local_input = Path(input_path)
        if (
            not is_s3_uri(output_path)
            and local_input.is_file()
            and local_input.resolve() == Path(output_path).resolve()
        ):
            raise ValueError("outfile must not overwrite the input result file")
    if not result_location_exists(output_path):
        return
    document, error = load_result(output_path)
    if error is not None or document is None:
        return
    if "schema_version" in document:
        raise ValueError("outfile must not overwrite a probe result file")
