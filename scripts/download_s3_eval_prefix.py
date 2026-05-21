#!/usr/bin/env python3
"""Download a SkyRL eval S3 prefix without running any post-processing."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("s3_prefix")
    parser.add_argument("dest")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only download files whose relative path exactly matches this value. Can be repeated.",
    )
    args = parser.parse_args()

    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("boto3 is required. Install with: python -m pip install boto3") from exc

    bucket, prefix = parse_s3_uri(args.s3_prefix)
    prefix = prefix.rstrip("/") + "/"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    paginator = client.get_paginator("list_objects_v2")

    downloads: list[tuple[str, Path, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :]
            if args.only and rel not in set(args.only):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            size = int(obj.get("Size", 0))
            if out.exists() and out.stat().st_size == size:
                continue
            downloads.append((key, out, size))

    print(f"download plan: {len(downloads)} missing/stale files")
    bytes_done = 0

    def fetch(item: tuple[str, Path, int]) -> int:
        key, out, size = item
        client.download_file(bucket, key, str(out))
        return size

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item) for item in downloads]
        for n, future in enumerate(as_completed(futures), start=1):
            bytes_done += future.result()
            if n % 250 == 0 or n == len(futures):
                print(f"downloaded {n}/{len(futures)} files ({bytes_done / 1024 / 1024:.1f} MB)", flush=True)

    print(f"download complete: {dest}")


if __name__ == "__main__":
    main()
