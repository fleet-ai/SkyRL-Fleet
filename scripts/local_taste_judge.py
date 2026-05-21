#!/usr/bin/env python3
"""Download a SkyRL eval dump locally and run the Anthropic taste judge."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from judge_eval_outputs import main_async


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_s3_prefix(s3_prefix: str, dest: Path, *, workers: int) -> None:
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("boto3 is required for --s3-prefix. Install with: python -m pip install boto3") from exc

    bucket, prefix = parse_s3_uri(s3_prefix)
    dest.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    paginator = client.get_paginator("list_objects_v2")

    downloads: list[tuple[str, Path, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix.rstrip("/") + "/") :]
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == obj.get("Size", -1):
                continue
            downloads.append((key, out, int(obj.get("Size", 0))))
    print(f"download plan: {len(downloads)} missing/stale files", flush=True)

    def fetch(item: tuple[str, Path, int]) -> int:
        key, out, size = item
        client.download_file(bucket, key, str(out))
        return size

    n = 0
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in downloads]
        for future in as_completed(futures):
            bytes_done += future.result()
            n += 1
            if n % 250 == 0 or n == len(futures):
                print(f"downloaded {n}/{len(futures)} files ({bytes_done / 1024 / 1024:.1f} MB)", flush=True)
    print(f"download complete: {dest}", flush=True)


def find_eval_jsonl(root: Path) -> Path:
    preferred = root / "ticketmaster.jsonl"
    if preferred.exists():
        return preferred
    candidates = [
        p
        for p in root.rglob("*.jsonl")
        if p.name not in {"aggregated_results.jsonl", "taste_scores.jsonl"} and not p.name.startswith("taste_")
    ]
    if not candidates:
        raise SystemExit(f"No eval JSONL found under {root}")
    if len(candidates) > 1:
        names = ", ".join(str(p) for p in candidates[:8])
        raise SystemExit(f"Multiple eval JSONLs found; pass --input explicitly. Candidates: {names}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-prefix", help="Example: s3://skyrl-trajectories/evals/run/global_step_0/")
    parser.add_argument("--work-dir", default="local_runs/tm-qwen35-9b-baseline-s42/global_step_0")
    parser.add_argument("--input", help="Local eval JSONL. Defaults to ticketmaster.jsonl under --work-dir.")
    parser.add_argument("--out", help="Output JSONL. Defaults to taste_scores.jsonl next to input.")
    parser.add_argument("--summary", help="Summary JSON. Defaults to taste_summary.json next to input.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-screenshots", type=int, default=8)
    parser.add_argument("--text-model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--visual-model", default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if args.s3_prefix:
        download_s3_prefix(args.s3_prefix, work_dir, workers=args.download_workers)

    input_path = Path(args.input) if args.input else find_eval_jsonl(work_dir)
    judge_args = SimpleNamespace(
        input=str(input_path),
        out=args.out or str(input_path.with_name("taste_scores.jsonl")),
        summary=args.summary or str(input_path.with_name("taste_summary.json")),
        workers=args.workers,
        limit=args.limit,
        num_screenshots=args.num_screenshots,
        text_model=args.text_model,
        visual_model=args.visual_model,
    )
    asyncio.run(main_async(judge_args))


if __name__ == "__main__":
    main()
