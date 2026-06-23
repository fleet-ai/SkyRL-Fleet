#!/usr/bin/env python3
"""Build ``public/data/manifest.json`` for the SkyRL trace viewer.

Three ways to use it:

1. Index runs already sitting under ``public/data/`` (the default). Each immediate
   subdirectory that contains ``global_step_*.jsonl`` files is treated as one run::

       python build_manifest.py

2. Import a training run's trajectories from elsewhere on disk and index them.
   ``dump_training_trajectories`` writes to ``{export_path}/dumped_trajectories``::

       python build_manifest.py --import ~/exports/dumped_trajectories --name my-run

   This copies (or symlinks, with --link) the ``global_step_*.jsonl`` files into
   ``public/data/my-run/`` and regenerates the manifest.

3. Pull trajectories from S3 by run id (requires the ``aws`` CLI with credentials
   configured). Re-running is cheap — only new step files are downloaded::

       # list available run ids in the bucket
       python build_manifest.py --s3-list

       # download (or incrementally sync) a run, then rebuild the manifest
       python build_manifest.py --s3 fleet_qwen35_35b_negotiation_dnd_outcome_thinkon_fix

   Use ``--name`` to store the run under a different local name, and ``--s3-prefix``
   to point at a different bucket or prefix (default: s3://skyrl-trajectories/rollouts).

Then serve with ``./serve.sh`` and open the printed URL. (For ad-hoc inspection you
can also skip all of this and just drag the JSONL files onto the page.)
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "public" / "data"
STEP_RE = re.compile(r"global_step_(\d+)\.jsonl$")


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def index_run_dir(run_dir: Path):
    """Return a sorted list of {step, file, count} for a run directory, or None."""
    steps = []
    for p in sorted(run_dir.glob("global_step_*.jsonl")):
        m = STEP_RE.search(p.name)
        if not m:
            continue
        rel = p.relative_to(DATA_DIR).as_posix()
        steps.append({"step": int(m.group(1)), "file": rel, "count": count_lines(p)})
    steps.sort(key=lambda s: s["step"])
    return steps or None


def do_import(src: Path, name: str, link: bool):
    if not src.is_dir():
        raise SystemExit(f"--import path is not a directory: {src}")
    files = sorted(src.glob("global_step_*.jsonl"))
    if not files:
        raise SystemExit(f"no global_step_*.jsonl files found in {src}")
    dst = DATA_DIR / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dst / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if link:
            target.symlink_to(f.resolve())
        else:
            shutil.copy2(f, target)
    print(f"{'linked' if link else 'copied'} {len(files)} step files -> {dst}")


S3_DEFAULT_PREFIX = "s3://skyrl-trajectories/rollouts"


def do_s3_list(prefix: str):
    """Print one run id per line from the S3 prefix and exit."""
    cmd = ["aws", "s3", "ls", prefix.rstrip("/") + "/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"aws s3 ls failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    run_ids = []
    for line in result.stdout.splitlines():
        # Lines for common prefixes look like: "                       PRE run_name/"
        m = re.search(r"PRE\s+(\S+)/\s*$", line)
        if m:
            run_ids.append(m.group(1))
    if not run_ids:
        print("(no runs found)")
    else:
        for rid in run_ids:
            print(rid)


def do_s3_sync(run_id: str, name: str, prefix: str):
    """Sync global_step_*.jsonl files from S3 for *run_id* into public/data/<name>/."""
    src = prefix.rstrip("/") + "/" + run_id + "/"
    dst = DATA_DIR / name
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync", src, str(dst),
        "--exclude", "*",
        "--include", "global_step_*.jsonl",
    ]
    print(f"syncing {src} -> {dst} …")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"aws s3 sync failed (exit {result.returncode}). "
            "Check your AWS credentials and that the run id exists."
        )
    files = list(dst.glob("global_step_*.jsonl"))
    if not files:
        raise SystemExit(
            f"sync completed but no global_step_*.jsonl files found in {dst}. "
            f"Verify that '{run_id}' is a valid run id (try --s3-list)."
        )
    print(f"synced {len(files)} step file(s) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="import_dir", help="dumped_trajectories dir to import")
    ap.add_argument("--name", help="run name (dir under public/data) for --import or --s3")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy on import")
    ap.add_argument(
        "--s3",
        metavar="RUN_ID",
        help=(
            "download (or incrementally sync — only new steps are fetched) "
            "global_step_*.jsonl files for RUN_ID from S3, then rebuild the manifest. "
            "Safe to re-run against a live training run to pick up new steps."
        ),
    )
    ap.add_argument(
        "--s3-list",
        action="store_true",
        help="list available run ids in the S3 bucket and exit",
    )
    ap.add_argument(
        "--s3-prefix",
        default=S3_DEFAULT_PREFIX,
        metavar="S3_URI",
        help=f"S3 prefix for trajectory dumps (default: {S3_DEFAULT_PREFIX})",
    )
    args = ap.parse_args()

    if args.s3_list:
        do_s3_list(args.s3_prefix)
        return

    if args.import_dir and args.s3:
        raise SystemExit("--import and --s3 are mutually exclusive; use one or the other")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.import_dir:
        if not args.name:
            raise SystemExit("--name is required with --import")
        do_import(Path(args.import_dir).expanduser(), args.name, args.link)

    if args.s3:
        name = args.name or args.s3
        do_s3_sync(args.s3, name, args.s3_prefix)

    runs = []
    for child in sorted(DATA_DIR.iterdir()):
        if not child.is_dir():
            continue
        steps = index_run_dir(child)
        if steps:
            runs.append({"name": child.name, "steps": steps})

    manifest = {"runs": runs}
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    total = sum(s["count"] for r in runs for s in r["steps"])
    print(f"wrote {out} — {len(runs)} run(s), "
          f"{sum(len(r['steps']) for r in runs)} step files, {total} trajectories")
    for r in runs:
        print(f"  · {r['name']}: {len(r['steps'])} steps")


if __name__ == "__main__":
    main()
