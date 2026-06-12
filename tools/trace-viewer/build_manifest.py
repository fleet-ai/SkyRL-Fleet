#!/usr/bin/env python3
"""Build ``public/data/manifest.json`` for the SkyRL trace viewer.

Two ways to use it:

1. Index runs already sitting under ``public/data/`` (the default). Each immediate
   subdirectory that contains ``global_step_*.jsonl`` files is treated as one run::

       python build_manifest.py

2. Import a training run's trajectories from elsewhere on disk and index them.
   ``dump_training_trajectories`` writes to ``{export_path}/dumped_trajectories``::

       python build_manifest.py --import ~/exports/dumped_trajectories --name my-run

   This copies (or symlinks, with --link) the ``global_step_*.jsonl`` files into
   ``public/data/my-run/`` and regenerates the manifest.

Then serve with ``./serve.sh`` and open the printed URL. (For ad-hoc inspection you
can also skip all of this and just drag the JSONL files onto the page.)
"""
import argparse
import json
import os
import re
import shutil
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="import_dir", help="dumped_trajectories dir to import")
    ap.add_argument("--name", help="run name (dir under public/data) for --import")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy on import")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.import_dir:
        if not args.name:
            raise SystemExit("--name is required with --import")
        do_import(Path(args.import_dir).expanduser(), args.name, args.link)

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
