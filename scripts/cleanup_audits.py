#!/usr/bin/env python3
"""One-time script to organize audit files into model subdirectories and remove duplicates.

For each (benchmark, dialect, model) group, keeps only the file with the most
successful pairs (fewest errors). Moves all files into data/audits/{model}/.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

AUDIT_DIR = Path("data/audits")

# Pattern: audit_{benchmark}_{dialect}_{model}_{timestamp}.json
FILE_RE = re.compile(
    r"audit_(.+?)_(aave|hiberno_english|indian_english)_(.+?)_(\d{8}_\d{6})\.json"
)


def parse_audit_file(path: Path) -> dict:
    """Read an audit file and return key stats."""
    with open(path) as f:
        data = json.load(f)

    total_pairs = data.get("total_pairs", 0)
    pairs = data.get("pairs", [])
    errors = sum(1 for p in pairs if p.get("error"))
    successful = len(pairs) - errors

    return {
        "path": path,
        "total_pairs": total_pairs,
        "num_pairs": len(pairs),
        "errors": errors,
        "successful": successful,
    }


def main():
    # Collect all audit JSON files (top-level and in subdirs, but not archive)
    all_files = []
    for f in AUDIT_DIR.glob("audit_*.json"):
        all_files.append(f)
    for subdir in AUDIT_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("archive"):
            for f in subdir.glob("audit_*.json"):
                all_files.append(f)

    # Group by (benchmark, dialect, model)
    groups = defaultdict(list)
    for f in all_files:
        m = FILE_RE.match(f.name)
        if not m:
            print(f"SKIP (no match): {f}")
            continue
        benchmark, dialect, model, timestamp = m.groups()
        groups[(benchmark, dialect, model)].append(f)

    to_delete = []
    to_move = []  # (src, dst)

    for key, files in sorted(groups.items()):
        benchmark, dialect, model = key
        model_dir = AUDIT_DIR / model
        model_dir.mkdir(exist_ok=True)

        if len(files) == 1:
            f = files[0]
            dst = model_dir / f.name
            if f.parent != model_dir:
                to_move.append((f, dst))
            continue

        # Multiple files: pick the best one
        stats = []
        for f in files:
            try:
                s = parse_audit_file(f)
                stats.append(s)
            except Exception as e:
                print(f"ERROR reading {f}: {e}")
                to_delete.append(f)

        # Sort by successful pairs descending, then by timestamp (latest first)
        stats.sort(key=lambda s: (s["successful"], s["path"].stem), reverse=True)

        best = stats[0]
        print(f"\n{benchmark} / {dialect} / {model}: {len(stats)} files")
        for s in stats:
            marker = " <-- KEEP" if s is best else " <-- DELETE"
            print(f"  {s['path'].name}: {s['successful']} ok, {s['errors']} errors{marker}")

        # Keep best, delete rest
        for s in stats[1:]:
            to_delete.append(s["path"])

        # Move best to model dir if needed
        dst = model_dir / best["path"].name
        if best["path"].parent != model_dir:
            to_move.append((best["path"], dst))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Files to DELETE: {len(to_delete)}")
    for f in to_delete:
        print(f"  rm {f}")
    print(f"\nFiles to MOVE: {len(to_move)}")
    for src, dst in to_move:
        print(f"  {src} -> {dst}")

    confirm = input("\nProceed? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    for f in to_delete:
        f.unlink()
        print(f"Deleted: {f}")

    for src, dst in to_move:
        if dst.exists():
            # Best file is already in the model dir, just delete the source
            src.unlink()
            print(f"Removed duplicate source: {src}")
        else:
            shutil.move(str(src), str(dst))
            print(f"Moved: {src} -> {dst}")

    print("\nDone!")


if __name__ == "__main__":
    main()
