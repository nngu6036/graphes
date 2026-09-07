#!/usr/bin/env python3
"""Apply the CatFlow v2 correction to the previous GraphER wrapper release.

All files are checked before any change. Unknown local edits are refused.
Existing files are backed up inside .catflow_fix_backups before replacement.
This does not modify datasets, checkpoints, generations, or external sources.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = args.repo.expanduser().resolve()
    if not (repo / "src/grapher/models").is_dir():
        parser.error("--repo must be the GraphER repository root containing src/grapher/models.")
    manifest = json.loads((here / "manifest.json").read_text())
    todo, conflicts, skipped = [], [], 0
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path: " + str(rel))
        source, target = here / "files" / rel, repo / rel
        if not target.resolve().is_relative_to(repo):
            raise ValueError("Target escapes the repository: " + str(target))
        if sha256(source) != row["after_sha256"]:
            raise ValueError("Patch file checksum mismatch: " + str(rel))
        current = sha256(target) if target.is_file() else None
        if current == row["after_sha256"]:
            skipped += 1
            continue
        if target.exists() and not target.is_file():
            conflicts.append(str(rel) + " (not a regular file)")
        elif current != row["before_sha256"]:
            conflicts.append(str(rel))
        else:
            todo.append((source, target, rel))
    if conflicts:
        print("REFUSED: locally changed or missing baseline files; nothing was written:", file=sys.stderr)
        for rel in conflicts:
            print("  " + rel, file=sys.stderr)
        print("Review/merge those files against the patch payload; keep local work and original runs.", file=sys.stderr)
        return 2
    print(f"Preflight passed: {len(todo)} files to install; {skipped} already match.")
    if args.dry_run or not todo:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = repo / ".catflow_fix_backups" / stamp
    backup.mkdir(parents=True)
    for source, target, rel in todo:
        if target.exists():
            previous = backup / rel
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, previous)
    # Backups are complete before the first replacement.
    for source, target, rel in todo:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".catflow_v2_tmp")
        shutil.copy2(source, staging)
        staging.replace(target)
    (backup / "applied_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Installed CatFlow v2. Original replaced files backed up to " + str(backup))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
