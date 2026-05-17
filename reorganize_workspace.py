#!/usr/bin/env python3
"""
Workspace Reorganization Script - PixelScribe
Moves files from the flat layout into a clean production tree.

Usage
-----
    python reorganize_workspace.py              # dry-run (preview)
    python reorganize_workspace.py --execute    # actually move files

Safety
------
  - Every move is logged.
  - Dry-run by default -- nothing touches disk unless --execute is passed.
  - Existing files at the destination are NEVER overwritten.
  - Source files are only removed after a successful copy + verify.
  - __pycache__ and .DS_Store cleanup only runs in --execute mode.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path


# --- Project root = wherever this script lives ---
ROOT = Path(__file__).resolve().parent

# --- Directories to create ---
DIRS_TO_CREATE = [
    ROOT / "backend",
    ROOT / "models",
    ROOT / "frontend" / "fonts",
    ROOT / "data" / "uploads",
    ROOT / "data" / "results",
]

# --- File moves: (source relative to ROOT, destination relative to ROOT)
FILE_MOVES = [
    # Backend Python code
    ("app.py",                      "backend/app.py"),
    ("server.py",                   "backend/server.py"),
    ("worker.py",                   "backend/worker.py"),
    ("text_pipeline.py",            "backend/text_pipeline.py"),
    ("train_font_classifier.py",    "backend/train_font_classifier.py"),
    ("requirements.txt",            "backend/requirements.txt"),
    ("Dockerfile",                  "backend/Dockerfile"),
    # Ephemeral data
    ("jobs.db",                     "data/jobs.db"),
]

# --- Artifacts to purge ---
PURGE_PATTERNS = ["__pycache__", ".DS_Store", "Thumbs.db"]


def _log(msg: str, *, dry: bool = False):
    prefix = "[DRY-RUN]" if dry else "[EXEC]"
    print(f"  {prefix}  {msg}")


def create_dirs(*, dry: bool):
    print("\n-- Creating directories ------------------------------")
    for d in DIRS_TO_CREATE:
        if d.exists():
            _log(f"EXISTS  {d.relative_to(ROOT)}/", dry=dry)
        else:
            _log(f"MKDIR   {d.relative_to(ROOT)}/", dry=dry)
            if not dry:
                d.mkdir(parents=True, exist_ok=True)


def move_files(*, dry: bool):
    print("\n-- Moving files -------------------------------------")
    for src_rel, dst_rel in FILE_MOVES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel

        if not src.exists():
            _log(f"SKIP    {src_rel}  (source does not exist)", dry=dry)
            continue
        if dst.exists():
            _log(f"SKIP    {src_rel} -> {dst_rel}  (destination already exists)", dry=dry)
            continue

        _log(f"MOVE    {src_rel}  ->  {dst_rel}", dry=dry)
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            # Verify copy integrity
            if dst.stat().st_size == src.stat().st_size:
                src.unlink()
            else:
                print(f"  WARNING: SIZE MISMATCH -- kept source: {src_rel}")


def migrate_directory_contents(src_name: str, dst_name: str, *, dry: bool):
    """Move all files inside an old directory into a new one."""
    src_dir = ROOT / src_name
    dst_dir = ROOT / dst_name
    if not src_dir.exists() or not src_dir.is_dir():
        return
    for item in src_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        target = dst_dir / item.name
        if target.exists():
            _log(f"SKIP    {src_name}/{item.name} (already at destination)", dry=dry)
            continue
        _log(f"MOVE    {src_name}/{item.name}  ->  {dst_name}/{item.name}", dry=dry)
        if not dry:
            shutil.move(str(item), str(target))


def create_gitkeep(*, dry: bool):
    """Place .gitkeep in empty data directories."""
    print("\n-- Placing .gitkeep files ----------------------------")
    for d in [
        ROOT / "data" / "uploads",
        ROOT / "data" / "results",
        ROOT / "frontend" / "fonts",
    ]:
        gk = d / ".gitkeep"
        if gk.exists():
            _log(f"EXISTS  {gk.relative_to(ROOT)}", dry=dry)
        else:
            _log(f"CREATE  {gk.relative_to(ROOT)}", dry=dry)
            if not dry:
                d.mkdir(parents=True, exist_ok=True)
                gk.touch()


def purge_artifacts(*, dry: bool):
    print("\n-- Purging cache artifacts --------------------------")
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False):
        p = Path(dirpath)
        # Skip .git internals
        if ".git" in p.parts:
            continue
        # Purge __pycache__ dirs
        if p.name == "__pycache__":
            _log(f"RMTREE  {p.relative_to(ROOT)}/", dry=dry)
            if not dry:
                shutil.rmtree(p, ignore_errors=True)
            continue
        # Purge .DS_Store / Thumbs.db files
        for fname in filenames:
            if fname in (".DS_Store", "Thumbs.db"):
                fp = p / fname
                _log(f"DELETE  {fp.relative_to(ROOT)}", dry=dry)
                if not dry:
                    fp.unlink(missing_ok=True)


def cleanup_empty_old_dirs(*, dry: bool):
    """Remove the now-empty old uploads/ and results/ from root."""
    print("\n-- Cleaning up old empty directories -----------------")
    for name in ("uploads", "results"):
        d = ROOT / name
        if d.exists() and d.is_dir():
            try:
                remaining = list(d.iterdir())
                if not remaining or all(f.name == ".gitkeep" for f in remaining):
                    _log(f"RMDIR   {name}/", dry=dry)
                    if not dry:
                        shutil.rmtree(d, ignore_errors=True)
                else:
                    _log(f"SKIP    {name}/  (still has {len(remaining)} file(s))", dry=dry)
            except Exception:
                pass


def preserve_permissions(*, dry: bool):
    """Ensure the Start script retains executable permission."""
    print("\n-- Verifying executable permissions ------------------")
    start = ROOT / "Start"
    if start.exists():
        current = start.stat().st_mode
        if not (current & stat.S_IXUSR):
            _log("CHMOD   Start  +x", dry=dry)
            if not dry:
                start.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _log("OK      Start  (already executable)", dry=dry)


def print_final_tree():
    print("\n-- Final directory tree ------------------------------")
    print("""
  editor/
  |-- backend/
  |   |-- app.py
  |   |-- server.py
  |   |-- worker.py
  |   |-- text_pipeline.py
  |   |-- train_font_classifier.py
  |   |-- requirements.txt
  |   +-- Dockerfile
  |-- models/
  |   +-- font_classifier.onnx
  |-- frontend/
  |   |-- index.html
  |   |-- main.js
  |   +-- fonts/
  |       +-- .gitkeep
  |-- data/
  |   |-- uploads/
  |   |   +-- .gitkeep
  |   |-- results/
  |   |   +-- .gitkeep
  |   +-- jobs.db
  |-- nginx.conf
  |-- docker-compose.yml
  |-- Start
  |-- readme.md
  +-- .gitignore
""")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize PixelScribe workspace into a clean production layout."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform the moves. Without this flag, runs in dry-run mode."
    )
    args = parser.parse_args()
    dry = not args.execute

    print("=" * 60)
    if dry:
        print("  MODE: DRY-RUN  (pass --execute to apply changes)")
    else:
        print("  MODE: EXECUTE  (files will be moved)")
    print("=" * 60)

    create_dirs(dry=dry)
    move_files(dry=dry)
    migrate_directory_contents("uploads", "data/uploads", dry=dry)
    migrate_directory_contents("results", "data/results", dry=dry)
    create_gitkeep(dry=dry)
    purge_artifacts(dry=dry)
    cleanup_empty_old_dirs(dry=dry)
    preserve_permissions(dry=dry)
    print_final_tree()

    if dry:
        print("  (i) This was a dry run. Re-run with --execute to apply.\n")
    else:
        print("  [OK] Reorganization complete!")
        print("  [!!] Remember to update path constants in the moved files.")
        print("       See workspace_reorganization.md for the exact diffs.\n")


if __name__ == "__main__":
    main()
