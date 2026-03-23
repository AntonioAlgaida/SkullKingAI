#!/usr/bin/env python3
# scripts/clear_reset.py
#
# Wipes all training state so the loop can start from scratch.
# Keeps source code, configuration, and data/artifacts/ intact.
#
# Usage:
#   uv run python scripts/clear_reset.py           # prompts for confirmation
#   uv run python scripts/clear_reset.py --yes     # skip confirmation (CI / scripted use)

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRS_TO_DELETE = [
    PROJECT_ROOT / "data" / "chroma_db",   # Grimoire (ChromaDB vector store)
    PROJECT_ROOT / "outputs",              # Hydra run dirs + game traces
]

FILES_TO_DELETE = [
    PROJECT_ROOT / "data" / "elo_ratings.json",     # ELO ratings
    PROJECT_ROOT / "data" / "eval_log.jsonl",        # evaluation checkpoints
    PROJECT_ROOT / "data" / "processed_traces.json", # sleep cycle tracker
]

PRESERVED = [
    "data/artifacts/   (research notes, plots, exported rules)",
    "src/              (source code)",
    "conf/             (configuration)",
    "scripts/          (scripts)",
]


def main():
    parser = argparse.ArgumentParser(description="Reset SkullZero training state.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    print("=" * 55)
    print("  SkullZero — Clean Reset")
    print("=" * 55)

    print("\nThe following will be PERMANENTLY DELETED:\n")
    for d in DIRS_TO_DELETE:
        exists = "  (exists)" if d.exists() else "  (not found — skip)"
        print(f"  DIR   {d.relative_to(PROJECT_ROOT)}{exists}")
    for f in FILES_TO_DELETE:
        exists = "  (exists)" if f.exists() else "  (not found — skip)"
        print(f"  FILE  {f.relative_to(PROJECT_ROOT)}{exists}")

    print("\nThe following will be PRESERVED:\n")
    for p in PRESERVED:
        print(f"  {p}")

    if not args.yes:
        print()
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print()
    deleted = 0

    for d in DIRS_TO_DELETE:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Deleted  {d.relative_to(PROJECT_ROOT)}/")
            deleted += 1
        else:
            print(f"  Skipped  {d.relative_to(PROJECT_ROOT)}/  (not found)")

    for f in FILES_TO_DELETE:
        if f.exists():
            f.unlink()
            print(f"  Deleted  {f.relative_to(PROJECT_ROOT)}")
            deleted += 1
        else:
            print(f"  Skipped  {f.relative_to(PROJECT_ROOT)}  (not found)")

    print()
    print(f"Reset complete. {deleted} item(s) removed.")
    print("Ready to run:  ./run_loop.sh 20 3")
    print("=" * 55)


if __name__ == "__main__":
    main()
