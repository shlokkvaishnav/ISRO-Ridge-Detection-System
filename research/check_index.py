#!/usr/bin/env python3
"""Checks that every research/<name>/ directory containing a SPEC.md is
referenced by name in research/README.md's experiment index.

This only checks that a directory is *named* somewhere in the index table --
it cannot check that the row's status is still true (a row reading "Not
started" for finished work passes this check; that's a reviewer's job, per
research/AGENT_PIPELINE.md's Implementer instructions).

Usage: python research/check_index.py
Exit code 0 = every SPEC.md-containing directory is referenced. Exit 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
README_PATH = RESEARCH_DIR / "README.md"


def find_experiment_dirs() -> list[str]:
    dirs = []
    for spec in RESEARCH_DIR.rglob("SPEC.md"):
        rel = spec.parent.relative_to(RESEARCH_DIR).as_posix()
        dirs.append(rel)
    return sorted(dirs)


def main() -> int:
    if not README_PATH.exists():
        print(f"ERROR: {README_PATH} does not exist", file=sys.stderr)
        return 1

    readme_text = README_PATH.read_text(encoding="utf-8")
    experiment_dirs = find_experiment_dirs()

    missing = [d for d in experiment_dirs if d not in readme_text]

    if missing:
        print("The following research/<name>/ directories (containing a SPEC.md) are not")
        print("referenced anywhere in research/README.md's experiment index:")
        for d in missing:
            print(f"  - research/{d}/")
        print()
        print("Add a row for each to research/README.md before merging.")
        return 1

    print(f"OK: all {len(experiment_dirs)} experiment director{'y is' if len(experiment_dirs) == 1 else 'ies are'} indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
