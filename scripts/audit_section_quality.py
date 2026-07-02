#!/usr/bin/env python3
"""Lightweight editorial triage for section files."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTION_GLOB = "chapters/*/*.md"
GENERIC_MARKERS = [
    "Use this extension to turn",
    "### Experiment Log Template",
    "### Oral Exam Bank",
    "### Deliverables Checklist",
    "### Concept Audit",
    "### Worked Example Protocol",
    "### Implementation Protocol",
    "### Transfer Prompt",
]


def section_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob(SECTION_GLOB)
        if path.is_file() and path.name != "README.md"
    )


def headings(text: str) -> list[str]:
    return [
        match.group(1).strip().lower()
        for match in re.finditer(r"^#{2,6}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    ]


def score(path: Path) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    notes: list[str] = []
    value = 0

    for marker in GENERIC_MARKERS:
        count = text.count(marker)
        if count:
            value += count * 3
            notes.append(f"{count}x {marker}")

    heading_counts = Counter(headings(text))
    repeated = [name for name, count in heading_counts.items() if count > 1]
    if repeated:
        value += len(repeated) * 2
        notes.append(f"duplicate headings: {', '.join(repeated[:4])}")

    if "Source" not in text[:800] and "source" not in text[:800]:
        value += 2
        notes.append("weak source signal near top")

    if "References" not in text and "Further Reading" not in text:
        value += 1
        notes.append("no references section")

    return value, notes


def main() -> int:
    rows: list[tuple[int, Path, list[str]]] = []
    marker_files = 0
    high_score = 0
    files = section_files()

    for path in files:
        value, notes = score(path)
        if any("x " in note for note in notes):
            marker_files += 1
        if value >= 20:
            high_score += 1
        if value:
            rows.append((value, path, notes))

    rows.sort(reverse=True, key=lambda item: item[0])

    print(f"Section files audited: {len(files)}")
    print(f"Files with generic markers: {marker_files}")
    print(f"Files scoring 20+: {high_score}")

    if rows:
        print("\nTop editorial triage candidates:")
        for value, path, notes in rows[:30]:
            rel = path.relative_to(ROOT)
            print(f"- score {value:02d}: {rel} ({'; '.join(notes[:4])})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
