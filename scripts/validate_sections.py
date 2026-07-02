#!/usr/bin/env python3
"""Validate course section files for basic publication quality.

Run from ai-learning/courses:
    python3 scripts/validate_sections.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTION_GLOB = "chapters/*/*.md"
MIN_SECTION_LINES = 120
MIN_LAB_LINES = 50

RAW_OR_TEMPLATE_MARKERS = [
    "TODO",
    "TBD",
    "lorem ipsum",
    "paste source",
    "raw ocr",
]

GENERIC_READABLE_FORMS = [
    "The neural-network equation tracks how activations, weights, and biases transform one representation into the next.",
    "The activation converts a linear preactivation into the signal passed to the next layer.",
    "The layer computes a weighted combination of inputs plus bias before passing information forward.",
]


def section_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob(SECTION_GLOB)
        if path.is_file() and path.name != "README.md"
    )


def minimum_lines_for(path: Path) -> int:
    if path.name.startswith("section-lab-"):
        return MIN_LAB_LINES
    return MIN_SECTION_LINES


def strip_code_fences(lines: list[str]) -> list[str]:
    stripped: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            stripped.append("")
        elif in_fence:
            stripped.append("")
        else:
            stripped.append(line)
    return stripped


def display_blocks(lines: list[str]) -> tuple[list[tuple[int, int]], list[int]]:
    blocks: list[tuple[int, int]] = []
    unclosed: list[int] = []
    start: int | None = None
    for idx, line in enumerate(lines, start=1):
        if line.strip() == "$$":
            if start is None:
                start = idx
            else:
                blocks.append((start, idx))
                start = None
    if start is not None:
        unclosed.append(start)
    return blocks, unclosed


def has_readable_form_after(lines: list[str], end_line: int) -> bool:
    for line in lines[end_line : min(len(lines), end_line + 4)]:
        text = line.strip()
        if not text:
            continue
        return text.startswith(">") and "Readable form" in text
    return False


def grouped_display_blocks(blocks: list[tuple[int, int]], lines: list[str]) -> list[tuple[int, int]]:
    if not blocks:
        return []

    groups: list[tuple[int, int]] = []
    group_start, group_end = blocks[0]

    for start, end in blocks[1:]:
        between = lines[group_end:start - 1]
        only_blank = all(not line.strip() for line in between)
        if only_blank:
            group_end = end
        else:
            groups.append((group_start, group_end))
            group_start, group_end = start, end

    groups.append((group_start, group_end))
    return groups


def validate(path: Path) -> list[str]:
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    visible = strip_code_fences(lines)
    issues: list[str] = []

    minimum_lines = minimum_lines_for(path)
    if len(lines) < minimum_lines:
        issues.append(f"{rel}: under minimum lines ({len(lines)} < {minimum_lines})")

    blocks, unclosed = display_blocks(visible)
    for start in unclosed:
        issues.append(f"{rel}:{start}: unclosed display math delimiter")

    for start, end in blocks:
        if end <= start + 1:
            issues.append(f"{rel}:{start}: empty display math block")

    for start, end in grouped_display_blocks(blocks, visible):
        if not has_readable_form_after(visible, end):
            issues.append(f"{rel}:{start}-{end}: display equation group missing readable form")

    lowered = text.lower()
    for marker in RAW_OR_TEMPLATE_MARKERS:
        if marker.lower() in lowered:
            issues.append(f"{rel}: raw/template marker found: {marker}")

    for phrase in GENERIC_READABLE_FORMS:
        if phrase in text:
            issues.append(f"{rel}: generic readable form warning")

    return issues


def main() -> int:
    files = section_files()
    all_issues: list[str] = []
    under_minimum = bad_latex = missing_readable = raw_markers = generic = 0

    for path in files:
        issues = validate(path)
        all_issues.extend(issues)
        for issue in issues:
            if "under minimum lines" in issue:
                under_minimum += 1
            elif "display math" in issue or "display equation" in issue:
                if "missing readable form" in issue:
                    missing_readable += 1
                else:
                    bad_latex += 1
            elif "raw/template marker" in issue:
                raw_markers += 1
            elif "generic readable form" in issue:
                generic += 1

    print(f"Total sections: {len(files)}")
    print(f"Under minimum lines: {under_minimum}")
    print(f"Bad LaTeX delimiters: {bad_latex}")
    print(f"Display equations missing readable form: {missing_readable}")
    print(f"Raw/OCR/template markers: {raw_markers}")
    print(f"Bad generic readable forms: {generic}")

    fatal_issues = [
        issue for issue in all_issues
        if "generic readable form warning" not in issue
    ]

    if all_issues:
        print("\nIssues:")
        for issue in all_issues[:200]:
            print(f"- {issue}")
        if len(all_issues) > 200:
            print(f"- ... {len(all_issues) - 200} more")

    return 1 if fatal_issues else 0


if __name__ == "__main__":
    sys.exit(main())
