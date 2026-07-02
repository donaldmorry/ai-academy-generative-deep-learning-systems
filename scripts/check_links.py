#!/usr/bin/env python3
"""Check local Markdown links in the course tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:", "#")


def markdown_files() -> list[Path]:
    return sorted(ROOT.rglob("*.md"))


def simple_href(href: str) -> str:
    href = href.strip()
    if " " in href:
        href = href.split()[0]
    if href.startswith("<") and href.endswith(">"):
        href = href[1:-1]
    return href


def should_check(path: str) -> bool:
    if not path:
        return False
    if path.startswith(EXTERNAL_PREFIXES) or path.startswith("/"):
        return False
    path = path.split("#", 1)[0]
    if not path:
        return False
    return path.endswith(".md") or "/" in path or path.startswith(".")


def main() -> int:
    broken: list[tuple[Path, str]] = []
    local_links = 0
    files = markdown_files()

    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            href = simple_href(match.group(1))
            if not should_check(href):
                continue
            target_path = href.split("#", 1)[0]
            target = (source.parent / unquote(target_path)).resolve()
            local_links += 1
            if not target.exists():
                broken.append((source.relative_to(ROOT), href))

    print(f"Markdown files: {len(files)}")
    print(f"Local links: {local_links}")
    print(f"Broken local links: {len(broken)}")

    if broken:
        print("\nBroken links:")
        for source, href in broken[:200]:
            print(f"- {source}: {href}")
        if len(broken) > 200:
            print(f"- ... {len(broken) - 200} more")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
