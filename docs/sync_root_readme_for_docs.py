#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sync the project root README into the docs tree with adjusted links.

Canonical source:
    /README.md

Generated target:
    /docs/project_overview/README.md

This script rewrites documentation links so they work from within the docs tree,
while keeping the root README clean for GitHub and local repo usage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_README = PROJECT_ROOT / "README.md"
TARGET_DIR = DOCS_DIR / "project_overview"
TARGET_README = TARGET_DIR / "README.md"


def transform_links(text: str) -> str:
    """
    Adjust guide, demo and license links for the docs-local copy.

    Replace the published demo link with a local Sphinx document link so local
    builds use their own documentation and do not require internet access.

    Root README uses:
        docs/guides/XYZ.md

    Docs-local copy should use:
        ../guides/XYZ.md
    """
    text = text.replace("](docs/guides/", "](../guides/")
    text = text.replace("](docs/demo_docs/", "](../demo_docs/")
    text = text.replace(
        "[documentation](https://NVIDIA.github.io/ACCV-Lab/demo_docs/demos.html)",
        "[documentation](../demo_docs/demos.rst)",
    )
    return text.replace("](THIRD_PARTY_LICENSES.md)", "](../third_party_licenses.md)")


def add_quick_start_label(text: str) -> str:
    """
    Inject a MyST section label for the Quick Start section **only** in the
    docs-local copy of the README.

    This keeps the project root README.md clean (no MyST-specific syntax),
    while allowing Sphinx to reference the Quick Start section via
    ``:ref:`project_overview-quick-start``` from within the docs.
    """
    label = "(project_overview-quick-start)="
    marker = "## Quick Start"

    # If the label is already present (e.g., from a previous run), do nothing.
    if label in text:
        return text

    # Insert the label immediately before the first "## Quick Start" heading.
    if marker in text:
        return text.replace(marker, f"{label}\n{marker}", 1)

    # Fallback: no Quick Start section found; return text unchanged.
    return text


def main() -> int:
    if not SOURCE_README.exists():
        raise SystemExit(f"Source README not found: {SOURCE_README}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    raw = SOURCE_README.read_text(encoding="utf-8")
    transformed = transform_links(raw)
    transformed = add_quick_start_label(transformed)

    TARGET_README.write_text(transformed, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
