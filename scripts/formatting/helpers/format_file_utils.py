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

"""Shared file discovery helpers for formatting scripts."""

import configparser
import os
from pathlib import Path


def find_project_root(start_path):
    """Walk upward from start_path until the repository root marker is found."""
    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    for candidate in [current_path, *current_path.parents]:
        marker_path = candidate / ".nav"
        if marker_path.is_file() and marker_path.read_text(encoding="utf-8").strip() == "project root":
            return candidate

    raise RuntimeError(
        f"Could not find project root marker .nav containing 'project root' above {start_path}"
    )


PROJECT_ROOT = find_project_root(Path(__file__))


def load_submodule_paths(project_root=PROJECT_ROOT):
    """Return submodule paths declared in .gitmodules, relative to project_root."""
    gitmodules = project_root / ".gitmodules"
    if not gitmodules.exists():
        return set()

    config = configparser.ConfigParser()
    config.read(gitmodules)

    submodule_paths = set()
    for section in config.sections():
        if not section.startswith("submodule "):
            continue

        if "path" not in config[section]:
            raise ValueError(f"Submodule section {section!r} in .gitmodules is missing required 'path'")

        submodule_path = Path(config[section]["path"].strip()).as_posix().rstrip("/")
        if submodule_path:
            submodule_paths.add(submodule_path)

    return submodule_paths


def relative_to_project(path, project_root=PROJECT_ROOT):
    """Return path relative to project_root, using POSIX separators."""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def is_in_submodule(relative_path, submodule_paths):
    """Return true if relative_path is inside a declared submodule path."""
    relative_path = relative_path.rstrip("/")
    return any(
        relative_path == submodule_path or relative_path.startswith(f"{submodule_path}/")
        for submodule_path in submodule_paths
    )


def iter_format_files(roots, extensions, project_root=PROJECT_ROOT):
    """Yield files under roots matching extensions, excluding Git submodules."""
    submodule_paths = load_submodule_paths(project_root)

    for root in roots:
        root_path = (project_root / root).resolve()
        if not root_path.exists():
            continue

        # Allow callers to pass an individual file as a root.
        if root_path.is_file():
            relative_path = relative_to_project(root_path, project_root)
            if not is_in_submodule(relative_path, submodule_paths) and root_path.name.endswith(extensions):
                yield relative_path
            continue

        for current_root, dirs, files in os.walk(root_path):
            current_path = Path(current_root)

            # `dirs` will be used in the next iteration of the loop by `os.walk()``.
            # Removing sub-module directories in-place will prevent them os.walk from
            # walking into them, pruning the search to non-submodule directories.
            dirs[:] = [
                dirname
                for dirname in dirs
                if not is_in_submodule(
                    relative_to_project(current_path / dirname, project_root), submodule_paths
                )
            ]

            # Yield paths relative to the project root so formatter output is
            # stable regardless of where the helper script itself lives.
            for filename in files:
                file_path = current_path / filename
                relative_path = relative_to_project(file_path, project_root)
                if not is_in_submodule(relative_path, submodule_paths) and filename.endswith(extensions):
                    yield relative_path
