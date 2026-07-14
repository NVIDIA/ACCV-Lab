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

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _format_boxed_message(lines: Sequence[str]) -> str:
    content_width = max(len(line) for line in lines)
    border = "#" * (content_width + 4)
    boxed_lines = [border]
    boxed_lines.extend(f"# {line:<{content_width}} #" for line in lines)
    boxed_lines.append(border)
    return "\n".join(boxed_lines)


_MISSING_BUILD_CONFIG_MESSAGE = _format_boxed_message(
    [
        "Missing or outdated build dependency: accvlab-build-config.",
        "",
        "The C++ unit test runner uses ACCV-Lab's shared build configuration helpers. Install",
        "or update them in the active environment and retry:",
        "",
        f'    pip install {PROJECT_ROOT / "build_config"}',
        "",
        "Alternatively, run this script from a full ACCV-Lab checkout after installing the",
        "build_config helper package.",
    ]
)


try:
    from accvlab_build_config.helpers import (
        CppUnitTestPackage,
        YamlConfigError,
        build_cmake_args,
        load_cpp_unit_test_package_configs,
        select_cpp_unit_test_packages,
    )
except ModuleNotFoundError as exc:
    if exc.name != "accvlab_build_config":
        raise
    print(_MISSING_BUILD_CONFIG_MESSAGE, file=sys.stderr)
    sys.exit(1)
except ImportError as exc:
    print(_MISSING_BUILD_CONFIG_MESSAGE, file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


def load_configured_package_names() -> List[str]:
    config_path = PROJECT_ROOT / "namespace_packages_config.py"
    spec = importlib.util.spec_from_file_location("accvlab_namespace_packages_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load namespace package config: {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return _get_package_names_from_module(module)


def _get_package_names_from_module(module: ModuleType) -> List[str]:
    get_package_names = getattr(module, "get_package_names", None)
    if callable(get_package_names):
        package_names = get_package_names()
    else:
        namespace_packages = getattr(module, "NAMESPACE_PACKAGES", None)
        if not isinstance(namespace_packages, list):
            raise SystemExit("Failed to load namespace package config: missing `get_package_names()`.")
        package_names = [namespace_package.split(".")[-1] for namespace_package in namespace_packages]

    if not isinstance(package_names, list) or not all(isinstance(name, str) for name in package_names):
        raise SystemExit("Failed to load namespace package config: package names must be a list of strings.")
    return package_names


def run_command(command: Sequence[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def configure_and_run_package(package: CppUnitTestPackage, *, configure_only: bool) -> None:
    if not package.cmake_source_dir.is_dir():
        raise SystemExit(
            f"Configured CMake source directory does not exist for {package.name}: "
            f"{package.cmake_source_dir}"
        )

    package.build_dir.mkdir(parents=True, exist_ok=True)

    cmake_configure_command = [
        "cmake",
        "-S",
        str(package.cmake_source_dir),
        "-B",
        str(package.build_dir),
        *build_cmake_args(cuda_arch_strategy=package.cuda_arch_strategy),
        *package.cmake_args,
    ]
    run_command(cmake_configure_command, cwd=PROJECT_ROOT)

    if configure_only:
        return

    cmake_build_command = [
        "cmake",
        "--build",
        str(package.build_dir),
        "--target",
        package.test_target,
    ]
    run_command(cmake_build_command, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACCV-Lab native C++/CUDA unit test targets.")
    parser.add_argument(
        "packages",
        nargs="*",
        help="Package names to test, or `all` to run every configured package.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        help=(
            "Path to a C++ unit test YAML config. May be passed multiple times. "
            "By default, the runner checks configured namespace packages for cpp_unit_tests.yaml."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured packages and exit.",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Configure selected packages without building test targets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured_package_names = load_configured_package_names()
    try:
        package_configs = load_cpp_unit_test_package_configs(
            PROJECT_ROOT,
            config_paths=args.config,
            package_names=configured_package_names,
        )
    except YamlConfigError as exc:
        raise SystemExit(f"Failed to load C++ unit test config: {exc}") from exc

    if args.list:
        if not package_configs:
            print("No C++ unit test packages are configured.")
            return 0
        for package in package_configs.values():
            print(
                f"{package.name}: source={package.cmake_source_dir.relative_to(PROJECT_ROOT)} "
                f"build={package.build_dir.relative_to(PROJECT_ROOT)} target={package.test_target}"
            )
        return 0

    try:
        selected_packages = select_cpp_unit_test_packages(args.packages, package_configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for package in selected_packages:
        print("", flush=True)
        print(f"=== C++ unit tests: {package.name} ===", flush=True)
        configure_and_run_package(package, configure_only=args.configure_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
