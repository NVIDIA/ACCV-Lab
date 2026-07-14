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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .cmake_args import CUDA_ARCH_STRATEGY_CMAKE, CUDA_ARCH_STRATEGY_TORCH
from .yaml_config import YamlConfigError, load_yaml_config

_CPP_UNIT_TEST_CONFIG_NAME = "cpp_unit_tests.yaml"
_VALID_CUDA_ARCH_STRATEGIES = frozenset({CUDA_ARCH_STRATEGY_CMAKE, CUDA_ARCH_STRATEGY_TORCH})


@dataclass(frozen=True)
class CppUnitTestPackage:
    name: str
    cmake_source_dir: Path
    build_dir: Path
    cuda_arch_strategy: str
    test_option: str
    cmake_args: List[str]
    test_target: str
    config_path: Path


def _discover_cpp_unit_test_config_paths(
    project_root: Path,
    *,
    package_names: Optional[Sequence[str]] = None,
    package_config_name: str = _CPP_UNIT_TEST_CONFIG_NAME,
) -> List[Path]:
    packages_dir = project_root / "packages"
    if not packages_dir.is_dir():
        return []
    if package_names is not None:
        return [
            config_path
            for package_name in package_names
            for config_path in [packages_dir / package_name / package_config_name]
            if config_path.is_file()
        ]
    return sorted(
        package_dir / package_config_name
        for package_dir in packages_dir.iterdir()
        if package_dir.is_dir() and (package_dir / package_config_name).is_file()
    )


def load_cpp_unit_test_package_configs(
    project_root: Path,
    *,
    config_paths: Optional[Sequence[Path]] = None,
    package_names: Optional[Sequence[str]] = None,
    package_config_name: str = _CPP_UNIT_TEST_CONFIG_NAME,
) -> Dict[str, CppUnitTestPackage]:
    if config_paths is not None:
        paths = list(config_paths)
    else:
        paths = list(
            _discover_cpp_unit_test_config_paths(
                project_root,
                package_names=package_names,
                package_config_name=package_config_name,
            )
        )

    package_configs: Dict[str, CppUnitTestPackage] = {}
    for config_path in paths:
        for package in _load_config_path(project_root, Path(config_path)):
            if package.name in package_configs:
                existing = package_configs[package.name]
                raise YamlConfigError(
                    f"Duplicate C++ unit test package {package.name!r}: "
                    f"{existing.config_path} and {package.config_path}."
                )
            package_configs[package.name] = package
    return package_configs


def select_cpp_unit_test_packages(
    requested_packages: Sequence[str],
    package_configs: Dict[str, CppUnitTestPackage],
) -> List[CppUnitTestPackage]:
    if not requested_packages:
        raise ValueError("No package selected. Use `all`, `--list`, or one of the configured package names.")

    if "all" in requested_packages:
        if len(requested_packages) > 1:
            raise ValueError("Use either `all` or explicit package names, not both.")
        return list(package_configs.values())

    unknown_packages = [name for name in requested_packages if name not in package_configs]
    if unknown_packages:
        known = ", ".join(package_configs) if package_configs else "<none>"
        raise ValueError(f"Unknown package(s): {', '.join(unknown_packages)}. Configured packages: {known}.")

    return [package_configs[name] for name in requested_packages]


def _load_config_path(project_root: Path, config_path: Path) -> List[CppUnitTestPackage]:
    resolved_config_path = config_path.resolve()
    try:
        config = load_yaml_config(resolved_config_path)
    except YamlConfigError:
        raise

    package_entries = _package_entries_from_config(config, config_path=resolved_config_path)
    return [
        _package_from_entry(project_root, resolved_config_path, package_entry, index)
        for index, package_entry in enumerate(package_entries)
    ]


def _package_entries_from_config(config: Dict[str, Any], *, config_path: Path) -> List[Dict[str, Any]]:
    if "packages" in config:
        packages = config["packages"]
        if not isinstance(packages, list):
            raise YamlConfigError(f"{config_path}: top-level `packages` must be a list.")
        if not all(isinstance(package, dict) for package in packages):
            raise YamlConfigError(f"{config_path}: each `packages` entry must be a mapping.")
        return packages
    return [config]


def _package_from_entry(
    project_root: Path,
    config_path: Path,
    package: Dict[str, Any],
    index: int,
) -> CppUnitTestPackage:
    package_dir = config_path.parent
    default_name = _derive_package_name_from_config_path(project_root, config_path)

    name = package.get("name", default_name)
    cmake_source_dir = package.get("cmake_source_dir", "ext_impl")
    cuda_arch_strategy = package.get("cuda_arch_strategy")
    test_option = package.get("test_option")
    test_target = package.get("test_target")

    missing_fields = [
        field_name
        for field_name, field_value in (
            ("name", name),
            ("cmake_source_dir", cmake_source_dir),
            ("cuda_arch_strategy", cuda_arch_strategy),
            ("test_option", test_option),
            ("test_target", test_target),
        )
        if not isinstance(field_value, str) or not field_value
    ]
    if missing_fields:
        raise YamlConfigError(
            f"{config_path}: package entry {index} has missing or invalid fields: "
            f"{', '.join(missing_fields)}."
        )

    if cuda_arch_strategy not in _VALID_CUDA_ARCH_STRATEGIES:
        valid = ", ".join(sorted(_VALID_CUDA_ARCH_STRATEGIES))
        raise YamlConfigError(
            f"{config_path}: package entry {index} field `cuda_arch_strategy` must be one of: {valid}."
        )

    cmake_args = _as_string_list(package.get("cmake_args"), field_name="cmake_args", package_name=name)

    return CppUnitTestPackage(
        name=name,
        cmake_source_dir=_resolve_config_path(cmake_source_dir, base_dir=package_dir),
        build_dir=project_root / "build" / "cpp_unit_tests" / name,
        cuda_arch_strategy=cuda_arch_strategy,
        test_option=test_option,
        cmake_args=[f"-D{test_option}=ON", *cmake_args],
        test_target=test_target,
        config_path=config_path,
    )


def _derive_package_name_from_config_path(project_root: Path, config_path: Path) -> str:
    try:
        relative_config_path = config_path.relative_to(project_root)
    except ValueError:
        return config_path.parent.name

    parts = relative_config_path.parts
    if len(parts) >= 3 and parts[0] == "packages":
        return parts[1]
    return config_path.parent.name


def _resolve_config_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def _as_string_list(value: Any, *, field_name: str, package_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise YamlConfigError(f"Package {package_name!r} field {field_name!r} must be a list of strings.")
    return value
