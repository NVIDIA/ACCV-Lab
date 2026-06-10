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

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class PackageDocsContext:
    project_root: Path
    namespace_package: str
    package_name: str
    package_root: Path
    docs_root: Path
    generated_dir: Path


HookFunction = Callable[[PackageDocsContext], None]
_GENERATED_ASSET_GITIGNORE = "*\n"


def _load_hook_module(hook_path: Path, package_name: str) -> ModuleType:
    # Temporary module name for the imported hook.
    module_name = f"_accvlab_docs_assets_{package_name}"

    # Import
    spec = importlib.util.spec_from_file_location(module_name, hook_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for docs asset hook: {hook_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _get_hook_function(module: ModuleType, hook_path: Path) -> HookFunction:
    hook_function = getattr(module, "generate_docs_assets", None)
    if not callable(hook_function):
        raise AttributeError(
            f"Docs asset hook must define a callable generate_docs_assets(context): {hook_path}"
        )
    return hook_function


def _prepare_generated_dir(context: PackageDocsContext) -> None:
    """Create the package's generated docs asset directory and keep it untracked."""
    context.generated_dir.mkdir(parents=True, exist_ok=True)
    (context.generated_dir / ".gitignore").write_text(_GENERATED_ASSET_GITIGNORE, encoding="utf-8")


def _build_context(project_root: Path, namespace_package: str) -> PackageDocsContext:
    package_name = namespace_package.split(".")[-1]
    package_root = project_root / "packages" / package_name
    docs_root = package_root / "docs"
    generated_dir = docs_root / "_generated"
    ctx = PackageDocsContext(
        project_root=project_root,
        namespace_package=namespace_package,
        package_name=package_name,
        package_root=package_root,
        docs_root=docs_root,
        generated_dir=generated_dir,
    )
    return ctx


def _generate_assets_for_package(
    *,
    project_root: Path,
    namespace_package: str,
    verbose: bool,
) -> bool:
    context = _build_context(project_root, namespace_package)
    hook_path = context.docs_root / "_on_doc_generation.py"
    if not hook_path.exists():
        if verbose:
            print(f"No docs asset hook for {context.package_name}")
        return False

    if verbose:
        print(f"Running docs asset hook for {context.package_name}: {hook_path}")
    module = _load_hook_module(hook_path, context.package_name)
    _prepare_generated_dir(context)
    hook_function = _get_hook_function(module, hook_path)
    hook_function(context)
    print(f"Generated docs assets for {context.package_name}")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional package-local documentation asset generation hooks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--package",
        dest="package_names",
        action="append",
        help="Package name to process, such as lane_helpers. Can be passed more than once.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    docs_dir = Path(__file__).resolve().parent
    project_root = docs_dir.parent
    sys.path.insert(0, str(project_root))

    try:
        from namespace_packages_config import NAMESPACE_PACKAGES
    except ImportError as exc:
        print(
            f"Error: Could not import NAMESPACE_PACKAGES from namespace_packages_config.py: {exc}",
            file=sys.stderr,
        )
        return 1

    package_filter = set(args.package_names or [])
    namespace_packages = [
        namespace_package
        for namespace_package in NAMESPACE_PACKAGES
        if not package_filter or namespace_package.split(".")[-1] in package_filter
    ]
    if package_filter and len(namespace_packages) != len(package_filter):
        found_package_names = {namespace_package.split(".")[-1] for namespace_package in namespace_packages}
        missing_package_names = sorted(package_filter - found_package_names)
        print(f"Error: Unknown namespace package(s): {', '.join(missing_package_names)}", file=sys.stderr)
        return 1

    hook_count = 0
    for namespace_package in namespace_packages:
        package_name = namespace_package.split(".")[-1]
        try:
            hook_ran = _generate_assets_for_package(
                project_root=project_root,
                namespace_package=namespace_package,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"Error: docs asset generation failed for {package_name}: {exc}", file=sys.stderr)
            return 1
        if hook_ran:
            hook_count += 1

    if args.verbose:
        print(f"Ran {hook_count} package docs asset hook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
