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

"""Run a formatter across all namespace packages, excluding Git submodules."""

import argparse
import sys

from format_file_utils import PROJECT_ROOT, iter_format_files
from formatter_command_utils import run_formatter_in_batches

sys.path.insert(0, str(PROJECT_ROOT))

from namespace_packages_config import get_package_names  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a formatter for every namespace package under packages/, while "
            "excluding directories declared as Git submodules."
        )
    )
    parser.add_argument(
        "--extension",
        action="append",
        required=True,
        help="File extension to include, such as .py or .cpp. Can be repeated.",
    )
    parser.add_argument(
        "--language-name",
        required=True,
        help="Human-readable language name for progress messages, such as Python or C++.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum number of files to pass to one formatter invocation.",
    )
    parser.add_argument(
        "formatter_command",
        nargs=argparse.REMAINDER,
        help="Formatter command to run after '--', for example: -- black",
    )
    args = parser.parse_args()

    if args.formatter_command and args.formatter_command[0] == "--":
        args.formatter_command = args.formatter_command[1:]

    if not args.formatter_command:
        parser.error("formatter command is required after '--'")

    return args


def main():
    args = parse_args()
    packages = get_package_names()

    if not packages:
        print("  No namespace packages found")
        return

    for package in packages:
        package_path = f"packages/{package}"
        print(f"  Formatting {args.language_name} in namespace package: {package}")

        files = list(iter_format_files([package_path], tuple(args.extension)))
        if not files:
            print(f"    No {args.language_name} files found in {package_path}")
            continue

        run_formatter_in_batches(args.formatter_command, files, args.batch_size)


if __name__ == "__main__":
    main()
