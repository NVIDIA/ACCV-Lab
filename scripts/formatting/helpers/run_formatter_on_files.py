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

"""Run a formatter on files under selected roots, excluding Git submodules."""

import argparse

from format_file_utils import iter_format_files
from formatter_command_utils import run_formatter_in_batches


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover files under one or more roots, exclude paths declared as "
            "Git submodules, and run a formatter command on the resulting files."
        )
    )
    parser.add_argument(
        "--extension",
        action="append",
        required=True,
        help="File extension to include, such as .py or .cpp. Can be repeated.",
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="File or directory to scan, relative to the project root. Can be repeated.",
    )
    parser.add_argument(
        "--empty-message",
        default="No matching files found",
        help="Message to print when no files match after submodule exclusions.",
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
    files = list(iter_format_files(args.root, tuple(args.extension)))

    if not files:
        print(args.empty_message)
        return

    run_formatter_in_batches(args.formatter_command, files, args.batch_size)


if __name__ == "__main__":
    main()
