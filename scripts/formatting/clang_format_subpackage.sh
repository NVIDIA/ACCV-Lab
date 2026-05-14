#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script to format C++ code for a single namespace package with clang-format
# Usage: ./clang_format_subpackage.sh <package_name>
# Example: ./clang_format_subpackage.sh examples

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLANG_FORMAT_CONFIG="$PROJECT_ROOT/.clang-format" 

# Change to project root
cd "$PROJECT_ROOT"

# Verify we can find the clang-format config file
if [ ! -f "$CLANG_FORMAT_CONFIG" ]; then
    echo "Error: .clang-format file not found at $CLANG_FORMAT_CONFIG"
    echo "Please ensure the .clang-format file exists in the project root directory"
    exit 1
fi

if [ $# -eq 0 ] || [ $# -gt 1 ] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <package_name>"
    if [ $# -gt 1 ]; then
        echo "Error: Unexpected extra arguments: ${@:2}"
    elif [[ "$1" == -* ]]; then
        echo "Error: Unknown option: $1"
    fi
    echo "Available namespace packages:"
    python3 "$SCRIPT_DIR/helpers/list_namespace_packages.py"
    exit 1
fi

PACKAGE=$1

# Check if package exists
if [ ! -d "packages/$PACKAGE" ]; then
    echo "Error: Namespace package 'packages/$PACKAGE' does not exist"
    exit 1
fi

echo "Formatting C++ code for namespace package: $PACKAGE"
echo "Using configuration: $CLANG_FORMAT_CONFIG"

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Warning: clang-format is not installed or not in PATH. Skipping C++ formatting."
    echo "To install clang-format, use your package manager (e.g., apt install clang-format)"
    return 0 2>/dev/null || exit 0
fi

# Format C++ files while excluding directories declared in .gitmodules.
echo "Formatting C++ files in packages/$PACKAGE/..."
if [ -d "packages/$PACKAGE" ]; then
    python3 "$SCRIPT_DIR/helpers/run_formatter_on_files.py" \
        --extension .cpp \
        --extension .cc \
        --extension .c \
        --extension .cu \
        --extension .hpp \
        --extension .h \
        --extension .cuh \
        --root "packages/$PACKAGE" \
        --empty-message "  No C++ files found in packages/$PACKAGE" \
        -- clang-format -style=file -fallback-style=none -i
fi

echo "C++ formatting for namespace package '$PACKAGE' completed successfully!"
