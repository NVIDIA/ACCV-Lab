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

# Script to format all C++ code with clang-format
# Usage: ./clang_format_all.sh [--skip-subpackages]
# 
# By default, formats all subpackages (both in accvlab/ and tests/)
# Since C++ code only exists in subpackages and their tests, there's no "common" C++ code
#
# With --skip-subpackages:
# - Does nothing (since there's no common C++ code to format)

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

INCLUDE_SUBPACKAGES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --include-subpackages)
            INCLUDE_SUBPACKAGES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--include-subpackages]"
            echo ""
            echo "Format C++ code in the project with clang-format."
            echo ""
            echo "By default, does nothing since C++ code only exists in subpackages."
            echo ""
            echo "Options:"
            echo "  --include-subpackages    Format subpackages (where C++ code exists)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Formatting C++ code with clang-format..."

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Warning: clang-format is not installed or not in PATH. Skipping C++ formatting."
    echo "To install clang-format, use your package manager (e.g., apt install clang-format)"
    exit 0
fi

if [ "$INCLUDE_SUBPACKAGES" = true ]; then
    # Format all namespace packages, excluding directories declared in .gitmodules.
    echo "Formatting namespace packages..."
    echo "Using nearest .clang-format (auto-discovery)"
    python3 "$SCRIPT_DIR/helpers/run_formatter_on_namespace_packages.py" \
        --extension .cpp \
        --extension .cc \
        --extension .c \
        --extension .cu \
        --extension .hpp \
        --extension .h \
        --extension .cuh \
        --language-name "C++" \
        -- clang-format -style=file -fallback-style=none -i
else
    echo "Skipping namespace packages (use --include-subpackages to format them)"
    echo "Note: C++ code only exists in namespace packages"
fi

echo "C++ code formatting completed successfully!"