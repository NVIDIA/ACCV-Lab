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

# Script to format a single namespace package with Black
# Usage: ./black_format_subpackage.sh <package_name>
# Example: ./black_format_subpackage.sh examples

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

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

echo "Formatting namespace package: $PACKAGE"

# Format Python files while excluding directories declared in .gitmodules.
echo "Formatting packages/$PACKAGE/..."
if [ -d "packages/$PACKAGE" ]; then
    python3 "$SCRIPT_DIR/helpers/run_formatter_on_files.py" \
        --extension .py \
        --root "packages/$PACKAGE" \
        --empty-message "  No Python files found in packages/$PACKAGE" \
        -- black
fi

echo "Formatting of namespace package '$PACKAGE' completed successfully!" 