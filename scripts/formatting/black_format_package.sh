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

# Script to format all Python code with Black
# Usage: ./format_all.sh [--skip-subpackages]
# 
# By default, formats:
# - Root Python files (namespace_packages_config.py, etc.)
# - Docs Python files (docs/**/*.py)
# - Scripts Python files (scripts/**/*.py)
# - Common accvlab code (accvlab/__init__.py, etc.)
# - Common build_config code (build_config/_helpers/, build_config/__init__.py)
# - All subpackages (both in accvlab/ and build_config/subpackages/)
#
# With --skip-subpackages:
# - Only formats root files and common code, skips individual subpackages

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
            echo "Format Python code in the project with Black."
            echo ""
            echo "By default, formats only common code (root files, docs, scripts, common accvlab, common build_config)."
            echo ""
            echo "Options:"
            echo "  --include-subpackages    Also format individual subpackages"
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

# Change to project root
cd "$PROJECT_ROOT"

echo "Formatting Python code with Black..."

# Format root Python files
echo "Formatting root Python files..."
black *.py

# Format common accvlab code (excluding subpackage directories)
echo "Formatting common accvlab code..."
if [ -d "accvlab" ]; then
    # Format Python files in accvlab root (but not subdirectories yet)
    find accvlab/ -maxdepth 1 -name "*.py" -exec black {} \;
fi

# Format common build_config code
echo "Formatting common build_config code..."
# Format all build_config files except subpackages (which are handled separately)
find build_config/ -name "*.py" -not -path "build_config/subpackages/*" -exec black {} \;

# Format docs Python files if they exist
if [ -d "docs" ]; then
    echo "Formatting docs Python files..."
    find docs/ -name "*.py" -exec black {} \;
fi

# Format scripts Python files if they exist
if [ -d "scripts" ]; then
    echo "Formatting scripts Python files..."
    find scripts/ -name "*.py" -exec black {} \;
fi

if [ "$INCLUDE_SUBPACKAGES" = true ]; then
    # Format all namespace packages, excluding directories declared in .gitmodules.
    echo "Formatting namespace packages..."
    python3 "$SCRIPT_DIR/helpers/run_formatter_on_namespace_packages.py" \
        --extension .py \
        --language-name Python \
        -- black
else
    echo "Skipping namespace packages (use --include-subpackages to format them)"
fi

echo "Python code formatting completed successfully!" 