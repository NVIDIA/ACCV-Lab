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

# Build ACCV-Lab documentation locally

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse flags
VERBOSE_FLAG=""
WARNING_FILE=""
PDF_MODE=0
OPEN_MODE=0
SPELLING_MODE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE_FLAG="--verbose"
            echo "Verbose mode enabled"
            ;;
        --warning-file)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: --warning-file requires a file path"
                exit 1
            fi
            WARNING_FILE="$2"
            shift
            ;;
        -p|--pdf)
            PDF_MODE=1
            ;;
        -o|--open)
            OPEN_MODE=1
            ;;
        -s|--spelling)
            SPELLING_MODE=1
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--verbose] [--warning-file PATH] [-p|--pdf] [-s|--spelling] [-o|--open]"
            echo "  Default builds HTML docs. Use --pdf to build PDF via LaTeX."
            echo "  Use --warning-file PATH to write Sphinx warnings to a dedicated file."
            echo "    Relative warning-file paths are resolved from the project root."
            echo "  Use --spelling to run spelling checks with sphinxcontrib-spelling."
            echo "  Use --open to auto-open the built HTML/PDF if supported."
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Use -h|--help for usage."
            exit 1
            ;;
    esac
    shift
done

echo "=========================="
echo "ACCV-Lab Documentation Build"
echo "=========================="

# Change to project root
cd "$PROJECT_ROOT"

# Plausibility check if we are in the right directory
if [ ! -f "namespace_packages_config.py" ]; then
    echo "Error: This script should be run from the ACCV-Lab root directory"
    exit 1
fi

# There should be a docs directory, containing the documentation source files
if [ ! -d "docs" ]; then
    echo "Error: docs directory not found"
    exit 1
fi

if [[ -n "$WARNING_FILE" && "$WARNING_FILE" != /* ]]; then
    WARNING_FILE="$PROJECT_ROOT/$WARNING_FILE"
fi

# Install documentation dependencies, which are listed in the requirements.txt file
# of the docs directory
echo "Installing documentation dependencies..."
pip install -r docs/requirements.txt

cd docs
make clean

SPHINXOPTS_ARGS="$VERBOSE_FLAG"
if [[ -n "$WARNING_FILE" ]]; then
    mkdir -p "$(dirname "$WARNING_FILE")"
    : > "$WARNING_FILE"
    SPHINXOPTS_ARGS="${SPHINXOPTS_ARGS:+$SPHINXOPTS_ARGS }-w $WARNING_FILE"
    echo "Sphinx warnings will be written to: $WARNING_FILE"
fi

if [[ "$PDF_MODE" -eq 1 && "$SPELLING_MODE" -eq 1 ]]; then
    echo "Error: --pdf and --spelling cannot be used together. Choose one."
    exit 1
fi

if [[ "$PDF_MODE" -eq 1 ]]; then
    echo "Building PDF documentation..."
    make generate
    make latexpdf SPHINXOPTS="$SPHINXOPTS_ARGS"
    echo ""
    echo "PDF build complete!"
    # Try to open the resulting PDF (handle common naming variants)
    if [[ -f "_build/latex/accvlab.pdf" ]]; then
        PDF_PATH="_build/latex/accvlab.pdf"
    elif [[ -f "_build/latex/ACCV-Lab.pdf" ]]; then
        PDF_PATH="_build/latex/ACCV-Lab.pdf"
    else
        PDF_PATH=""
    fi
    if [[ -n "$PDF_PATH" ]]; then
        echo "PDF located at: docs/$PDF_PATH"
        if [[ "$OPEN_MODE" -eq 1 ]]; then
            if command -v xdg-open > /dev/null; then
                echo "Opening PDF..."
                xdg-open "$PDF_PATH" || true
            elif command -v open > /dev/null; then
                echo "Opening PDF..."
                open "$PDF_PATH" || true
            fi
        fi
    else
        echo "PDF file not found in _build/latex (build may have failed)."
    fi
elif [[ "$SPELLING_MODE" -eq 1 ]]; then
    echo "Running spelling checks..."
    echo "Checking spelling dependencies..."
    python3 - <<'PY'
import sys

def fail(msg: str) -> None:
    print("Error: " + msg)
    sys.exit(1)

try:
    import sphinxcontrib.spelling  # noqa: F401
except Exception as e:
    fail(
        "sphinxcontrib-spelling is not available. "
        f"Original error: {e}."
    )

try:
    import enchant  # pyenchant
    try:
        # Force-check that the Enchant C library is usable
        enchant.Dict("en_US")
    except Exception as de:
        fail(
            "pyenchant is installed but the Enchant C library is not available/usable. "
            f"Original error: {de}.\n"
            "Install the system library (e.g., on Debian/Ubuntu: sudo apt install libenchant-2-2)"
        )
except Exception as e:
    fail(
        "pyenchant is not available. "
        f"Original error: {e}.\n"
        "Install it via: pip install pyenchant"
    )
PY
    # Ensure generated/mirrored docs are up to date before spelling
    make generate
    # Run the spelling builder
    make spelling SPHINXOPTS="$SPHINXOPTS_ARGS"
    echo ""
    echo "Spelling check complete!"
    # Point to common output location
    if [[ -f "_build/spelling/output.txt" ]]; then
        echo "Spelling report: docs/_build/spelling/output.txt"
        if [[ "$OPEN_MODE" -eq 1 ]]; then
            echo "Showing spelling report (tail):"
            tail -n 200 "_build/spelling/output.txt" || true
        fi
    else
        echo "Spelling output not found at _build/spelling/output.txt."
        echo "Please inspect the build logs above for details."
    fi
else
    echo "Building HTML documentation..."
    make html SPHINXOPTS="$SPHINXOPTS_ARGS"
    echo ""
    echo "Documentation build complete!"
    echo "Open docs/_build/html/index.html in your browser to view the documentation."
    # Optional: automatically open the documentation (only if requested)
    if [[ "$OPEN_MODE" -eq 1 ]]; then
        if command -v xdg-open > /dev/null; then
            echo "Opening documentation in your default browser..."
            xdg-open _build/html/index.html || true
        elif command -v open > /dev/null; then
            echo "Opening documentation in your default browser..."
            open _build/html/index.html || true
        fi
    fi
fi

cd ..