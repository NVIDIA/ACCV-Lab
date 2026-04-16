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

# Unified script to install or build wheels for all namespace packages
# Usage: ./package_manager.sh {install|wheel} [OPTIONS]
# 
# Install mode: ./package_manager.sh install [-e] [--optional] [--with-build-isolation]
#   -e  Install in editable mode
#   --optional  Install with optional dependencies
#   --with-build-isolation  Enable pip build isolation so build-time dependencies can be installed automatically
# 
# Wheel mode: ./package_manager.sh wheel [-o OUTPUT_DIR] [--with-deps] [--no-index] [--optional] [--with-build-isolation]
#   -o OUTPUT_DIR  Specify output directory for wheels (default: ./wheels)
#   --with-deps  (Compatibility alias) Explicitly request dependency resolution when building wheels (default behavior)
#   --no-index  Don't use PyPI index (use current environment only)
#   --optional  Build wheels with optional dependencies
#   --with-build-isolation  Enable pip build isolation so build-time dependencies can be installed automatically

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to show usage
show_usage() {
    echo "Usage: $0 {install|wheel} [OPTIONS]"
    echo ""
    echo "Install mode:"
    echo "  $0 install [-e] [--optional] [--with-build-isolation]"
    echo "    -e  Install in editable mode"
    echo "    --optional  Install with optional dependencies"
    echo "    --with-build-isolation  Enable pip build isolation"
    echo "    Default: Install in regular (non-editable) mode and without build isolation '--no-build-isolation'"
    echo ""
    echo "Wheel mode:"
    echo "  $0 wheel [-o OUTPUT_DIR] [--with-deps] [--no-index] [--optional] [--with-build-isolation]"
    echo "    -o OUTPUT_DIR  Specify output directory for wheels (default: ./wheels)"
    echo "    --with-deps  (Compatibility alias) Explicitly request dependency resolution when building wheels (default behavior)"
    echo "    --no-index  Don't use PyPI index (use current environment only)"
    echo "    --optional  Build wheels with optional dependencies"
    echo "    --with-build-isolation  Enable pip build isolation to allow installing build-time dependencies"
    echo "    Default: Build wheels with '--no-build-isolation' and '--no-deps'"
    echo ""
    echo "This script manages all namespace packages defined in namespace_packages_config.py"
    echo ""
    echo "Examples:"
    echo "  $0 install                                # Install all packages with basic dependencies"
    echo "  $0 install -e                             # Install all packages in editable mode"
    echo "  $0 install --optional                     # Install all packages with optional dependencies"
    echo "  $0 install --with-build-isolation         # Install all packages with build isolation"
    echo "  $0 install -e --optional                  # Install all packages in editable mode with optional dependencies"
    echo "  $0 wheel                                  # Build wheels for all packages"
    echo "  $0 wheel --with-deps                      # Build include the wheels of the dependencies (default git wheel behavior)"
    echo "  $0 wheel --optional                       # Build wheels with optional dependencies"
    echo "  $0 wheel -o /tmp                          # Build wheels in /tmp directory"
    echo "  $0 wheel --no-index                       # Build wheels using only current environment"
    echo "  $0 wheel --with-build-isolation           # Build wheels with build isolation"
}

# Check if mode is provided
if [[ $# -eq 0 ]]; then
    echo "Error: No mode specified"
    show_usage
    exit 1
fi

MODE="$1"
shift

# Validate mode
if [[ "$MODE" != "install" && "$MODE" != "wheel" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'install' or 'wheel'"
    show_usage
    exit 1
fi

# Parse command line arguments
EDITABLE_MODE=false
OUTPUT_DIR=""
WITH_DEPS=false
NO_INDEX=false
OPTIONAL_DEPS=false
WITH_BUILD_ISOLATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e)
            if [[ "$MODE" == "wheel" ]]; then
                echo "Error: -e (editable) is not supported in wheel mode"
                show_usage
                exit 1
            fi
            EDITABLE_MODE=true
            shift
            ;;
        -o)
            if [[ "$MODE" != "wheel" ]]; then
                echo "Error: -o option is only valid for wheel mode"
                show_usage
                exit 1
            fi
            if [[ -n "$2" && "$2" != -* ]]; then
                OUTPUT_DIR="$2"
                shift 2
            else
                echo "Error: -o requires an output directory argument"
                show_usage
                exit 1
            fi
            ;;
        --with-deps)
            if [[ "$MODE" != "wheel" ]]; then
                echo "Error: --with-deps option is only valid for wheel mode"
                show_usage
                exit 1
            fi
            WITH_DEPS=true
            shift
            ;;
        --no-index)
            if [[ "$MODE" != "wheel" ]]; then
                echo "Error: --no-index option is only valid for wheel mode"
                show_usage
                exit 1
            fi
            NO_INDEX=true
            shift
            ;;
        --optional)
            OPTIONAL_DEPS=true
            shift
            ;;
        --with-build-isolation)
            WITH_BUILD_ISOLATION=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Determine build isolation behavior (default: disable build isolation via --no-build-isolation)
if [[ "$WITH_BUILD_ISOLATION" == true ]]; then
    BUILD_ISOLATION_FLAG=""
else
    BUILD_ISOLATION_FLAG="--no-build-isolation"
fi

# Set default output directory for wheel mode
if [[ "$MODE" == "wheel" && -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/wheels"
fi

# Create output directory for wheel mode if it doesn't exist
if [[ "$MODE" == "wheel" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Resolve the Python interpreter used for all package operations
export PYTHON_EXECUTABLE=$(command -v python)

echo "Using Python: $PYTHON_EXECUTABLE"
echo "Using Pip: $($PYTHON_EXECUTABLE -m pip --version)"
echo "Python version: $($PYTHON_EXECUTABLE --version)"

if [[ "$MODE" == "wheel" ]]; then
    echo "Output directory: $OUTPUT_DIR"
    if [[ "$NO_INDEX" == true ]]; then
        echo "PyPI index: disabled (using current environment only)"
    fi
fi

if [[ "$EDITABLE_MODE" == true ]]; then
    echo "Mode: $MODE (editable)"
else
    echo "Mode: $MODE (regular)"
fi

if [[ "$OPTIONAL_DEPS" == true ]]; then
    echo "Optional dependencies: enabled"
else
    echo "Optional dependencies: disabled"
fi

echo ""
echo "Installing helper package: build_config"

# Always install the build_config helper package first (needed by other packages)
HELPER_DIR="build_config"
if [ -f "$HELPER_DIR/setup.py" ]; then
    echo ""
    echo "Installing helper package: $HELPER_DIR (required for other packages)"
    HELPER_PATH="$PROJECT_ROOT/$HELPER_DIR"

    if [[ "$EDITABLE_MODE" == true ]]; then
        echo "  Installing helper in editable mode..."
        $PYTHON_EXECUTABLE -m pip install -e "$HELPER_PATH" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install helper package"; exit 1; }
    else
        echo "  Installing helper in regular mode..."
        $PYTHON_EXECUTABLE -m pip install "$HELPER_PATH" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install helper package"; exit 1; }
    fi

    echo "  ✓ Successfully installed helper package"
else
    echo "  ✗ Helper package setup.py not found in $HELPER_DIR, aborting."
    exit 1
fi

# Build pip command with appropriate flags as an argv array.
# This avoids `eval` and keeps paths with spaces intact.
build_pip_command() {
    PIP_COMMAND=("$PYTHON_EXECUTABLE" -m pip)
    PIP_COMMAND+=("$@")

    if [[ "$MODE" == "wheel" ]]; then
        PIP_COMMAND+=(--wheel-dir "$OUTPUT_DIR")
        if [[ "$NO_INDEX" == true ]]; then
            PIP_COMMAND+=(--no-index)
        fi
        # In wheel mode, control dependency resolution based on WITH_DEPS (analogous to WITH_BUILD_ISOLATION)
        # Default behavior (WITH_DEPS=false) is to avoid resolving dependencies with '--no-deps'
        if [[ "$WITH_DEPS" == false ]]; then
            PIP_COMMAND+=(--no-deps)
        fi
    fi

    if [[ -n "$BUILD_ISOLATION_FLAG" ]]; then
        PIP_COMMAND+=("$BUILD_ISOLATION_FLAG")
    fi
}

# For wheel mode, also build the helper package wheel
if [[ "$MODE" == "wheel" ]]; then
    echo ""
    echo "Building helper package wheel: build_config"
    cd "$HELPER_DIR"
    
    echo "  Building wheel..."
    build_pip_command wheel .
    "${PIP_COMMAND[@]}" || { echo "  ✗ Failed to build helper package wheel"; exit 1; }
    
    cd - > /dev/null
    echo "  ✓ Successfully built helper package wheel"
fi

echo ""
if [[ "$MODE" == "install" ]]; then
    echo "Installing namespace packages..."
else
    echo "Building namespace package wheels..."
fi

# Get the list of namespace packages from the config
NAMESPACE_PACKAGES=$($PYTHON_EXECUTABLE -c "
from namespace_packages_config import get_namespace_packages
packages = get_namespace_packages()
for pkg in packages:
    print(pkg)
")

# Count packages for progress tracking
TOTAL_PACKAGES=$(echo "$NAMESPACE_PACKAGES" | wc -l)
CURRENT_PACKAGE=0

# Process each namespace package
for pkg in $NAMESPACE_PACKAGES; do
    CURRENT_PACKAGE=$((CURRENT_PACKAGE + 1))
    echo ""
    if [[ "$MODE" == "install" ]]; then
        echo "[$CURRENT_PACKAGE/$TOTAL_PACKAGES] Installing namespace package: $pkg"
    else
        echo "[$CURRENT_PACKAGE/$TOTAL_PACKAGES] Building wheel for namespace package: $pkg"
    fi
    
    # Extract package name (last part after the dot)
    PACKAGE_NAME=$(echo "$pkg" | sed 's/.*\.//')
    PACKAGE_DIR="packages/$PACKAGE_NAME"
    PACKAGE_PATH="$PROJECT_ROOT/$PACKAGE_DIR"
    
    # Check if package directory exists
    if [ ! -d "$PACKAGE_DIR" ]; then
        echo "  Warning: Package directory '$PACKAGE_DIR' not found, skipping..."
        continue
    fi
    
    # Check if setup.py exists
    if [ ! -f "$PACKAGE_DIR/setup.py" ]; then
        echo "  Warning: setup.py not found in '$PACKAGE_DIR', skipping..."
        continue
    fi
    
    if [[ "$MODE" == "install" ]]; then
        if [[ "$EDITABLE_MODE" == true ]]; then
            if [[ "$OPTIONAL_DEPS" == true ]]; then
                echo "  Installing in editable mode with optional dependencies..."
                $PYTHON_EXECUTABLE -m pip install -e "$PACKAGE_PATH[optional]" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install $pkg"; exit 1; }
            else
                echo "  Installing in editable mode..."
                $PYTHON_EXECUTABLE -m pip install -e "$PACKAGE_PATH" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install $pkg"; exit 1; }
            fi
        else
            if [[ "$OPTIONAL_DEPS" == true ]]; then
                echo "  Installing in regular mode with optional dependencies..."
                $PYTHON_EXECUTABLE -m pip install "$PACKAGE_PATH[optional]" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install $pkg"; exit 1; }
            else
                echo "  Installing in regular mode..."
                $PYTHON_EXECUTABLE -m pip install "$PACKAGE_PATH" $BUILD_ISOLATION_FLAG || { echo "  ✗ Failed to install $pkg"; exit 1; }
            fi
        fi
        echo "  ✓ Successfully installed $pkg"
    elif [[ "$MODE" == "wheel" ]]; then
        cd "$PACKAGE_DIR"
        if [[ "$OPTIONAL_DEPS" == true ]]; then
            echo "  Building wheel with optional dependencies..."
            build_pip_command wheel ".[optional]"
            "${PIP_COMMAND[@]}" || { echo "  ✗ Failed to build wheel for $pkg"; exit 1; }
        else
            echo "  Building wheel..."
            build_pip_command wheel .
            "${PIP_COMMAND[@]}" || { echo "  ✗ Failed to build wheel for $pkg"; exit 1; }
        fi
        cd - > /dev/null
        echo "  ✓ Successfully built wheel for $pkg"
    fi
done

echo ""
if [[ "$MODE" == "install" ]]; then
    echo "Installation complete!"
    echo "Testing imports..."
    
    # Test importing each namespace package
    for pkg in $NAMESPACE_PACKAGES; do
        echo "Testing import: $pkg"
        $PYTHON_EXECUTABLE -c "import $pkg; print('  ✓ $pkg imported successfully!')" 2>/dev/null || {
            echo "  ✗ Failed to import $pkg"
            exit 1
        }
    done
    
    echo ""
    echo "All namespace packages installed successfully and can be imported!"
else
    echo "Wheel building complete!"
    echo "Wheels saved to: $OUTPUT_DIR"
    
    # List the generated wheels
    echo ""
    echo "Generated wheels:"
    ls -la "$OUTPUT_DIR"/*.whl 2>/dev/null || echo "  No wheel files found in output directory"
    
    echo ""
    echo "All namespace package wheels built successfully!"
fi 