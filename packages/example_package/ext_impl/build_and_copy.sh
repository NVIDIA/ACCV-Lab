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

set -euo pipefail

echo "Building external CUDA implementation for example_package..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set up build directory
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Get PyTorch cmake config path
TORCH_CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

echo "Computing CMake arguments from environment..."
# Ensure helper is available; instruct user if not installed
if ! python - <<'PY'
from accvlab_build_config.helpers.cmake_args import build_cmake_args  # noqa: F401
PY
then
    echo "Error: Python package 'accvlab_build_config' not found." >&2
    echo "Please install it first, e.g.:" >&2
    echo "  pip install -e build_config    # from repo root" >&2
    exit 1
fi
# Read helper-produced cmake -D args into array
readarray -t CMAKE_ARGS < <(python -c "from accvlab_build_config.helpers.cmake_args import build_cmake_args; print('\n'.join(build_cmake_args()))")

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "${CMAKE_ARGS[@]}"

# Build the library
echo "Building library..."
make -j$(nproc)

# Find the built library (Python module) - pybind11 creates files with platform-specific names
LIB_FILE=$(find . -name "accvlab_example_package_ext*.so" | head -1)
if [ ! -f "$LIB_FILE" ]; then
    echo "Error: Could not find built library accvlab_example_package_ext*.so"
    echo "Available files:"
    ls -la
    exit 1
fi

# Create destination directory
DEST_DIR="../../accvlab/example_package"
mkdir -p "$DEST_DIR"

# Copy the library to the examples package with the correct Python module name
echo "Copying library to example_package package..."
cp "$LIB_FILE" "$DEST_DIR/"

echo "External build completed successfully!"
echo "Library copied to: $DEST_DIR/$(basename "$LIB_FILE")" 