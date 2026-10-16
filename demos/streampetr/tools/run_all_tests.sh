#!/usr/bin/env bash

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

set -euo pipefail

# Number of configs to skip (default: 0)
skip_count=${1:-0}

res_dir=eval_res_all_tests
configs_dir="/streampetr/projects/configs/StreamPETR/Tests"

mkdir -p $res_dir

echo "Running tests for all configs in $configs_dir"
echo "Skipping first $skip_count configs"
echo "Results will be saved in $res_dir"
echo "----------------------------------------"

# Get top-level .py config files only (non-recursive)
mapfile -t all_configs < <(find "$configs_dir" -maxdepth 1 -type f -name "*.py" | sort)
total_configs=${#all_configs[@]}

echo "Found $total_configs configs total"

# If nothing remains after filtering, exit gracefully
if [ $total_configs -eq 0 ]; then
    echo "No configurations found. Exiting."
    exit 0
fi

# Check if skip count is valid
if [ $skip_count -ge $total_configs ]; then
    echo "Error: Skip count ($skip_count) is greater than or equal to total configs ($total_configs)"
    exit 1
fi

# Loop through configs starting from the skip_count
for ((i=skip_count; i<total_configs; i++)); do
    config_file="${all_configs[$i]}"
    # Extract the base filename without extension
    base_name=$(basename "$config_file" .py)
    
    echo "[$((i+1-skip_count))/$((total_configs-skip_count))] Running test for $base_name..."
    
    # Create output files based on config name
    output_file="$res_dir/${base_name}_res.txt"
    error_file="$res_dir/${base_name}_err.txt"
    
    # Check if the filename contains the pattern "_[n]_gpu"
    if [[ $base_name =~ _([0-9]+)_gpu ]]; then
        # Extract the number of GPUs from the regex match
        num_gpus="${BASH_REMATCH[1]}"
        echo "Detected multi-GPU config with $num_gpus GPUs, using dist_train.sh"
        
        # Call dist_train.sh with the appropriate arguments
        bash $(dirname "$0")/dist_train.sh "$config_file" "$num_gpus" > "$output_file" 2> "$error_file"
    else
        # For regular configs, use the original command
        python /streampetr/tools/train.py --no-validate "$config_file" > "$output_file" 2> "$error_file"
    fi
    
    echo "Test for $base_name completed"
    echo "----------------------------------------"
done

echo "All tests completed!"
