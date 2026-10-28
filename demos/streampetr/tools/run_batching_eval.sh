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

res_dir=res_batching_eval
mkdir -p "$res_dir"

echo Running batched implementation ...
python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_test.py > "$res_dir/batching_helpers_test.txt" 2>&1
echo Running original implementation ...
python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_orig_impl.py > "$res_dir/batching_helpers_orig_impl.txt" 2>&1
echo Running original implementation with nan_to_num on the GPU...
python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_ref.py > "$res_dir/batching_helpers_ref.txt" 2>&1
