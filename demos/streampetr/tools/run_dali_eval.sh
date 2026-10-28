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

res_dir=res_dali_eval
mkdir -p "$res_dir"

echo Evaluating DALI ...
echo ... for 2 GPUs ...
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_2_gpu.py 2 > "$res_dir/dali_test_2_gpu.txt" 2>&1

echo ... for 8 GPUs
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_8_gpu.py 8 > "$res_dir/dali_test_8_gpu.txt" 2>&1

echo Evaluating reference ...
echo ... for 2 GPUs ...
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_2_gpu.py 2 > "$res_dir/dali_ref_2_gpu.txt" 2>&1

echo ... for 8 GPUs
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_8_gpu.py 8 > "$res_dir/dali_ref_8_gpu.txt" 2>&1

echo Evaluating finished
