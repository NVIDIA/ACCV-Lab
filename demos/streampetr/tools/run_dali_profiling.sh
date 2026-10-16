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

res_dir=profiling_res
mkdir -p "$res_dir"

echo Profiling reference ...
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Profiling/stream_petr_r50_flash_1024_bs8_seq_24e_original_data_loader_8_gpu.py 8 --profile --profile-output original_profile > "$res_dir/original_res.txt" 2> "$res_dir/original_err.txt"

echo Profiling DALI ...
bash dist_train.sh /streampetr/projects/configs/StreamPETR/Profiling/stream_petr_r50_flash_1024_bs8_seq_24e_original_dali_8_gpu.py 8 --profile --profile-output dali_profile > "$res_dir/dali_res.txt" 2> "$res_dir/dali_err.txt"

echo Profiling finished
