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

res_dir=eval_res
mkdir -p "$res_dir"

echo Profiling batched ...
nsys profile  -s cpu  --trace-fork-before-exec true -t cuda,nvtx,osrt,cudnn,cublas  -f true -x true -o streampetr_batched python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Profiling/stream_petr_r50_flash_704_bs8_seq_24e_batched_profile.py
echo Profiling original ...
nsys profile  -s cpu  --trace-fork-before-exec true -t cuda,nvtx,osrt,cudnn,cublas  -f true -x true -o streampetr_orig python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Profiling/stream_petr_r50_flash_704_bs8_seq_24e_original_profile.py
echo Profiling original with nan_to_num on the GPU...
nsys profile  -s cpu  --trace-fork-before-exec true -t cuda,nvtx,osrt,cudnn,cublas  -f true -x true -o streampetr_orig_switched python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Profiling/stream_petr_r50_flash_704_bs8_seq_24e_original_nan2numgpu_profile.py
