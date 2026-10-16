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

res_dir=eval_res_dali_load_only
mkdir -p "$res_dir"

echo Evaluating data loading only for DALI ...
python /streampetr/tools/train.py --no-validate \
    /streampetr/projects/configs/StreamPETR/DALILoadOnly/dali_load_only.py \
    > "$res_dir/dali_load_only_res.txt" \
    2> "$res_dir/dali_load_only_err.txt"

echo Evaluating data loading only for the reference data loader ...
python /streampetr/tools/train.py --no-validate \
    /streampetr/projects/configs/StreamPETR/DALILoadOnly/ref_load_only.py \
    > "$res_dir/original_load_only_res.txt" \
    2> "$res_dir/original_load_only_err.txt"

echo Evaluation finished
