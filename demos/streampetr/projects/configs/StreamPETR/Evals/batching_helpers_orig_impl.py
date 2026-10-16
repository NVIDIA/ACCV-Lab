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

# Select the model and optimization settings for this comparison variant.
_base_ = ["../_Base/base.py"]

num_gpus = 1

# Override the shared model settings for this variant.
model = {'img_roi_head': {'use_cust_op_heatmap': False}, 'train_cfg': {'pts': {'nan_to_num_on_gpu': False}}}
