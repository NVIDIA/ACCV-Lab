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

# Keep the reference profiling workload and switch to the DALI Pipeline Framework loader.
_base_ = ["./stream_petr_r50_flash_1024_bs8_seq_24e_original_data_loader_8_gpu.py"]

# Override the reference loader with the DALI Pipeline Framework loader.
data_loading_config = {'use_dali_pipeline': True, 'nuscenes_version': 'v1.0-mini'}
