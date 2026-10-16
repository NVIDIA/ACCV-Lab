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

# Override the larger-image workload with 100 profiling iterations and NVTX ranges enabled.
_base_ = ["../_Base/dali_larger_images.py"]

num_gpus = 8

num_iters_per_epoch = 100

evaluation = {'interval': 100}

checkpoint_config = {'interval': 100}

runner = {'max_iters': 100}

dali_preproc_config = {'max_iterations_input_callable': 100}

runtime_meas_config = {'nvtx_range_wrapper': {'enable': True}}

# Override shared settings for the replicated mini-dataset workload.
data_loading_config = dict(replicate_dataset_per_gpu=True)
model = dict(train_cfg=dict(pts=dict(nan_to_num_on_gpu=False)))
