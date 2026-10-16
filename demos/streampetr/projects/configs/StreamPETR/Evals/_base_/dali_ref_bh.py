# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Configuration adjusted for DALI reference test
#          (based on stream_petr_r50_flash_704_bs2_seq_24e.py).
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

# Override the reference model with batched heads and assigners. Replace the pts training dictionary to
# remove the reference-only NaN-handling option rather than inheriting it.
_base_ = ["./dali_ref.py"]

model = {
    'img_roi_head': {
        'type': 'FocalHeadBatched',
        'train_cfg': {
            'assigner2d': {
                'type': 'HungarianAssigner2DBatched',
                'cls_cost': {'type': 'FocalLossCostBatched'},
                'reg_cost': {'type': 'BBoxL1CostBatched'},
                'iou_cost': {'type': 'IoUCostBatched'},
            }
        },
    },
    'pts_bbox_head': {'type': 'StreamPETRHeadBatched'},
    'train_cfg': {
        'pts': {
            '_delete_': True,
            'grid_size': [512, 512, 1],
            'voxel_size': [0.2, 0.2, 8],
            'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'out_size_factor': 4,
            'assigner': {
                'type': 'HungarianAssigner3DBatched',
                'cls_cost': {'type': 'FocalLossCostBatched', 'weight': 2.0},
                'reg_cost': {'type': 'BBox3DL1Cost', 'weight': 0.25},
                'iou_cost': {'type': 'IoUCost', 'weight': 0.0},
                'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            },
        }
    },
}
