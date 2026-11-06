# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Configuration adjusted for profiling
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

_base_ = ["../../_Base/base.py"]

# Override random augmentations with fixed values for repeatable profiling.
# MMCV replaces pipeline lists as a whole; update every pipeline consumer below.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

collect_keys = [
    'lidar2img',
    'intrinsics',
    'extrinsics',
    'timestamp',
    'img_timestamp',
    'ego_pose',
    'ego_pose_inv',
]

ida_aug_conf = {
    "resize_lim": (0.44, 0.44),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
__global_rot_scale_config = {
    "rot_range": [0.3925, 0.3925],
    "translation_std": [0, 0, 0],
    "scale_ratio_range": [1.05, 1.05],
    "reverse_angle": True,
    "training": True,
}
__dali_affine_trafo_config = {
    "rand_flip": False,
    "resize_lim": (1.0, 1.0),
    "output_hw": (256, 704),
}
__dali_bev_bboxes_trafo_config = {
    "rotation_range": [0.3925, 0.3925],
    "translation_range": [0, 0, 0],
    "scaling_range": [1.05, 1.05],
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', **__global_rot_scale_config),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d',
            'gt_labels_3d',
            'img',
            'gt_bboxes',
            'gt_labels',
            'centers2d',
            'depths',
            'prev_exists',
        ]
        + collect_keys,
        meta_keys=(
            'filename',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'scene_token',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ),
    ),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type='Collect3D',
                keys=['img'] + collect_keys,
                meta_keys=(
                    'filename',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'scene_token',
                ),
            ),
        ],
    ),
]


# Override worker count, sampling and measurement duration for profiling.
num_gpus = 1
num_iters_per_epoch = 50
data = dict(
    workers_per_gpu=16,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    shuffler_sampler=dict(randomize=False),
)
evaluation = dict(interval=num_iters_per_epoch, pipeline=test_pipeline)
checkpoint_config = dict(interval=num_iters_per_epoch)
runner = dict(max_iters=num_iters_per_epoch)
dali_preproc_config = dict(
    affine_trafo_config=__dali_affine_trafo_config,
    bev_bboxes_trafo_config=__dali_bev_bboxes_trafo_config,
    randomize_sample_selection=False,
    max_iterations_input_callable=num_iters_per_epoch,
)
runtime_meas_config = dict(nvtx_range_wrapper=dict(enable=True))
