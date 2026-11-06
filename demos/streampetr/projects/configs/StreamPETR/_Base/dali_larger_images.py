# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Shared larger-image configuration for the ACCV-Lab demo
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

# Override image geometry with the 1024 x 372 workload shared by both loaders.
# Loader selection and dataset replication are configured by the individual variants.
_base_ = ["./base.py"]

# MMCV replaces pipeline lists as a whole. Rebuild them with the larger image
# size, then update the training, validation and evaluation pipeline consumers.
scale = 1024 / 1600

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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

# `resize_lim` below combines the resizing relative to the original viewport as well as the image
# resolution change. The 1024 / 1600 factor converts the original viewport width
# to the output width; the other factors specify the random resizing.
ida_aug_conf = {
    "resize_lim": (0.86 * (1024 / 1600), 1.25 * (1024 / 1600)),
    "final_dim": (372, 1024),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
__global_rot_scale_config = {
    "rot_range": [-0.3925, 0.3925],
    "translation_std": [0, 0, 0],
    "scale_ratio_range": [0.95, 1.05],
    "reverse_angle": True,
    "training": True,
}
# Note that for DALI, the `resize_lim` is relative to the original viewport, and the image resolution
# change is not considered (the corresponding factor is applied internally based on `output_hw` and
# the original image size).
__dali_affine_trafo_config = {
    "rand_flip": True,
    "resize_lim": (0.86, 1.25),
    "output_hw": (372, 1024),
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

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

evaluation = dict(pipeline=test_pipeline)

dali_preproc_config = dict(affine_trafo_config=__dali_affine_trafo_config)
