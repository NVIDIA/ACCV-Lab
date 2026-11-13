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

# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# Batch Utils - Utilities for batch assembly and metadata handling
# ------------------------------------------------------------------------

import numpy as np
import torch
from typing import List, Dict, Any, Tuple


def update_metadata_for_transform(
    metadata: Dict[str, Any], resize: float, crop_h: int, crop_w: int, final_dim: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Update intrinsics and lidar2img matrices after image transformation.

    Args:
        metadata: Original metadata dictionary
        resize: Resize scale factor
        crop_h: Crop offset in height (top)
        crop_w: Crop offset in width (left)
        final_dim: Final image dimensions (H, W)

    Returns:
        Updated metadata with modified intrinsics and lidar2img
    """
    fH, fW = final_dim
    intrinsics = metadata.get('intrinsics', [])
    new_intrinsics = []

    for i in range(len(intrinsics)):
        intr = intrinsics[i].copy()
        # Build IDA transformation matrix
        ida_mat = np.eye(3, dtype=np.float32)
        ida_mat[0, 0] = resize  # scale x
        ida_mat[1, 1] = resize  # scale y
        ida_mat[0, 2] = -crop_w  # translate x
        ida_mat[1, 2] = -crop_h  # translate y
        intr[:3, :3] = ida_mat @ intr[:3, :3]
        new_intrinsics.append(intr)

    if new_intrinsics:
        metadata['intrinsics'] = new_intrinsics
        extrinsics = metadata.get('extrinsics', [])
        if extrinsics:
            metadata['lidar2img'] = [new_intrinsics[i] @ extrinsics[i] for i in range(len(extrinsics))]

    metadata['img_shape'] = [(fH, fW, 3) for _ in range(len(intrinsics))]

    return metadata


def build_img_metas(
    metadata: Dict[str, Any],
    pad_shape: Tuple[int, int],
    img_norm_cfg: Dict[str, Any],
    ida_aug_conf: Dict[str, Any],
    num_views: int,
) -> Dict[str, Any]:
    """
    Build img_metas dictionary for a single sample.

    Args:
        metadata: Sample metadata
        pad_shape: Padded image dimensions (H, W)
        img_norm_cfg: Image normalization config
        ida_aug_conf: Image augmentation config
        num_views: Number of camera views

    Returns:
        img_metas dictionary
    """
    from mmdet3d.core.bbox import get_box_type

    box_type_3d, box_mode_3d = get_box_type('LiDAR')

    return {
        'filename': metadata['img_filename'],
        'ori_shape': (ida_aug_conf['H'], ida_aug_conf['W'], 3),
        'img_shape': metadata.get('img_shape', [(pad_shape[0], pad_shape[1], 3)] * num_views),
        'pad_shape': [(pad_shape[0], pad_shape[1], 3)] * num_views,
        'scale_factor': 1.0,
        'flip': False,
        'box_mode_3d': box_mode_3d,
        'box_type_3d': box_type_3d,
        'img_norm_cfg': img_norm_cfg,
        'scene_token': metadata['scene_token'],
        'sample_idx': metadata['sample_idx'],
    }


def build_sample(
    img_tensor: torch.Tensor, metadata: Dict[str, Any], img_metas: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a single sample dictionary.

    Args:
        img_tensor: Image tensor (N, C, H, W)
        metadata: Sample metadata
        img_metas: Image metadata dictionary

    Returns:
        Sample dictionary
    """
    return {
        'img': img_tensor,
        'img_metas': img_metas,
        'lidar2img': np.stack(metadata['lidar2img'], axis=0),
        'intrinsics': np.stack(metadata['intrinsics'], axis=0),
        'extrinsics': np.stack(metadata['extrinsics'], axis=0),
        'timestamp': np.array(metadata['timestamp'], dtype=np.float64),
        'img_timestamp': np.array(metadata['img_timestamp'], dtype=np.float64),
        'ego_pose': metadata['ego_pose'],
        'ego_pose_inv': metadata['ego_pose_inv'],
    }


def collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate multiple samples into a batch.

    Args:
        samples: List of sample dictionaries

    Returns:
        Batch dictionary with stacked tensors
    """
    # Stack image tensors
    img = torch.stack([s['img'] for s in samples], dim=0)  # (B, N, C, H, W)

    # Stack numpy arrays and convert to tensor
    lidar2img = torch.from_numpy(np.stack([s['lidar2img'] for s in samples], axis=0)).float()  # (B, N, 4, 4)

    intrinsics = torch.from_numpy(
        np.stack([s['intrinsics'] for s in samples], axis=0)
    ).float()  # (B, N, 4, 4)

    extrinsics = torch.from_numpy(
        np.stack([s['extrinsics'] for s in samples], axis=0)
    ).float()  # (B, N, 4, 4)

    timestamp = torch.from_numpy(np.stack([s['timestamp'] for s in samples], axis=0)).double()  # (B,)

    img_timestamp = torch.from_numpy(
        np.stack([s['img_timestamp'] for s in samples], axis=0)
    ).double()  # (B, N)

    ego_pose = torch.from_numpy(np.stack([s['ego_pose'] for s in samples], axis=0)).float()  # (B, 4, 4)

    ego_pose_inv = torch.from_numpy(
        np.stack([s['ego_pose_inv'] for s in samples], axis=0)
    ).float()  # (B, 4, 4)

    # Collect img_metas as list
    img_metas = [s['img_metas'] for s in samples]

    return {
        'img': img,
        'img_metas': img_metas,
        'lidar2img': lidar2img,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'timestamp': timestamp,
        'img_timestamp': img_timestamp,
        'ego_pose': ego_pose,
        'ego_pose_inv': ego_pose_inv,
    }
