# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
#  (StreamPETR: https://github.com/exiawsh/StreamPETR)
#
# ------------------------------------------------------------------------
#
# The sample metadata and camera calibration in DataIndex.get_data_info and
# camera_matrices adapt StreamPETR's datasets/nuscenes_dataset.py (95f6470).
# The upstream dataset also credits SECOND.Pytorch for the metadata protocol.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Added standalone pickle indexing by scene, NumPy quaternion
#          conversion, and per-camera video paths and frame IDs.
#
# This file contains contributions made under each of the licenses below;
# both apply to it. Each license governs the contributions made under it.
#
# ........................................................................
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ........................................................................
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
# DataIndex - Data index management for pkl files
# ------------------------------------------------------------------------

import pickle
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from . import add_streampetr_import_path


def camera_matrices(cam_info):
    """Build float32 camera calibration for inference and visualization."""
    add_streampetr_import_path()
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
        convert_egopose_to_matrix_numpy,
        invert_matrix_egopose_numpy,
    )

    cam2lidar = convert_egopose_to_matrix_numpy(
        cam_info['sensor2lidar_rotation'], cam_info['sensor2lidar_translation']
    )
    lidar2cam = invert_matrix_egopose_numpy(cam2lidar)
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad, lidar2cam, viewpad @ lidar2cam


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: Quaternion in [w, x, y, z] format

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion

    n = w * w + x * x + y * y + z * z
    if n == 0:
        return np.eye(3, dtype=np.float32)

    s = 2.0 / n
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    rotation_matrix = np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ],
        dtype=np.float32,
    )

    return rotation_matrix


class DataIndex:
    """
    Manages pkl data index and provides metadata queries.

    Handles:
    - Loading and parsing pkl files
    - Metadata extraction for samples
    - Video index building for multi-video sampling
    - Path resolution (relative to absolute)
    """

    def __init__(self, pkl_path: str, data_root: str):
        """
        Initialize data index.

        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.data_infos = data['infos']
        if not self.data_infos:
            raise ValueError('The dataset contains no samples')
        self.data_root = data_root
        self.video_indices = self._build_video_index()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_infos)

    def _build_video_index(self) -> Dict[str, List[int]]:
        """
        Build index mapping scene_token to list of sample indices.

        Samples within each video are sorted by frame_idx to ensure
        temporal order is preserved.

        Returns:
            Dict mapping scene_token to list of sample indices
        """
        video_index = defaultdict(list)

        for idx, info in enumerate(self.data_infos):
            scene_token = info['scene_token']
            frame_idx = info.get('frame_idx', info['timestamp'])
            video_index[scene_token].append((idx, frame_idx))

        sorted_index = {}
        for scene_token, frames in video_index.items():
            sorted_frames = sorted(frames, key=lambda x: x[1])
            sorted_index[scene_token] = [idx for idx, _ in sorted_frames]

        return sorted_index

    def get_data_info(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for a single sample.

        Parses the pkl info and extracts:
            - Image paths (img_filename)
            - Video paths and frame IDs (if available)
            - Camera intrinsics and extrinsics
            - Ego pose information

        Args:
            index: Data index

        Returns:
            Dictionary containing metadata
        """
        add_streampetr_import_path()
        from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
            convert_egopose_to_matrix_numpy,
            invert_matrix_egopose_numpy,
        )

        info = self.data_infos[index]

        # Parse ego pose
        e2g_rotation = quaternion_to_rotation_matrix(info['ego2global_rotation'])
        e2g_translation = np.array(info['ego2global_translation'], dtype=np.float32)
        l2e_rotation = quaternion_to_rotation_matrix(info['lidar2ego_rotation'])
        l2e_translation = np.array(info['lidar2ego_translation'], dtype=np.float32)

        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose = e2g_matrix @ l2e_matrix
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)

        input_dict = {
            'sample_idx': info['token'],
            'pts_filename': info.get('lidar_path', ''),
            'sweeps': info.get('sweeps', []),
            'ego_pose': ego_pose,
            'ego_pose_inv': ego_pose_inv,
            'prev_idx': info.get('prev', ''),
            'next_idx': info.get('next', ''),
            'scene_token': info['scene_token'],
            'frame_idx': info.get('frame_idx', 0),
            'timestamp': info['timestamp'] / 1e6,
        }

        # Parse camera info
        image_paths = []
        video_paths = []
        video_frame_ids = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []

        for cam_info in info['cams'].values():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])

            if 'video_path' in cam_info:
                video_paths.append(cam_info['video_path'])
                video_frame_ids.append(cam_info['video_frame_id'])

            viewpad, lidar2cam_rt, lidar2img_rt = camera_matrices(cam_info)

            intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt)
            lidar2img_rts.append(lidar2img_rt)

        input_dict.update(
            {
                'img_timestamp': img_timestamp,
                'img_filename': image_paths,
                'lidar2img': lidar2img_rts,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'video_paths': video_paths if len(video_paths) > 0 else None,
                'video_frame_ids': video_frame_ids if len(video_frame_ids) > 0 else None,
            }
        )

        return input_dict
