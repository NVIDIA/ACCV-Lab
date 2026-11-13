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
# DataIndex - Data index management for pkl files
# ------------------------------------------------------------------------

import pickle
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 egopose matrix.

    Args:
        egopose: 4x4 transformation matrix

    Returns:
        Inverse of the transformation matrix
    """
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Convert rotation and translation to a 4x4 transformation matrix.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3-element translation vector

    Returns:
        4x4 transformation matrix
    """
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


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

        for cam_type, cam_info in info['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])

            if 'video_path' in cam_info:
                video_paths.append(cam_info['video_path'])
                video_frame_ids.append(cam_info['video_frame_id'])

            cam2lidar_r = np.array(cam_info['sensor2lidar_rotation'], dtype=np.float32)
            cam2lidar_t = np.array(cam_info['sensor2lidar_translation'], dtype=np.float32)
            cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
            lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

            intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
            viewpad = np.eye(4, dtype=np.float32)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt

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
