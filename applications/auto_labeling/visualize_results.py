#!/usr/bin/env python

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
# Offline Visualization Script
# Read detection_results.json and project 3D detection boxes onto original images
# ------------------------------------------------------------------------

import os
import sys
import json
import pickle
import argparse
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from applications.auto_labeling.data_index import (
    convert_egopose_to_matrix_numpy,
    invert_matrix_egopose_numpy,
)

CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (0, 200, 255),
    'construction_vehicle': (0, 165, 255),
    'bus': (255, 255, 0),
    'trailer': (255, 0, 255),
    'barrier': (128, 128, 128),
    'motorcycle': (255, 0, 0),
    'bicycle': (255, 128, 0),
    'pedestrian': (0, 0, 255),
    'traffic_cone': (0, 128, 255),
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize Detection Results')
    parser.add_argument(
        '--json-path',
        type=str,
        default='auto_labeling_results/detection_results.json',
        help='Path to detection results JSON file',
    )
    parser.add_argument(
        '--pkl-path', type=str, required=True, help='Path to data pkl file for lidar2img matrices'
    )
    parser.add_argument('--data-root', default='.', help='Root for relative image filenames')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='auto_labeling_results/visualizations',
        help='Output directory for visualization images',
    )
    parser.add_argument(
        '--num-samples', type=int, default=None, help='Number of samples to visualize (default: all)'
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.3, help='Score threshold for filtering detections'
    )
    return parser.parse_args()


def get_lidar2img_from_info(info: Dict) -> List[np.ndarray]:
    """
    Compute lidar2img transformation matrix from pkl info

    Args:
        info: Info dictionary for a single sample

    Returns:
        List of lidar2img matrices, one for each camera
    """
    lidar2img_list = []
    for cam_type, cam_info in info['cams'].items():
        # Camera to LiDAR transformation
        cam2lidar_r = np.array(cam_info['sensor2lidar_rotation'], dtype=np.float32)
        cam2lidar_t = np.array(cam_info['sensor2lidar_translation'], dtype=np.float32)
        cam2lidar = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
        lidar2cam = invert_matrix_egopose_numpy(cam2lidar)

        # Camera intrinsic
        intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

        # lidar2img = intrinsic @ lidar2cam
        lidar2img = viewpad @ lidar2cam
        lidar2img_list.append(lidar2img)

    return lidar2img_list


def get_3d_box_corners(detection: Dict) -> np.ndarray:
    """
    Get 8 corner coordinates of 3D bbox using nuscenes Box class

    Note: The z coordinate from model output is the bottom center (not geometric center)

    Args:
        detection: Detection result dictionary containing translation, size, rotation_yaw

    Returns:
        corners: shape (3, 8) - 3D coordinates of 8 corners
    """
    x, y, z = detection['translation']
    # Model output: [x_size, y_size, z_size]
    # nuScenes Box expects: [width, length, height] = [y_size, x_size, z_size]
    x_size, y_size, z_size = detection['size']
    yaw = detection['rotation_yaw']

    # Convert yaw to quaternion (rotation around z-axis)
    # nuScenes uses quaternion format [w, x, y, z]
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import Box

    quaternion = Quaternion(axis=[0, 0, 1], angle=yaw)

    # nuScenes Box expects size in [width, length, height] format
    # where width is y-direction, length is x-direction
    wlh = [y_size, x_size, z_size]

    # The model outputs z as bottom center, but nuScenes Box uses geometric center
    # So we need to adjust z to be the geometric center: z_center = z_bottom + h/2
    center = [x, y, z + z_size / 2]

    # Create nuScenes Box and get corners
    box = Box(center=center, size=wlh, orientation=quaternion)
    corners = box.corners()  # Returns (3, 8) array

    return corners


def project_to_image(points_3d: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image

    Returns:
        points_2d: shape (2, N)
        valid_mask: shape (N,) - Valid point mask
    """
    n = points_3d.shape[1]
    pts_homo = np.vstack([points_3d, np.ones((1, n))])
    pts_2d_homo = lidar2img @ pts_homo
    depths = pts_2d_homo[2, :]
    valid = depths > 0.1

    pts_2d = np.zeros((2, n))
    pts_2d[0, valid] = pts_2d_homo[0, valid] / depths[valid]
    pts_2d[1, valid] = pts_2d_homo[1, valid] / depths[valid]

    return pts_2d, valid


def draw_3d_box(
    image: np.ndarray,
    corners_2d: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw 3D bbox on image"""
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]
    h, w = image.shape[:2]

    for i, j in edges:
        if valid[i] and valid[j]:
            pt1 = (int(corners_2d[0, i]), int(corners_2d[1, i]))
            pt2 = (int(corners_2d[0, j]), int(corners_2d[1, j]))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                t = thickness + 1 if (i, j) in front_edges else thickness
                cv2.line(image, pt1, pt2, color, t)
    return image


def draw_label(
    image: np.ndarray,
    corners_2d: np.ndarray,
    valid: np.ndarray,
    label: str,
    score: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw label above 3D bbox"""
    h, w = image.shape[:2]
    valid_pts = corners_2d[:, valid]
    if valid_pts.shape[1] == 0:
        return image

    # Get the topmost point
    min_y_idx = np.argmin(valid_pts[1, :])
    lx = int(np.clip(valid_pts[0, min_y_idx], 0, w - 50))
    ly = int(np.clip(valid_pts[1, min_y_idx] - 10, 15, h - 5))

    text = f"{label}: {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(image, (lx, ly - th - 2), (lx + tw, ly + 2), color, -1)
    cv2.putText(image, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image


def visualize_sample(
    sample_data: Dict,
    lidar2img_list: List[np.ndarray],
    output_path: str,
    score_threshold: float = 0.3,
    data_root: str = '.',
    camera_names=None,
):
    """
    Visualize a single sample

    Args:
        sample_data: Single sample data from JSON
        lidar2img_list: lidar2img matrices for 6 cameras
        output_path: Output image path
        score_threshold: Score threshold
    """
    filenames = sample_data['filename']
    detections = sample_data['detections']
    if camera_names is None or len(filenames) != 6 or len(lidar2img_list) != 6 or len(camera_names) != 6:
        raise ValueError('Visualization requires six camera names, images and calibration matrices')

    vis_images = []
    for cam_idx, img_path in enumerate(filenames):
        # Read image
        if not os.path.isabs(img_path):
            img_path = os.path.join(data_root, img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')

        lidar2img = lidar2img_list[cam_idx]

        # Draw each detection box
        for det in detections:
            if det['score'] < score_threshold:
                continue

            # Get 3D corners
            corners_3d = get_3d_box_corners(det)

            # Project to 2D
            corners_2d, valid = project_to_image(corners_3d, lidar2img)

            if not valid.any():
                continue

            # Get color
            label_name = det['label_name']
            color = CLASS_COLORS.get(label_name, (0, 255, 0))

            # Draw
            img = draw_3d_box(img, corners_2d, valid, color)
            img = draw_label(img, corners_2d, valid, label_name, det['score'], color)

        # Add camera name
        cv2.putText(img, camera_names[cam_idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vis_images.append(img)

    # Concatenate images (2 rows, 3 columns)
    # Reorder: FRONT_LEFT, FRONT, FRONT_RIGHT / BACK_LEFT, BACK, BACK_RIGHT
    reorder = [
        camera_names.index(name)
        for name in (
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
        )
    ]
    vis_images = [vis_images[i] for i in reorder]

    row1 = np.hstack(vis_images[:3])
    row2 = np.hstack(vis_images[3:])
    combined = np.vstack([row1, row2])

    # Scale to save space
    scale = 0.5
    combined = cv2.resize(combined, None, fx=scale, fy=scale)

    if not cv2.imwrite(output_path, combined):
        raise OSError(f'Cannot write visualization: {output_path}')


def main():
    """Main function"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Visualize Detection Results")
    print("=" * 60)
    print(f"JSON path: {args.json_path}")
    print(f"PKL path: {args.pkl_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Score threshold: {args.score_threshold}")
    print("=" * 60)

    # Load detection results
    print("\nLoading detection results...")
    with open(args.json_path, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} samples")

    # Load pkl file to get lidar2img
    print("\nLoading pkl file for lidar2img matrices...")
    with open(args.pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    infos = pkl_data['infos']

    # Build sample_idx to info mapping
    token_to_info = {info['token']: info for info in infos}
    print(f"Loaded {len(infos)} infos from pkl")

    # Visualize
    num_samples = args.num_samples or len(results)
    num_samples = min(num_samples, len(results))

    print(f"\nVisualizing {num_samples} samples...")

    for i, sample in enumerate(results[:num_samples]):
        sample_idx = sample['sample_idx']

        # Get corresponding info
        if sample_idx not in token_to_info:
            raise ValueError(f'Sample {sample_idx} not found in the info pickle')

        info = token_to_info[sample_idx]
        lidar2img_list = get_lidar2img_from_info(info)

        # Visualize
        output_path = os.path.join(args.output_dir, f"sample_{i:04d}.jpg")
        visualize_sample(
            sample, lidar2img_list, output_path, args.score_threshold, args.data_root, list(info['cams'])
        )

        print(f"\r  Processed {i+1}/{num_samples} samples", end='')

    print(f"\n\nDone! Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
