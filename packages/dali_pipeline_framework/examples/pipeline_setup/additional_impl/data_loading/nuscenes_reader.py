# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import os
from typing import List, Sequence, Union, Optional, Tuple

import nuscenes.utils.data_classes

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

import numpy as np

from pyquaternion import Quaternion

from .nuscenes_data import *

from .geometry_helpers import (
    get_ego_to_world_transform,
    get_sensor_to_ego_transformation,
    get_cam_intrinsics,
)

from .bbox_projector import BboxProjector


class NuScenesReader:
    '''Reader for the Nuscenes dataset.

    The reader reads the data and converts it into an internal format, which facilitates easy and efficient
    access to the data by the individual converters.
    See documentation of :class:`NuScenesDataSample` and :class:`NuScenesData` for details regarding the
    internal format.
    '''

    # _cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    _cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    _lidar = 'LIDAR_TOP'

    def __init__(
        self,
        base_dir: str,
        nuscenes_version: str,
        can_bus_root_path: Optional[str] = None,
        can_messages_to_add: Optional[Union[str, Sequence[str]]] = None,
        add_image_annotations: bool = True,
        add_projected_bboxes_annotations: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        '''

        Args:
            base_dir: Base directory of the NuScenes dataset
            nuscenes_version: NuScenes version to load
            can_bus_root_path: Root path of the CAN bus data. Needs to be set if any CAN bus messages should
                be added
            can_messages_to_add: Name or names of CAN bus messages to include in the data
            add_image_annotations: Whether to add image annotations (as generated using the
                ``export_2d_annotations_as_json`` script from the NuScenes DevKit). Note that they can
                be generated using the `prepare_dataset.py` convenience script in the DALI pipeline framework
                examples. Set to ``False`` if exported image annotations are not needed;
                this does not disable generation of projected bounding boxes.
            add_projected_bboxes_annotations: Whether to add projected bounding boxes annotations. The
                projection is done using the ``BboxProjector`` defined in ``bbox_projector.py``. (used in the
                StreamPETR example)
            image_size: Size of the images. It is needed if ``add_projected_bboxes_annotations`` is set to
            ``True``. Otherwise, it is not needed (and ignored).
        '''
        if isinstance(can_messages_to_add, str):
            can_messages_to_add = [can_messages_to_add]
        self._base_dir = base_dir
        self._data = self._get_data(
            base_dir,
            nuscenes_version,
            can_bus_root_path,
            can_messages_to_add,
            add_image_annotations,
            add_projected_bboxes_annotations,
            image_size,
        )

    @staticmethod
    def load_data_if_available_else_create_and_store(
        base_dir: str,
        filename_pkl: str,
        nuscenes_version: str,
        can_bus_root_path: Optional[str] = None,
        can_messages_to_add: Optional[Union[str, Sequence[str]]] = None,
        add_image_annotations: bool = True,
        add_projected_bboxes_annotations: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> NuScenesData:
        '''Load data in the internal format if it is available, otherwise load from original format and store in internal format for future use.

        Args:
            base_dir: Base directory of the NuScenes dataset
            filename_pkl: Filename of the file containing the dataset in the internal file. Will be loaded if available. If not available
                it will be created after loading the dataset in the original format & converting to internal format.
            nuscenes_version: NuScenes version to load
            can_bus_root_path: Root path of the CAN bus data. Needs to be set is any CAN bus messages should be added
            can_messages_to_add: Name or names of CAN bus messages to include in the data
            add_image_annotations: Whether to add exported image annotations from
                ``<base_dir>/<nuscenes_version>/image_annotations.json``.
            add_projected_bboxes_annotations: Whether to add projected bounding boxes annotations
            image_size: Size of the images. It is needed if ``add_projected_bboxes_annotations`` is set to
                ``True``. Otherwise, it is not needed (and ignored).
        Returns:
            The loaded dataset in the internal format

        '''

        path_file_pkl = os.path.join(base_dir, filename_pkl)
        try:
            res = NuScenesData.load_pickled(path_file_pkl)
        except:
            # If we need to add image annotations, ensure that they have been generated beforehand.
            # The annotations are expected to be created using the `prepare_dataset.py` script in the
            # DALI pipeline framework examples, which calls the NuScenes DevKit script
            # `export_2d_annotations_as_json` and stores the result as `image_annotations.json`
            # in the version subdirectory of the NuScenes dataset root.
            if add_image_annotations:
                annotations_path = os.path.join(base_dir, nuscenes_version, "image_annotations.json")
                if not os.path.exists(annotations_path):
                    raise RuntimeError(
                        "2D image annotations not found.\n"
                        f"Expected file: '{annotations_path}'.\n"
                        "Please prepare the NuScenes 2D image annotations first by running the "
                        "`prepare_dataset.py` script in the DALI pipeline framework examples, "
                        "then re-run this pipeline."
                    )

            reader = NuScenesReader(
                base_dir,
                nuscenes_version,
                can_bus_root_path,
                can_messages_to_add,
                add_image_annotations,
                add_projected_bboxes_annotations,
                image_size,
            )
            res = reader.data
            res.store_pickled(path_file_pkl)
        return res

    @property
    def data(self) -> NuScenesData:
        '''Get the loaded data'''
        return self._data

    @property
    def cams(self) -> List[str]:
        '''Get a list of available cams.

        Note that the order of the cameras returned here corresponds to the order in which the cameras are stored in the data.

        '''
        return self._cams

    @property
    def base_dir(self) -> str:
        '''Get the NuScenes base directory'''
        return self._base_dir

    def _get_data(
        self,
        base_dir: str,
        nuscenes_version: str,
        can_bus_root_path: Optional[str],
        can_messages_to_add: Optional[Sequence[str]],
        add_image_annotations: bool,
        add_projected_bboxes_annotations: bool,
        image_size: Optional[Tuple[int, int]],
    ) -> NuScenesData:
        '''Convert the original dataset to the internal format and get the data

        Args:
            base_dir: Base directory of the NuScenes dataset
            nuscenes_version: NuScenes version to load
            can_bus_root_path: Root path of the CAN bus data. Needs to be set is any CAN bus messages should be added
            can_messages_to_add: Names of CAN bus messages to include in the data
            add_image_annotations: Whether to add image annotations
            add_projected_bboxes_annotations: Whether to add projected bounding boxes annotations
            image_size: Size of the images. It is needed if ``add_projected_bboxes_annotations`` is set to
                ``True``. Otherwise, it is not needed (and ignored).
        Returns:
            The read data
        '''
        # Get the dataset in the original format
        data_in = self._get_dataset(base_dir, nuscenes_version)
        # Get the individual driving sequences
        sequences_of_samples_in = self._get_sequences_of_samples(data_in)

        # Exported image annotations are optional. Projected bounding boxes are
        # generated separately below and do not require this annotation table.
        image_annotations_mapped = (
            self._build_image_annotations_map(data_in) if add_image_annotations else None
        )

        # If needed, read the CAN bus
        can_bus_in = NuScenesCanBus(can_bus_root_path) if can_bus_root_path is not None else None

        num_sequences = len(sequences_of_samples_in)
        sequences = [None] * num_sequences
        for i, samples in enumerate(sequences_of_samples_in):
            if i % 100 == 0:
                if i == 0:
                    print("NuScenesReader: Processing scenes...")
                else:
                    print(f"  Processed scenes: {i}")

            # Get the data for the sequence (i.e. "scene" as defined in NuScenes). See the implementation of the
            # individual methods for details on how the data is obtained.

            name = data_in.get('scene', sequences_of_samples_in[i][0]['scene_token'])['name']

            cam_sample_data = [self._get_sample_data_cams(data_in, s) for s in samples]
            camera_calibration_data = self._get_calibrated_sensors(
                data_in, cam_sample_data, is_single_sensor=False
            )
            ego_poses_at_cam_timestamps = self._get_ego_poses(
                data_in, cam_sample_data, is_single_sensor=False
            )
            annotations = self._get_annotations(data_in, samples)
            annotation_vels = self._get_annotation_velocities(data_in, annotations)
            visibilities = self._get_visibilities(data_in, annotations)
            if add_image_annotations:
                image_annotations = self._get_image_data_annotations(
                    image_annotations_mapped, cam_sample_data
                )
            else:
                image_annotations = None

            lidar_sample_data = [self._get_sample_data_lidar(data_in, s) for s in samples]
            objects_in_lidar_coords = [data_in.get_sample_data(s['data']['LIDAR_TOP'])[1] for s in samples]

            lidar_calibration_data = self._get_calibrated_sensors(
                data_in, lidar_sample_data, is_single_sensor=True
            )
            ego_poses_at_lidar_timestamp = self._get_ego_poses(
                data_in, lidar_sample_data, is_single_sensor=True
            )

            # Update the velocities from world to lidar coordinates
            self._set_velocities_for_boxes_in_place(
                objects_in_lidar_coords, annotation_vels, ego_poses_at_lidar_timestamp, lidar_calibration_data
            )

            # Get CAN messages if needed
            if can_messages_to_add is not None:
                assert (
                    can_bus_in is not None
                ), "`can_messages_to_add` set, but no `can_bus_root_path` is defined"
                can_messages = self._get_can_messages_for_samples(
                    data_in, can_bus_in, samples, can_messages_to_add
                )
            else:
                can_messages = None

            if add_projected_bboxes_annotations:
                projected_bboxes_annotations = self._generate_projected_bboxes_annotations(
                    lidar_calibration_data,
                    ego_poses_at_lidar_timestamp,
                    camera_calibration_data,
                    ego_poses_at_cam_timestamps,
                    objects_in_lidar_coords,
                    annotations,
                    image_size,
                )
            else:
                projected_bboxes_annotations = None

            # Store the loaded data in an object representing the sequence
            sequence_data = NuScenesDataSequence(
                name,
                samples,
                cam_sample_data,
                camera_calibration_data,
                ego_poses_at_cam_timestamps,
                annotations,
                objects_in_lidar_coords,
                annotation_vels,
                visibilities,
                image_annotations,
                projected_bboxes_annotations,
                lidar_calibration_data,
                ego_poses_at_lidar_timestamp,
                can_messages,
            )
            sequences[i] = sequence_data

        # Store all sequences (as well as info on the used cameras and the base directory of the NuScenes dataset) in the internal format & return
        data = NuScenesData(sequences, self._cams, self._base_dir)
        return data

    def _get_dataset(self, base_dir: str, version: str) -> NuScenes:
        '''Get the original dataset in the format used in the NuScenes DevKit.

        Args:
            base_dir: Base directory of the NuScenes dataset
            version: NuScenes version to load

        Returns:
            The original dataset in the format used in the NuScenes DevKit
        '''
        data_in = NuScenes(version=version, dataroot=base_dir, verbose=True)
        return data_in

    def _get_sequences_of_samples(self, data_in: NuScenes) -> List[List[dict]]:
        '''Get the sequences of samples in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit

        Returns:
            The sequences of samples in the original dataset
        '''
        res = [self._get_samples_for_scene(data_in, sc) for sc in data_in.scene]
        return res

    def _get_samples_for_scene(self, data_in: NuScenes, scene: dict) -> List[dict]:
        '''Get the samples for a given scene in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            scene: The scene to get the samples for

        Returns:
            The samples for the given scene
        '''
        res = []
        next_sample_token = scene['first_sample_token']
        num_samples = scene['nbr_samples']
        for _ in range(num_samples):
            curr_sample = data_in.get('sample', next_sample_token)
            res.append(curr_sample)
            next_sample_token = curr_sample['next']
        return res

    def _get_sample_data_cams(self, data_in: NuScenes, sample: dict) -> List[List[dict]]:
        '''Get the sample data for the cameras in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample: The sample to get the sample data for

        Returns:
            The sample data for the cameras in the original dataset
        '''
        res = [self._get_sample_data(data_in, sample, sn) for sn in self._cams]
        return res

    def _get_sample_data_lidar(self, data_in: NuScenes, sample: dict) -> dict:
        '''Get the sample data for the lidar in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample: The sample to get the sample data for

        Returns:
            The sample data for the lidar in the original dataset
        '''
        res = self._get_sample_data(data_in, sample, self._lidar)
        return res

    def _set_velocities_for_boxes_in_place(
        self,
        objects_in_lidar_coords_in_out: List[List[nuscenes.utils.data_classes.Box]],
        annotation_vels: List[List[dict]],
        ego_poses_at_lidar_timestamps: List[List[dict]],
        lidar_calibration_data: List[List[dict]],
    ):
        '''Set the velocities for the boxes in the original dataset.

        Note that the first lidar (i.e. the one with index 0) is the one that is used to get the velocities.
        In the original dataset, there is only one lidar sensor.

        Args:
            objects_in_lidar_coords_in_out: The objects in the original dataset.
                The inner list iterates over the individual objects.
            annotation_vels: The annotation velocities in the original dataset.
                The inner list iterates over the individual objects.
            ego_poses_at_lidar_timestamps: The ego poses at the lidar timestamps in the original dataset.
                The inner list contains the ego poses for the different lidar sensors.
            lidar_calibration_data: The lidar calibration data in the original dataset.
                The inner list contains the lidar calibration data for the different lidar sensors.
        '''
        for objects, velocities, ego_pose, lidar_calib in zip(
            objects_in_lidar_coords_in_out,
            annotation_vels,
            ego_poses_at_lidar_timestamps,
            lidar_calibration_data,
        ):
            ego_to_world_rot = Quaternion(ego_pose["rotation"]).rotation_matrix
            lidar_to_ego_rot = Quaternion(lidar_calib["rotation"]).rotation_matrix

            world_to_ego_rot = np.transpose(ego_to_world_rot)
            ego_to_lidar_rot = np.transpose(lidar_to_ego_rot)

            world_to_lidar_rot = ego_to_lidar_rot @ world_to_ego_rot

            for obj, vel in zip(objects, velocities):
                vel_in_lidar = world_to_lidar_rot @ vel
                obj.velocity = vel_in_lidar

    def _get_sample_data(self, data_in: NuScenes, sample: dict, sensor_name: str) -> dict:
        '''Get the sample data for a given sensor in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample: The sample to get the sample data for
            sensor_name: The name of the sensor to get the sample data for

        Returns:
            The sample data for the given sensor in the original dataset
        '''
        sample_data = data_in.get('sample_data', sample['data'][sensor_name])
        return sample_data

    def _get_calibrated_sensors(
        self,
        data_in: NuScenes,
        sample_datas: Union[List[List[dict]], List[dict]],
        is_single_sensor: bool = False,
    ) -> Union[List[List[dict]], List[dict]]:
        '''Get the calibrated sensors for a given sample data in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample_datas: The sample data to get the calibrated sensors for.
                If ``is_single_sensor`` is ``True``, ``sample_datas`` is a list of sample data.
                If ``is_single_sensor`` is ``False``, the outer list iterates over the samples and the inner list
                over the individual sensors.
            is_single_sensor: Whether the sample data is a single sensor or a list of sensors.

        Returns:
            The calibrated sensors for the given sample data in the original dataset.
            If ``is_single_sensor`` is ``True``, returns a list of calibrated sensors.
                The outer list iterates over the samples.
            If ``is_single_sensor`` is ``False``, returns a list of lists of calibrated sensors.
                The outer list iterates over the samples and the inner list over the individual sensors.
            Each entry corresponds to the calibrated sensor as returned by ``NuScenes.get('calibrated_sensor', ...)``.
        '''
        if is_single_sensor:
            res = [self._get_calibrated_sensor(data_in, sensor) for sensor in sample_datas]
        else:
            res = [
                [self._get_calibrated_sensor(data_in, sensor) for sensor in sensors]
                for sensors in sample_datas
            ]
        return res

    def _get_calibrated_sensor(self, data_in: NuScenes, sample_data: dict) -> dict:
        '''Get the calibrated sensor for a given sample data in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample_data: The sample data to get the calibrated sensor for

        Returns:
            The calibrated sensor for the given sample data in the original dataset
        '''
        token = sample_data['calibrated_sensor_token']
        calibrated_sensor = data_in.get('calibrated_sensor', token)
        return calibrated_sensor

    def _get_ego_poses(
        self,
        data_in: NuScenes,
        sample_datas: Union[List[List[dict]], List[dict]],
        is_single_sensor: bool = False,
    ) -> Union[List[List[dict]], List[dict]]:
        '''Get the ego poses for a given sample data in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample_datas: The sample data to get the ego poses for.
                If `is_single_sensor` is `True`, `sample_datas` is a list of sample data.
                If `is_single_sensor` is `False`, the outer list iterates over the samples and the inner list
                over the individual sensors.
            is_single_sensor: Whether the sample data is a single sensor or a list of sensors.

        Returns:
            The ego poses for the given sample data in the original dataset.
            If `is_single_sensor` is `True`, returns a list of ego poses.
                The outer list iterates over the samples.
            If `is_single_sensor` is `False`, returns a list of lists of ego poses.
                The outer list iterates over the samples and the inner list over the individual sensors.
            Each entry corresponds to the ego pose as returned by `NuScenes.get('ego_pose', ...)`.
        '''
        if is_single_sensor:
            res = [data_in.get('ego_pose', s['ego_pose_token']) for s in sample_datas]
        else:
            res = [
                [data_in.get('ego_pose', s['ego_pose_token']) for s in sensors] for sensors in sample_datas
            ]
        return res

    def _get_annotation_velocities(
        self, data_in: NuScenes, annotations: List[List[dict]]
    ) -> List[List[dict]]:
        '''Get the annotation velocities for a given annotations in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            annotations: The annotations to get the annotation velocities for.
                The outer list iterates over the samples and the inner list over the
                annotation entries corresponding to the individual objects.

        Returns:
            The annotation velocities for the given annotations in the original dataset.
            Each entry corresponds to the velocity as returned by `NuScenes.box_velocity()`.
        '''
        res = [None] * len(annotations)
        for i in range(len(annotations)):
            res[i] = [data_in.box_velocity(ann['token']) for ann in annotations[i]]
        return res

    def _get_annotations(self, data_in: NuScenes, samples: List[dict]) -> List[List[dict]]:
        '''Get the annotations for given samples in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            samples: The samples to get the annotations for.

        Returns:
            The annotations for the given samples in the original dataset.
            Each entry corresponds to the annotation as returned by `NuScenes.get()`.
        '''
        res = [self._get_annotations_for_sample(data_in, s) for s in samples]
        return res

    def _get_annotations_for_sample(self, data_in: NuScenes, sample: dict) -> List[dict]:
        '''Get the annotations for a given sample in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            sample: The sample to get the annotations for

        Returns:
            The annotations for the given sample in the original dataset.
            Each entry corresponds to the annotation as returned by `NuScenes.get()`.
        '''
        annotation_list = sample['anns']
        annotations = [data_in.get('sample_annotation', ann) for ann in annotation_list]
        return annotations

    def _get_visibilities(self, data_in: NuScenes, annotations: List[List[dict]]) -> List[List[dict]]:
        '''Get the visibilities for a given annotations in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            annotations: The annotations to get the visibilities for.
                The outer list iterates over the samples and the inner list over the
                annotation entries corresponding to the individual objects.

        Returns:
            The visibilities for the given annotations in the original dataset.
            Each entry corresponds to the visibility as returned by
            `NuScenes.get('visibility', ...)`.
        '''
        res = [
            [data_in.get('visibility', ann['visibility_token']) for ann in sample_anns]
            for sample_anns in annotations
        ]
        return res

    def _get_can_messages_for_samples(
        self, data_in: NuScenes, can_bus_in: NuScenesCanBus, samples: List[dict], message_names: List[str]
    ) -> List[dict]:
        '''Get the CAN messages for a given samples in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            can_bus_in: The CAN bus in the original dataset
            samples: The samples to get the CAN messages for.
            message_names: The names of the CAN messages to get.

        Returns:
            The CAN messages for the given samples in the original dataset.
            Each entry corresponds to the CAN message as returned by `NuScenesCanBus.get_messages()`.
        '''
        res = [
            {msg: self._get_can_message_for_sample(data_in, can_bus_in, smp, msg) for msg in message_names}
            for smp in samples
        ]
        return res

    def _get_can_message_for_sample(
        self, data_in: NuScenes, can_bus_in: NuScenesCanBus, samples: dict, message_name: str
    ) -> dict:
        '''Get the CAN message for a given sample in the original dataset.

        Args:
            data_in: The original dataset in the format used in the NuScenes DevKit
            can_bus_in: The CAN bus in the original dataset
            samples: The sample to get the CAN message for
            message_name: The name of the CAN message to get

        Returns:
            The CAN message for the given sample in the original dataset.
            Each entry corresponds to the CAN message as returned by `NuScenesCanBus.get_messages()`.
        '''
        scene_name = data_in.get('scene', samples['scene_token'])['name']
        sample_time = samples['timestamp']
        message_list = can_bus_in.get_messages(scene_name, message_name)
        for message in message_list:
            if message['utime'] > sample_time:
                break
        return message

    def _generate_projected_bboxes_annotations(
        self,
        lidar_calibration_data: List[dict],
        ego_poses_at_lidar_timestamps: List[dict],
        cam_calibration_data: List[List[dict]],
        ego_poses_at_cam_timestamps: List[List[dict]],
        objects_in_lidar_coords: List[List[nuscenes.utils.data_classes.Box]],
        annotations: List[List[dict]],
        image_size: Tuple[int, int],
    ) -> List[List[dict]]:
        '''Generate projected bounding boxes annotations for a sequence of samples (scene).

        The projected bounding boxes annotations are generated by projecting the 3D bounding boxes to 2D.
        The projection is done using the :class:`BboxProjector`.

        Args:
            lidar_calibration_data: The lidar calibration data for the sequence.
            ego_poses_at_lidar_timestamps: The ego poses at the lidar timestamps for the sequence.
            cam_calibration_data: The camera calibration data for the sequence.
            ego_poses_at_cam_timestamps: The ego poses at the camera timestamps for the sequence.
            objects_in_lidar_coords: The objects in the lidar coordinates for the sequence.
            annotations: The annotations for the samples in the sequence (in the original format).
            image_size: The size of the images to generate the projected bounding boxes annotations for.

        Returns:
            The projected bounding boxes annotations for the given camera sample data.
            Each entry corresponds to the projected bounding boxes annotation as returned by
            `BboxProjector.project_3d_boxes_to_2d()`.
        '''

        num_samples = len(objects_in_lidar_coords)
        num_cams = len(cam_calibration_data[0])

        res = [[None] * num_cams for _ in range(num_samples)]
        for i in range(num_samples):
            ego_pose_at_lidar_timestamp = ego_poses_at_lidar_timestamps[i]
            lidar_calib = lidar_calibration_data[i]
            objects_in_lidar_coords_for_sample = objects_in_lidar_coords[i]
            anno = annotations[i]
            for c in range(num_cams):
                cam_calib = cam_calibration_data[i][c]
                ego_pose_at_cam_timestamp = ego_poses_at_cam_timestamps[i][c]

                lidar_to_ego = get_sensor_to_ego_transformation(lidar_calib)
                ego_to_world_lidar = get_ego_to_world_transform(ego_pose_at_lidar_timestamp)
                cam_to_ego = get_sensor_to_ego_transformation(cam_calib)
                cam_intrinsics = get_cam_intrinsics(cam_calib)
                ego_to_world_cam = get_ego_to_world_transform(ego_pose_at_cam_timestamp)
                cam_extrinsics = np.linalg.inv(cam_to_ego)

                res[i][c] = BboxProjector.project_3d_boxes_to_2d(
                    lidar_to_ego,
                    ego_to_world_lidar,
                    objects_in_lidar_coords_for_sample,
                    cam_extrinsics,
                    cam_intrinsics,
                    ego_to_world_cam,
                    anno,
                    image_size,
                )
        return res

    def _get_image_data_annotations(self, image_annotations_mapped, cam_sample_data):
        '''Get image annotations

        Args:
            image_annotations_mapped: Mapped image annotations for the dataset as returned by `_build_image_annotations_map()`
            cam_sample_data: Cam sample data for the sequence. For a specific sample i and a specific camera c, the
                sample data is `cam_sample_data[i][c]`

        Returns:
            Annotions for all cameras in all samples of the sequence. For a specific sample i
            and a specific camera c, the image annotation can be accessed as `image_data_annotations[i][c]`
        '''
        res = [
            [
                (
                    list(image_annotations_mapped[csd['token']].values())
                    # If image_annotations_mapped contains an entry for the sample data token
                    # (may be not the case if there are no objects to annotate in the
                    # corresponding image), store it in the per-sample result
                    if csd['token'] in image_annotations_mapped
                    # Else, add an emply list to the per-sample results in order to preserve the
                    # index-camera correspondence inside the sample)
                    else []
                )
                # For each camera inside the sample
                for csd in cam_sample_data_for_sample
            ]
            # For each sample (each sample containing all cameras)
            for cam_sample_data_for_sample in cam_sample_data
        ]
        return res

    def _build_image_annotations_map(self, data_in: NuScenes) -> dict:
        '''Build a map mapping from sample data tokens to the corresponding 2D image annotations

        Args:
            data_in: The NuScenes dataset (in the format used in the NuScenes DevKit)

        Returns:
            Map mapping from sample data tokens to the 2D image annotations. If for a certain image, no
            annotations are present (no objects visible in the image), the map will not contain the corresponding
            sample data token as a key. For a specific image, the annotations can be obtained as:
            'annotations = annotation_map[sample_data_token]'
        '''
        res_map = {}
        image_annotations_in = data_in.image_annotations
        for ann in image_annotations_in:
            sample_token = ann['sample_data_token']
            annotation_token = ann['sample_annotation_token']
            # Add visibility level directly to avoid having to call 'data_in.get(...)' later when actually ocnstructing the outputs or to create an additional
            # list containing the visibilities
            ann['visibility_level'] = data_in.get('visibility', ann['visibility_token'])
            if not (sample_token in res_map):
                # Store the first annotation for a given `sample_token` (creating a dictionary used to hold the annotations)
                res_map[sample_token] = {annotation_token: ann}
            else:
                # Store subsequence annotations for a given `sample_token`, for which a dictionary containing annotations is already present
                res_map[sample_token][annotation_token] = ann
        return res_map
