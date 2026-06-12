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

import pytest
import numpy as np
import torch

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType

from _dali_fake_random_generator import DaliFakeRandomGenerator

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import BEVBBoxesTransformer3D


def set_dali_uniform_generator_and_get_orig_and_replacement(sequences):
    generator = DaliFakeRandomGenerator(sequences)
    original_generator = fn.random.uniform
    fn.random.uniform = generator.get_generator()
    return original_generator, generator


def restore_generator(generator):
    fn.random.uniform = generator


class TestProvider(DataProvider):
    """Test data provider for BEVBBoxesTransformer3D tests."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create test data with 3D points (bounding box centers)
        res["points"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )

        # Velocities for the objects
        res["velocities"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        )

        # Sizes of bounding boxes (length, width, height)
        res["sizes"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [2.0, 1.5, 1.0],
                [3.0, 2.0, 1.5],
                [2.5, 1.8, 1.2],
            ],
            dtype=np.float32,
        )

        # Orientations (rotation around z-axis in radians)
        res["orientations"] = np.array(
            [
                0.0,
                -np.pi + 0.0001,
                np.pi - 0.0001,
                -np.pi / 2,
                np.pi / 3,
                -np.pi / 6,
                np.pi,
            ],
            dtype=np.float32,
        )

        # Ego to world transformation matrix (4x4 homogeneous)
        res["ego_to_world"] = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # World to ego transformation matrix (inverse of ego_to_world)
        res["world_to_ego"] = np.linalg.inv(res["ego_to_world"])

        # Projection matrix (3x4)
        # Note that in order to ensure that the points are in view, we move them to the front.
        proj_mat = np.array(
            [
                [1000.0, 0.0, 320.0, 0.0],
                [0.0, 1000.0, 240.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        move_points_to_front = np.eye(4, dtype=np.float32)
        move_points_to_front[2, 3] = 100.0
        proj_mat_front = proj_mat @ move_points_to_front
        res["proj_matrix"] = proj_mat_front

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 5

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("points", DALIDataType.FLOAT)
        res.add_data_field("velocities", DALIDataType.FLOAT)
        res.add_data_field("sizes", DALIDataType.FLOAT)
        res.add_data_field("orientations", DALIDataType.FLOAT)
        res.add_data_field("ego_to_world", DALIDataType.FLOAT)
        res.add_data_field("world_to_ego", DALIDataType.FLOAT)
        res.add_data_field("proj_matrix", DALIDataType.FLOAT)
        return res


def create_reference_transformation_matrices(rotation_angle_rad, rotation_axis, scaling_factor, translation):
    """Create reference transformation matrices for testing."""

    # Rotation matrix around Z-axis (axis 2)
    cos_angle = np.cos(rotation_angle_rad)
    sin_angle = np.sin(rotation_angle_rad)
    if rotation_axis == 0:
        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos_angle, -sin_angle, 0.0],
                [0.0, sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    elif rotation_axis == 1:
        rotation_matrix = np.array(
            [
                [cos_angle, 0.0, sin_angle, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_angle, 0.0, cos_angle, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    elif rotation_axis == 2:
        rotation_matrix = np.array(
            [
                [cos_angle, -sin_angle, 0.0, 0.0],
                [sin_angle, cos_angle, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # Scaling matrix
    scaling_matrix = np.array(
        [
            [scaling_factor, 0.0, 0.0, 0.0],
            [0.0, scaling_factor, 0.0, 0.0],
            [0.0, 0.0, scaling_factor, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Translation matrix
    translation_matrix = np.array(
        [
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return rotation_matrix, scaling_matrix, translation_matrix


def apply_transformation_to_ego_to_world(ego_to_world, rotation_matrix, scaling_matrix, translation_matrix):
    """Reference for BEVBBoxesTransformer3D ego_to_world updates (rotation, then scaling, then translation)."""

    res = ego_to_world.copy()
    res = res @ rotation_matrix.T
    res = res @ np.linalg.inv(scaling_matrix)
    res = res @ np.linalg.inv(translation_matrix)
    return res


def apply_transformation_to_world_to_ego(world_to_ego, rotation_matrix, scaling_matrix, translation_matrix):
    """Reference for BEVBBoxesTransformer3D world_to_ego updates (rotation, then scaling, then translation)."""

    res = world_to_ego.copy()
    res = rotation_matrix @ res
    res = scaling_matrix @ res
    res = translation_matrix @ res
    return res


def apply_transformation_to_vecs(vecs, rotation_matrix, scaling_matrix, translation_matrix, are_points):
    """Apply transformations to points using reference implementation."""

    # The actual implementation applies transformations in ego coordinates
    # So we need to transform points to ego coordinates first, apply transformations, then back to world

    # Transpose points to ensure each point is a column vector
    vecs = vecs.T

    # Convert points to homogeneous coordinates
    if are_points:
        vecs_homog = np.concatenate([vecs, np.ones((1, vecs.shape[1]))], axis=0)
    else:
        vecs_homog = np.concatenate([vecs, np.zeros((1, vecs.shape[1]))], axis=0)

    # Apply transformations in ego coordinates
    # The order is: rotation -> scaling -> translation
    # For points: from_left, not_inverted, transposed=True, make_homog=True

    # Apply the transformations in the correct order
    vecs_transformed_homog = translation_matrix @ scaling_matrix @ rotation_matrix @ vecs_homog

    # Convert back from homogeneous coordinates
    if are_points:
        vecs_transformed = vecs_transformed_homog[:3, :] / vecs_transformed_homog[3:, :]
    else:
        vecs_transformed = vecs_transformed_homog[:3, :]

    vecs_transformed = vecs_transformed.T

    return vecs_transformed


def apply_transformation_to_sizes(sizes, scaling_matrix):
    """Apply transformations to sizes using reference implementation."""

    res = apply_transformation_to_vecs(sizes, np.eye(4), scaling_matrix, np.eye(4), False)
    return res


def apply_transformation_to_velocities(velocities, rotation_matrix, scaling_matrix):
    """Apply transformations to velocities using reference implementation."""
    res = apply_transformation_to_vecs(velocities, rotation_matrix, scaling_matrix, np.eye(4), False)
    return res


def apply_transformation_to_points(points, rotation_matrix, scaling_matrix, translation_matrix):
    """Apply transformations to points using reference implementation."""
    res = apply_transformation_to_vecs(points, rotation_matrix, scaling_matrix, translation_matrix, True)
    return res


def verify_matrix_inverse_relationship(ego_to_world, world_to_ego, tolerance=1e-6):
    """Verify that world_to_ego is the inverse of ego_to_world."""

    expected_world_to_ego = torch.linalg.inv(ego_to_world)
    assert torch.allclose(world_to_ego, expected_world_to_ego, atol=tolerance), (
        "world_to_ego does not match inv(ego_to_world). "
        f"Max difference: {torch.max(torch.abs(world_to_ego - expected_world_to_ego)).item()}"
    )

    # Check that world_to_ego @ ego_to_world = identity
    identity_check = world_to_ego @ ego_to_world
    expected_identity = torch.eye(4, dtype=torch.float32)

    assert torch.allclose(
        identity_check, expected_identity, atol=tolerance
    ), f"world_to_ego is not the inverse of ego_to_world. Max difference: {torch.max(torch.abs(identity_check - expected_identity)).item()}"


def verify_transformation_consistency(
    original_points, original_transformation, transformed_points, transformed_transformation, tolerance=1e-6
):
    """Verify that original points with original transformation correspond to transformed points with transformed transformation."""

    original_points = original_points.T
    transformed_points = transformed_points.T

    # Convert points to homogeneous coordinates
    original_points_homog = np.concatenate([original_points, np.ones((1, original_points.shape[1]))], axis=0)
    transformed_points_homog = np.concatenate(
        [transformed_points, np.ones((1, transformed_points.shape[1]))], axis=0
    )

    # Project original points with original matrix
    original_transformed = original_transformation @ original_points_homog
    original_transformed = original_transformed[:-1, :] / original_transformed[-1, :]  # Perspective division
    original_transformed = torch.tensor(original_transformed, dtype=torch.float32)

    # Project transformed points with transformed matrix
    transformed_transformed = transformed_transformation @ transformed_points_homog
    transformed_transformed = (
        transformed_transformed[:-1, :] / transformed_transformed[-1, :]
    )  # Perspective division
    transformed_transformed = torch.tensor(transformed_transformed, dtype=torch.float32)

    # For projections, some of the points may be `NaN` (as we do not ensure that the points are in the
    # view frustum, and especially that they are not at distance 0 to the camera). We replace them with 0.0.
    # original_transformed = torch.nan_to_num(original_transformed, nan=0.0)
    # transformed_transformed = torch.nan_to_num(transformed_transformed, nan=0.0)

    # The projections should be the same
    assert torch.allclose(
        original_transformed, transformed_transformed, atol=tolerance
    ), f"Transformation consistency check failed. Max difference: {torch.max(torch.abs(original_transformed - transformed_transformed))}"


def verify(
    points_in,
    vels_in,
    sizes_in,
    orientations_in,
    proj_matrix_in,
    ego_to_world_in,
    world_to_ego_in,
    points_out,
    vels_out,
    sizes_out,
    orientations_out,
    proj_matrix_out,
    ego_to_world_out,
    world_to_ego_out,
    rotation,
    rotation_axis,
    scaling,
    translation,
    tolerance=1e-6,
):
    """Verify that the transformation is consistent."""

    rotation_matrix, scaling_matrix, translation_matrix = create_reference_transformation_matrices(
        rotation, rotation_axis, scaling, translation
    )

    points_ref = torch.tensor(
        apply_transformation_to_points(points_in, rotation_matrix, scaling_matrix, translation_matrix),
        dtype=torch.float32,
    )
    vels_ref = torch.tensor(
        apply_transformation_to_velocities(vels_in, rotation_matrix, scaling_matrix),
        dtype=torch.float32,
    )
    sizes_ref = torch.tensor(apply_transformation_to_sizes(sizes_in, scaling_matrix), dtype=torch.float32)
    orientations_ref = torch.tensor(orientations_in + rotation, dtype=torch.float32)
    # Bring the orientations into the range [-pi, pi]
    # Note that while using `while` loops is not the most efficient way to do this, it is used in the test
    # for simplicity.
    while torch.any(orientations_ref < -np.pi):
        orientations_ref[orientations_ref < -np.pi] += 2.0 * np.pi
    while torch.any(orientations_ref > np.pi):
        orientations_ref[orientations_ref > np.pi] -= 2.0 * np.pi

    assert torch.allclose(
        points_ref, points_out, atol=tolerance
    ), f"Points transformation does not match reference implementation"
    assert torch.allclose(
        vels_ref, vels_out, atol=tolerance
    ), f"Velocities transformation does not match reference implementation"
    assert torch.allclose(
        sizes_ref, sizes_out, atol=tolerance
    ), f"Sizes transformation does not match reference implementation"
    # Note that a small numerical error may occur when the orientation is close to -pi or pi, leading to
    # a difference of 2*pi. Therefore, we need to allow differences of either 2*pi or 0 (+- tolerance).
    abs_diff_orientations = torch.abs(orientations_ref - orientations_out)
    abs_diff_orientations_2pi_shifted = torch.abs(abs_diff_orientations - 2.0 * np.pi)
    abs_diff_orientations = torch.min(abs_diff_orientations, abs_diff_orientations_2pi_shifted)
    assert torch.allclose(
        abs_diff_orientations, torch.zeros_like(abs_diff_orientations), atol=tolerance
    ), f"Orientations transformation does not match reference implementation"

    ego_to_world_ref = torch.tensor(
        apply_transformation_to_ego_to_world(
            ego_to_world_in, rotation_matrix, scaling_matrix, translation_matrix
        ),
        dtype=torch.float32,
    )
    world_to_ego_ref = torch.tensor(
        apply_transformation_to_world_to_ego(
            world_to_ego_in, rotation_matrix, scaling_matrix, translation_matrix
        ),
        dtype=torch.float32,
    )
    assert torch.allclose(ego_to_world_ref, ego_to_world_out, atol=tolerance), (
        "ego_to_world transformation does not match reference implementation. "
        f"Max difference: {torch.max(torch.abs(ego_to_world_ref - ego_to_world_out)).item()}"
    )
    assert torch.allclose(world_to_ego_ref, world_to_ego_out, atol=tolerance), (
        "world_to_ego transformation does not match reference implementation. "
        f"Max difference: {torch.max(torch.abs(world_to_ego_ref - world_to_ego_out)).item()}"
    )

    verify_transformation_consistency(points_in, proj_matrix_in, points_out, proj_matrix_out, tolerance)
    verify_transformation_consistency(points_in, ego_to_world_in, points_out, ego_to_world_out, tolerance)
    verify_matrix_inverse_relationship(ego_to_world_out, world_to_ego_out, tolerance)


def run_transformation_test_with_reference_comparison(
    rotation_range,
    rotation_axis,
    scaling_range,
    translation_max_abs,
    expected_rotation_angle,
    expected_scaling_factor,
    expected_translation,
):
    """Run a transformation test and compare results with reference implementation."""

    # Set up mock random generator to get predictable transformations
    RangeRepl = DaliFakeRandomGenerator.RangeReplacement
    sequences = []

    # Add rotation range replacement if rotation is enabled
    if rotation_range is not None:
        rotation_range_repl = RangeRepl(rotation_range, [expected_rotation_angle])
        sequences.append(rotation_range_repl)

    # Add scaling range replacement if scaling is enabled
    if scaling_range is not None:
        scaling_range_repl = RangeRepl(scaling_range, [expected_scaling_factor])
        sequences.append(scaling_range_repl)

    # Add translation range replacements if translation is enabled
    if translation_max_abs is not None:
        translation_range_repl_x = RangeRepl(
            [-translation_max_abs[0], translation_max_abs[0]], [expected_translation[0]]
        )
        translation_range_repl_y = RangeRepl(
            [-translation_max_abs[1], translation_max_abs[1]], [expected_translation[1]]
        )
        translation_range_repl_z = RangeRepl(
            [-translation_max_abs[2], translation_max_abs[2]], [expected_translation[2]]
        )
        sequences.extend([translation_range_repl_x, translation_range_repl_y, translation_range_repl_z])
    original_gen, fake_gen = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        provider = TestProvider()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = BEVBBoxesTransformer3D(
            data_field_names_points="points",
            data_field_names_velocities="velocities",
            data_field_names_sizes="sizes",
            data_field_names_orientation="orientations",
            data_field_names_proj_matrices_and_extrinsics="proj_matrix",
            data_field_names_ego_to_world="ego_to_world",
            data_field_names_world_to_ego="world_to_ego",
            rotation_range=rotation_range,
            rotation_axis=2,  # Z-axis
            scaling_range=scaling_range,
            translation_max_abs=translation_max_abs,
        )

        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[step],
        )

        pipeline = pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
            exec_dynamic=True,
        )

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        result = next(iterator_iter)

        # Get original data for comparison
        original_data = provider.get_data(0)

        # Note that `result` has a batch dimension (as it is a DALI iterator output), so we need to index
        # with [0] to get the single sample. `original_data` does not have a batch dimension as it is the
        # input data for a single sample.
        verify(
            original_data["points"],
            original_data["velocities"],
            original_data["sizes"],
            original_data["orientations"],
            original_data["proj_matrix"],
            original_data["ego_to_world"],
            original_data["world_to_ego"],
            result["points"][0],
            result["velocities"][0],
            result["sizes"][0],
            result["orientations"][0],
            result["proj_matrix"][0],
            result["ego_to_world"][0],
            result["world_to_ego"][0],
            expected_rotation_angle,
            rotation_axis,
            expected_scaling_factor,
            expected_translation,
        )
    finally:
        restore_generator(original_gen)


def test_combined_transformations():
    """Test BEVBBoxesTransformer3D with combined transformations."""
    rotation_angle = np.pi / 2  # 90 degrees
    scaling_factor = 1.5
    translation = [0.5, 1.0, 1.5]
    run_transformation_test_with_reference_comparison(
        rotation_range=[-rotation_angle * 2.0, rotation_angle * 2.0],
        scaling_range=[0.3, 3.0],
        rotation_axis=2,
        translation_max_abs=[t * 2.0 for t in translation],
        expected_rotation_angle=rotation_angle,
        expected_scaling_factor=scaling_factor,
        expected_translation=translation,
    )


def test_no_translation():
    """Test BEVBBoxesTransformer3D with translation disabled (None)."""
    rotation_angle = np.pi / 4  # 45 degrees
    scaling_factor = 2.0
    run_transformation_test_with_reference_comparison(
        rotation_range=[-rotation_angle * 2.0, rotation_angle * 2.0],
        rotation_axis=2,
        scaling_range=[0.3, 3.0],
        translation_max_abs=None,
        expected_rotation_angle=rotation_angle,
        expected_scaling_factor=scaling_factor,
        expected_translation=[0.0, 0.0, 0.0],
    )


def test_only_translation():
    """Test BEVBBoxesTransformer3D with only translation enabled."""
    translation = [0.5, 1.5, 2.5]
    run_transformation_test_with_reference_comparison(
        rotation_range=None,
        rotation_axis=2,  # Should be ignored when rotation_range is None
        scaling_range=None,
        translation_max_abs=[t * 2.0 for t in translation],
        expected_rotation_angle=0.0,
        expected_scaling_factor=1.0,
        expected_translation=translation,
    )


# Edge case tests for missing data types
class TestProviderPointsOnly(DataProvider):
    """Test data provider with only points data."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        res["points"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 1

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("points", DALIDataType.FLOAT)
        return res


def test_points_only():
    """Test BEVBBoxesTransformer3D with only points data present."""
    rotation_angle = np.pi / 6  # 30 degrees
    scaling_factor = 1.5
    translation = [1.0, 2.0, 3.0]

    # Set up mock random generator
    RangeRepl = DaliFakeRandomGenerator.RangeReplacement
    rotation_range_repl = RangeRepl([-rotation_angle * 2.0, rotation_angle * 2.0], [rotation_angle])
    scaling_range_repl = RangeRepl([0.3, 3.0], [scaling_factor])
    translation_range_repl_x = RangeRepl([-translation[0] * 2.0, translation[0] * 2.0], [translation[0]])
    translation_range_repl_y = RangeRepl([-translation[1] * 2.0, translation[1] * 2.0], [translation[1]])
    translation_range_repl_z = RangeRepl([-translation[2] * 2.0, translation[2] * 2.0], [translation[2]])
    sequences = [
        rotation_range_repl,
        scaling_range_repl,
        translation_range_repl_x,
        translation_range_repl_y,
        translation_range_repl_z,
    ]
    original_gen, fake_gen = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        provider = TestProviderPointsOnly()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = BEVBBoxesTransformer3D(
            data_field_names_points="points",
            data_field_names_velocities=None,
            data_field_names_sizes=None,
            data_field_names_orientation=None,
            data_field_names_proj_matrices_and_extrinsics=None,
            data_field_names_ego_to_world=None,
            data_field_names_world_to_ego=None,
            rotation_range=[-rotation_angle * 2.0, rotation_angle * 2.0],
            rotation_axis=2,
            scaling_range=[0.3, 3.0],
            translation_max_abs=[t * 2.0 for t in translation],
        )

        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[step],
        )

        pipeline = pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
            exec_dynamic=True,
        )

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        result = next(iterator_iter)

        # Verify that points were transformed
        original_data = provider.get_data(0)
        rotation_matrix, scaling_matrix, translation_matrix = create_reference_transformation_matrices(
            rotation_angle, 2, scaling_factor, translation
        )
        expected_points = torch.tensor(
            apply_transformation_to_points(
                original_data["points"], rotation_matrix, scaling_matrix, translation_matrix
            ),
            dtype=torch.float32,
        )

        assert torch.allclose(
            result["points"][0], expected_points, atol=1e-5
        ), "Points transformation failed when only points data is present"

    finally:
        restore_generator(original_gen)


class TestProviderNoPoints(DataProvider):
    """Test data provider without points data."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        res["velocities"] = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )
        res["sizes"] = np.array(
            [
                [2.0, 1.5, 1.0],
                [3.0, 2.0, 1.5],
            ],
            dtype=np.float32,
        )
        res["orientations"] = np.array([0.0, np.pi / 4], dtype=np.float32)
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 1

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("velocities", DALIDataType.FLOAT)
        res.add_data_field("sizes", DALIDataType.FLOAT)
        res.add_data_field("orientations", DALIDataType.FLOAT)
        return res


def test_no_points_no_matrices():
    """Test BEVBBoxesTransformer3D without points data."""
    rotation_angle = np.pi / 4  # 45 degrees
    scaling_factor = 2.0

    # Set up mock random generator
    RangeRepl = DaliFakeRandomGenerator.RangeReplacement
    rotation_range_repl = RangeRepl([-rotation_angle * 2.0, rotation_angle * 2.0], [rotation_angle])
    scaling_range_repl = RangeRepl([0.3, 3.0], [scaling_factor])
    sequences = [rotation_range_repl, scaling_range_repl]
    original_gen, fake_gen = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        provider = TestProviderNoPoints()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = BEVBBoxesTransformer3D(
            data_field_names_points=None,
            data_field_names_velocities="velocities",
            data_field_names_sizes="sizes",
            data_field_names_orientation="orientations",
            data_field_names_proj_matrices_and_extrinsics=None,
            data_field_names_ego_to_world=None,
            data_field_names_world_to_ego=None,
            rotation_range=[-rotation_angle * 2.0, rotation_angle * 2.0],
            rotation_axis=2,
            scaling_range=[0.3, 3.0],
            translation_max_abs=None,
        )

        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[step],
        )

        pipeline = pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
            exec_dynamic=True,
        )

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        result = next(iterator_iter)

        # Verify that velocities, sizes, and orientations were transformed
        original_data = provider.get_data(0)
        rotation_matrix, scaling_matrix, translation_matrix = create_reference_transformation_matrices(
            rotation_angle, 2, scaling_factor, [0.0, 0.0, 0.0]
        )

        expected_velocities = torch.tensor(
            apply_transformation_to_velocities(original_data["velocities"], rotation_matrix, scaling_matrix),
            dtype=torch.float32,
        )
        expected_sizes = torch.tensor(
            apply_transformation_to_sizes(original_data["sizes"], scaling_matrix),
            dtype=torch.float32,
        )
        expected_orientations = torch.tensor(
            original_data["orientations"] + rotation_angle, dtype=torch.float32
        )

        assert torch.allclose(
            result["velocities"][0], expected_velocities, atol=1e-5
        ), "Velocities transformation failed when points data is not present"
        assert torch.allclose(
            result["sizes"][0], expected_sizes, atol=1e-5
        ), "Sizes transformation failed when points data is not present"
        assert torch.allclose(
            result["orientations"][0], expected_orientations, atol=1e-5
        ), "Orientations transformation failed when points data is not present"

    finally:
        restore_generator(original_gen)


def test_no_transformations_enabled():
    """Test BEVBBoxesTransformer3D with all transformations disabled."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = BEVBBoxesTransformer3D(
        data_field_names_points="points",
        data_field_names_velocities="velocities",
        data_field_names_sizes="sizes",
        data_field_names_orientation="orientations",
        data_field_names_proj_matrices_and_extrinsics="proj_matrix",
        data_field_names_ego_to_world="ego_to_world",
        data_field_names_world_to_ego="world_to_ego",
        rotation_range=None,
        rotation_axis=2,  # Should be ignored when rotation_range is None
        scaling_range=None,
        translation_max_abs=None,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        # No copy selectors: all output fields may pass through unchanged when no
        # transformations are enabled.
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
        exec_dynamic=True,
    )

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    result = next(iterator_iter)

    # Get original data for comparison
    original_data = provider.get_data(0)

    # Verify that data remains unchanged when no transformations are enabled
    assert torch.allclose(
        result["points"][0], torch.tensor(original_data["points"], dtype=torch.float32), atol=1e-5
    ), "Points should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["velocities"][0], torch.tensor(original_data["velocities"], dtype=torch.float32), atol=1e-5
    ), "Velocities should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["sizes"][0], torch.tensor(original_data["sizes"], dtype=torch.float32), atol=1e-5
    ), "Sizes should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["orientations"][0], torch.tensor(original_data["orientations"], dtype=torch.float32), atol=1e-5
    ), "Orientations should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["proj_matrix"][0], torch.tensor(original_data["proj_matrix"], dtype=torch.float32), atol=1e-5
    ), "Projection matrix should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["ego_to_world"][0], torch.tensor(original_data["ego_to_world"], dtype=torch.float32), atol=1e-5
    ), "Ego-to-world matrix should remain unchanged when no transformations are enabled"
    assert torch.allclose(
        result["world_to_ego"][0], torch.tensor(original_data["world_to_ego"], dtype=torch.float32), atol=1e-5
    ), "World-to-ego matrix should remain unchanged when no transformations are enabled"


if __name__ == "__main__":
    pytest.main([__file__])
