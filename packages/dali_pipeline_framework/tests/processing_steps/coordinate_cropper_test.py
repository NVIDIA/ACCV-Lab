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

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import CoordinateCropper


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Create test data with points in different ranges
        res["points_2d"] = np.array(
            [
                [0.0, 0.0],  # Inside range
                [-5.0, 10.0],  # X below min, Y above max
                [15.0, -3.0],  # X above max, Y below min
                [5.0, 5.0],  # Inside range
                [20.0, 20.0],  # Both above max
                [-10.0, -10.0],  # Both below min
            ]
        )
        res["points_3d"] = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside range
                [-5.0, 10.0, 15.0],  # X below min, Y above max, Z above max
                [15.0, -3.0, -8.0],  # X above max, Y below min, Z below min
                [5.0, 5.0, 5.0],  # Inside range
                [20.0, 20.0, 20.0],  # All above max
                [-10.0, -10.0, -10.0],  # All below min
            ]
        )
        res["bboxes"] = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],  # Inside range
                [-5.0, 5.0, 15.0, 15.0],  # Partially outside
                [5.0, -3.0, 20.0, 5.0],  # Partially outside
                [20.0, 20.0, 25.0, 25.0],  # Completely outside
                [-15.0, -15.0, -5.0, -5.0],  # Completely outside
                [2.0, 3.0, 8.0, 6.0],  # Inside range
            ]
        )
        res["other_data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("points_2d", DALIDataType.FLOAT)
        res.add_data_field("points_3d", DALIDataType.FLOAT)
        res.add_data_field("bboxes", DALIDataType.FLOAT)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        return res


@pytest.mark.parametrize(
    "points_field_name,minimum_point,maximum_point",
    [
        ("points_2d", [0.0, 2.0], [10.0, 8.0]),
        ("points_3d", [-2.0, 1.0, -3.0], [8.0, 6.0, 7.0]),
        ("bboxes", [0.0, 1.0, 2.0, 3.0], [10.0, 9.0, 8.0, 7.0]),
    ],
)
def test_coordinate_cropper(points_field_name, minimum_point, maximum_point):
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = CoordinateCropper(
        points_fields_name=points_field_name,
        minimum_point=minimum_point,
        maximum_point=maximum_point,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["points_2d", "points_3d", "bboxes", "other_data"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())

    iterator_iter = iter(iterator)

    res = next(iterator_iter)

    # Check that the shape remains the same (cropping doesn't change the number of points)
    # Note: batch dimension is added by DALI pipeline as dimension 0
    original_shape = res[points_field_name].shape
    assert original_shape[1] == 6  # 6 points in our test data

    # Check the cropped data based on the field type
    if points_field_name == "points_2d":
        # Expected: points should be clamped to [0, 10] for X and [2, 8] for Y
        expected_points = torch.tensor(
            [
                [0.0, 2.0],  # X=0 clamped to 0, Y=0 clamped to 2
                [0.0, 8.0],  # X=-5 clamped to 0, Y=10 clamped to 8
                [10.0, 2.0],  # X=15 clamped to 10, Y=-3 clamped to 2
                [5.0, 5.0],  # Inside range - unchanged
                [10.0, 8.0],  # X=20 clamped to 10, Y=20 clamped to 8
                [0.0, 2.0],  # X=-10 clamped to 0, Y=-10 clamped to 2
            ]
        )
        assert torch.equal(res[points_field_name][0], expected_points)

    elif points_field_name == "points_3d":
        # Expected: points should be clamped to [-2, 8] for X, [1, 6] for Y, [-3, 7] for Z
        expected_points = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # X=0 unchanged, Y=0 clamped to 1, Z=0 unchanged
                [-2.0, 6.0, 7.0],  # X=-5 clamped to -2, Y=10 clamped to 6, Z=15 clamped to 7
                [8.0, 1.0, -3.0],  # X=15 clamped to 8, Y=-3 clamped to 1, Z=-8 clamped to -3
                [5.0, 5.0, 5.0],  # Inside range - unchanged
                [8.0, 6.0, 7.0],  # X=20 clamped to 8, Y=20 clamped to 6, Z=20 clamped to 7
                [-2.0, 1.0, -3.0],  # X=-10 clamped to -2, Y=-10 clamped to 1, Z=-10 clamped to -3
            ]
        )
        assert torch.equal(res[points_field_name][0], expected_points)

    elif points_field_name == "bboxes":
        # Expected: bbox coordinates should be clamped to [0, 10] for X1, [1, 9] for Y1, [2, 8] for X2, [3, 7] for Y2
        expected_bboxes = torch.tensor(
            [
                # X1=0 unchanged, Y1=0 clamped to 1, X2=10 clamped to 8, Y2=10 clamped to 7
                [0.0, 1.0, 8.0, 7.0],
                # X1=-5 clamped to 0, Y1=5 unchanged, X2=15 clamped to 8, Y2=15 clamped to 7
                [0.0, 5.0, 8.0, 7.0],
                # X1=5 unchanged, Y1=-3 clamped to 1, X2=20 clamped to 8, Y2=5 unchanged
                [5.0, 1.0, 8.0, 5.0],
                # X1=20 clamped to 10, Y1=20 clamped to 9, X2=25 clamped to 8, Y2=25 clamped to 7
                [10.0, 9.0, 8.0, 7.0],
                # X1=-15 clamped to 0, Y1=-15 clamped to 1, X2=-5 clamped to 2, Y2=-5 clamped to 3
                [0.0, 1.0, 2.0, 3.0],
                # X1=2 unchanged, Y1=3 unchanged, X2=8 unchanged, Y2=6 unchanged
                [2.0, 3.0, 8.0, 6.0],
            ]
        )
        assert torch.equal(res[points_field_name][0], expected_bboxes)

    # Check that other_data is unchanged (not processed by the step)
    assert torch.equal(res["other_data"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


if __name__ == "__main__":
    # test_coordinate_cropper("points_2d", [0.0, 2.0], [10.0, 8.0])
    pytest.main([__file__])
