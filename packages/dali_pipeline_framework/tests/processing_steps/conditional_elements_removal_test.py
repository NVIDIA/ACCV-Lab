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
from accvlab.dali_pipeline_framework.processing_steps import ConditionalElementRemover


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Create test data with 6 objects
        res["annotation"]["points"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 1.0, 2.0],
                [10.0, -1.0, 2.0],
                [2.0, 20.0, 1.0],
                [5.0, 10.0, 3.0],
                [-10.0, 4.0, 4.0],
            ]
        )
        res["annotation"]["labels"] = np.array([0, 1, 2, 3, 4, 5])
        res["annotation"]["sizes"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
                [6.0, 7.0, 8.0],
            ]
        )
        # Add a 3D field to test different dimensions
        features = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0]],
                [[21.0, 22.0], [23.0, 24.0]],
            ]
        )
        features = np.transpose(features, (1, 0, 2))
        res["annotation"]["features"] = features
        res["annotation"]["is_active"] = np.array([True, True, False, False, True, False])
        res["annotation"]["other_field"] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        res["other_data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        annotation = SampleDataGroup()
        annotation.add_data_field("points", DALIDataType.FLOAT)
        annotation.add_data_field("labels", DALIDataType.INT32)
        annotation.add_data_field("sizes", DALIDataType.FLOAT)
        annotation.add_data_field("features", DALIDataType.FLOAT)
        annotation.add_data_field("is_active", DALIDataType.BOOL)
        annotation.add_data_field("other_field", DALIDataType.FLOAT)
        res.add_data_group_field("annotation", annotation)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        return res


@pytest.mark.parametrize("remove_mask_field", [True, False])
def test_conditional_element_removal(remove_mask_field):
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ConditionalElementRemover(
        annotation_field_name="annotation",
        mask_field_name="is_active",
        field_names_to_process=["points", "labels", "sizes", "features", "other_field"],
        field_dims_to_process=[0, 0, 0, 1, 0],  # Remove along different dimensions for different fields
        # points: 2D, labels: 1D, sizes: 2D, features: 3D, other_field: 1D
        fields_to_process_num_dims=[2, 1, 2, 3, 1],
        remove_mask_field=remove_mask_field,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["other_data", "is_active"],
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

    # Check that only 3 elements remain (indices 0, 1, 4 from original data)
    # Note: batch dimension is added by DALI pipeline as dimension 0
    assert res["annotation"]["points"].shape[1] == 3  # Dimension 1 (after batch dim 0)
    assert res["annotation"]["labels"].shape[1] == 3  # Dimension 1 (after batch dim 0)
    assert res["annotation"]["sizes"].shape[1] == 3  # Dimension 1 (after batch dim 0)
    assert res["annotation"]["features"].shape[2] == 3  # Dimension 2 for features (batch+3D array)
    assert res["annotation"]["other_field"].shape[1] == 3  # Dimension 1 (after batch dim 0)

    # Check the remaining data matches expected values
    # Original indices 0, 1, 4 should remain
    expected_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # index 0
            [-1.0, 1.0, 2.0],  # index 1
            [5.0, 10.0, 3.0],  # index 4
        ]
    )
    expected_labels = torch.tensor([0, 1, 4], dtype=torch.int32)  # indices 0, 1, 4
    expected_sizes = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # index 0
            [2.0, 3.0, 4.0],  # index 1
            [5.0, 6.0, 7.0],  # index 4
        ]
    )
    # For features, we remove along dimension 1, so we keep the first and third elements along that dimension
    expected_features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],  # index 0
            [[5.0, 6.0], [7.0, 8.0]],  # index 1
            [[17.0, 18.0], [19.0, 20.0]],  # index 4
        ]
    )
    expected_features = np.transpose(expected_features, (1, 0, 2))
    expected_other_field = torch.tensor([10.0, 20.0, 50.0])  # indices 0, 1, 4

    assert torch.equal(res["annotation"]["points"][0], expected_points)
    assert torch.equal(res["annotation"]["labels"][0], expected_labels)
    assert torch.equal(res["annotation"]["sizes"][0], expected_sizes)
    assert torch.equal(res["annotation"]["features"][0], expected_features)
    assert torch.equal(res["annotation"]["other_field"][0], expected_other_field)

    # Check mask field based on remove_mask_field flag
    if remove_mask_field:
        # Check that the mask field was removed
        assert "is_active" not in res["annotation"]
    else:
        # Check that the mask field is still present and contains only True values
        assert "is_active" in res["annotation"]
        assert torch.all(
            res["annotation"]["is_active"][0] == torch.tensor([True, True, False, False, True, False])
        )

    # Check that other_data is unchanged (not processed by the step)
    assert torch.equal(res["other_data"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


if __name__ == "__main__":
    # test_conditional_element_removal()
    pytest.main([__file__])
