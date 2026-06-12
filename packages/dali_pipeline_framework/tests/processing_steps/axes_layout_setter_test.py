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

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup, PipelineDefinition
from accvlab.dali_pipeline_framework.processing_steps import AxesLayoutSetter


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Create test data with different layouts
        res["image"] = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )  # Shape: (2, 2, 3) - HWC format
        res["image2"] = np.array(
            [
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
            ]
        )  # Shape: (2, 2, 3) - HWC format
        res["feature_map"] = np.array(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            ]
        )  # Shape: (2, 2, 2, 2) - NCHW format
        res["other_data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("image", DALIDataType.FLOAT)
        res.add_data_field("image2", DALIDataType.FLOAT)
        res.add_data_field("feature_map", DALIDataType.FLOAT)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        return res


@pytest.mark.parametrize(
    "field_names,layout_to_set",
    [
        ("image", "CHW"),
        ("feature_map", "NHWC"),
        (["image", "image2"], "HWC"),
    ],
)
def test_axes_layout_setter(field_names, layout_to_set):
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = AxesLayoutSetter(
        names_fields_to_set=field_names,
        layout_to_set=layout_to_set,
    )

    selected_field_names = {field_names} if isinstance(field_names, str) else set(field_names)
    passthrough_field_names = [
        name for name in ["image", "image2", "feature_map", "other_data"] if name not in selected_field_names
    ]

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=passthrough_field_names,
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    # Build the pipeline
    pipeline.build()

    # NOTE: Here, we are using the DALI pipeline directly, not the iterator.
    # This is done in order to retain the layout information, which is lost
    # when outputting the data as PyTorch tensors (which is done by the iterator).
    # Get the raw DALI output (tensor lists)
    output = pipeline.run()

    # Get the field names in the order they appear in the output
    field_names_flat = pipeline_def.check_and_get_output_data_structure().field_names_flat

    # Convert field_names to list if it's a string
    if isinstance(field_names, str):
        field_names = [field_names]

    # Check that the layout is correctly set for the specified fields
    for field_name in field_names:
        assert field_name in field_names_flat, f"Field '{field_name}' not found in the output"

        field_idx = field_names_flat.index(field_name)
        field_tensor_list = output[field_idx]

        # Check that we have a tensor list with one tensor (batch_size=1)
        assert len(field_tensor_list) == 1

        # Get the tensor and check its layout
        field_tensor = field_tensor_list[0]

        # Check that the layout is correctly set
        assert (
            field_tensor.layout() == layout_to_set
        ), f"Expected layout '{layout_to_set}' for field '{field_name}', got '{field_tensor.layout()}'"

        # Also check that the data values are preserved (layout setting doesn't change values)
        tensor_data = np.array(field_tensor)

        if field_name == "image":
            # Original shape was (2, 2, 3) - HWC format
            # No batch dimension when using pipeline.run() directly
            assert tensor_data.shape == (2, 2, 3)
            # Check that values are preserved
            expected_values = np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ]
            )
            assert np.allclose(tensor_data, expected_values)

        elif field_name == "image2":
            # Original shape was (2, 2, 3) - HWC format
            # No batch dimension when using pipeline.run() directly
            assert tensor_data.shape == (2, 2, 3)
            # Check that values are preserved
            expected_values = np.array(
                [
                    [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                    [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
                ]
            )
            assert np.allclose(tensor_data, expected_values)

        elif field_name == "feature_map":
            # Original shape was (2, 2, 2, 2) - NCHW format
            # No batch dimension when using pipeline.run() directly
            assert tensor_data.shape == (2, 2, 2, 2)
            # Check that values are preserved
            expected_values = np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                    [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
                ]
            )
            assert np.allclose(tensor_data, expected_values)

    # Check that other_data is unchanged (not processed by the step)
    other_data_idx = field_names_flat.index("other_data")
    other_data_tensor_list = output[other_data_idx]
    other_data_tensor = other_data_tensor_list[0]
    other_data_values = np.array(other_data_tensor)
    assert np.allclose(other_data_values, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


if __name__ == "__main__":
    # test_axes_layout_setter("image", "CHW")
    pytest.main([__file__])
