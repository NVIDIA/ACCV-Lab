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
from accvlab.dali_pipeline_framework.processing_steps import PointsInRangeCheck


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 1.0, 2.0],
                [10.0, -1.0, 2.0],
                [2.0, 20.0, 1.0],
                [5.0, 10.0, 3.0],
                [-10.0, 4.0, 4.0],
            ]
        )
        res["annotation"]["points"] = points
        res["annotation"]["labels"] = np.array([0, 1, 2, 3, 4, 5])
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
        annotation.add_data_field("other_field", DALIDataType.FLOAT)
        res.add_data_group_field("annotation", annotation)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        return res


def test_points_in_range_check():
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )
    step = PointsInRangeCheck(
        points_fields_name="points",
        is_inside_field_name="is_inside_range",
        minimum_point=[-5.0, -10.0, -3.0],
        maximum_point=[5.0, 10.0, 3.0],
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["labels", "other_field", "other_data"],
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
    assert torch.all(
        res["annotation"]["is_inside_range"][0] == torch.tensor([True, True, False, False, True, False])
    )


if __name__ == "__main__":
    test_points_in_range_check()
    pytest.main([__file__])
