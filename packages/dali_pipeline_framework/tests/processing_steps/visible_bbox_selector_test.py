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

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import VisibleBboxSelector


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Define image size (H, W)
        image_hw = np.array([100, 200], dtype=np.int32)

        # Define bounding boxes (min_x, min_y, max_x, max_y), integer coords for deterministic rasterization
        # Index 0: to be fully covered by two nearer boxes (indices 1 and 2)
        #   0: [10,10,40,40] depth=5.0  (farther)
        #   1: [10,10,25,40] depth=1.0  (nearer, left half)
        #   2: [25,10,40,40] depth=2.0  (nearer, right half)
        # Index 3: to be fully covered by a single nearer box (index 4)
        #   3: [60,60,80,80] depth=4.0  (farther)
        #   4: [50,50,90,90] depth=1.5  (nearer, fully covering 3)
        # Index 5: isolated visible box
        #   5: [0,0,5,5] depth=0.5      (nearest)
        # Index 6: too small to be visible
        #   6: [10,10,11,100] depth=0.1
        #   7: [10,10,100, 11] depth=0.2
        # Index 8: Outside the image
        #   8: [210,120,300,400] depth=0.3
        # Index 9: Partially overlapping, visible
        #   9: [110,10,150,60] depth=0.3
        #  10: [130, 5, 150, 70] depth=0.3
        bboxes = np.array(
            [
                [10, 10, 40, 40],
                [10, 10, 25, 40],
                [25, 10, 40, 40],
                [60, 60, 80, 80],
                [50, 50, 90, 90],
                [0, 0, 5, 5],
                [10, 10, 11, 100],
                [10, 10, 100, 11],
                [110, 120, 200, 300],
                [110, 10, 150, 60],
                [130, 5, 150, 70],
            ],
            dtype=np.float32,
        )

        # Depths: smaller = nearer
        depths = np.array([5.0, 1.0, 2.0, 4.0, 1.5, 0.5, 0.1, 0.2, 0.3, 0.3, 0.3], dtype=np.float32)

        res["bboxes"] = bboxes
        res["depths"] = depths
        res["image_hw"] = image_hw

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 1

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("bboxes", DALIDataType.FLOAT)
        res.add_data_field("depths", DALIDataType.FLOAT)
        res.add_data_field("image_hw", DALIDataType.INT32)
        return res


@pytest.mark.parametrize(
    "use_fixed_image_hw, check_for_bbox_occlusion, check_for_minimum_size",
    [
        (False, True, True),
        (True, True, False),
        (True, False, True),
    ],
    ids=[
        "occlusion_minimum_size",
        "occlusion_no_minimum_size_fixed_image_hw",
        "no_occlusion_minimum_size_fixed_image_hw",
    ],
)
def test_visible_bbox_selector_occlusions(
    use_fixed_image_hw, check_for_bbox_occlusion, check_for_minimum_size
):
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Use fixed image size; enable occlusion check only
    step = VisibleBboxSelector(
        bboxes_field_name="bboxes",
        depths_field_name="depths",
        resulting_mask_field_path="visible_mask",
        image_hw_field_name="image_hw" if not use_fixed_image_hw else None,
        image_hw=[100, 200] if use_fixed_image_hw else None,
        check_for_bbox_occlusion=check_for_bbox_occlusion,
        check_for_minimum_size=check_for_minimum_size,
        minimum_bbox_size=2 if check_for_minimum_size else None,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["bboxes", "depths", "image_hw"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    res = next(iter(iterator))

    # Expected visibility mask:
    #  - idx 0: `not check_for_bbox_occlusion` (fully covered by 1 and 2; ensure index 0 is invisible)
    #  - idx 1: `True`  (nearer, visible)
    #  - idx 2: `True`  (nearer, visible)
    #  - idx 3: `not check_for_bbox_occlusion` (fully covered by 4)
    #  - idx 4: `True`  (nearer, visible)
    #  - idx 5: `True`  (isolated, visible)
    #  - idx 6: `not check_for_minimum_size` (too small)
    #  - idx 7: `not check_for_minimum_size` (too small)
    #  - idx 8: `False` (outside the image)
    #  - idx 9: `True`  (partially overlapping, visible)
    #  - idx 10: `True`  (partially overlapping, visible)
    expected = torch.tensor(
        [
            not check_for_bbox_occlusion,
            True,
            True,
            not check_for_bbox_occlusion,
            True,
            True,
            not check_for_minimum_size,
            not check_for_minimum_size,
            False,
            True,
            True,
        ],
        dtype=torch.bool,
    )
    assert torch.equal(res["visible_mask"][0], expected)


if __name__ == "__main__":
    pytest.main([__file__])
