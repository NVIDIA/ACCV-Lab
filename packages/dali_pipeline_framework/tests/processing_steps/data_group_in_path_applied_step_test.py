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

"""
Test that a contained processing step is only applied to a selected sub-tree when
wrapped with `DataGroupInPathAppliedStep`.

We create a hierarchical `SampleDataGroup` where the same field name
(`"field_to_change"`) appears multiple times across different branches. The
contained step replaces all occurrences of that field name with a predefined
value. The wrapper ensures that only the occurrences inside the selected
sub-tree are affected.
"""

import pytest

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch

from nvidia.dali import fn, types
from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import PipelineStepBase, DataGroupInPathAppliedStep
from accvlab.dali_pipeline_framework.internal_helpers.helper_functions import get_as_data_node


class _ReplaceFieldsByName(PipelineStepBase):
    """Simple step that finds all fields with a given name and replaces their values.

    The replacement preserves the original dtype by casting the constant to the
    field's dtype. Shapes are preserved by using broadcasting via arithmetic with
    the original tensor.
    """

    def __init__(self, field_name_to_replace: str, replacement_value):
        self._field_name = field_name_to_replace
        self._replacement_value = replacement_value

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        paths = data.find_all_occurrences(self._field_name)
        for p in paths:
            parent = data.get_parent_of_path(p)
            field_name = p[-1]
            curr = parent[field_name]
            curr_dtype = parent.get_type_of_field(field_name)

            # Broadcast a scalar constant to current shape via arithmetic
            const_node = get_as_data_node(self._replacement_value)
            replaced = curr * 0 + const_node

            # Cast back to original dtype to keep data format identical
            replaced = fn.cast(replaced, dtype=curr_dtype)

            parent[field_name] = replaced
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        # No structural/type changes beyond value replacement; keep blueprint unchanged
        return data_empty


class _TestProvider(DataProvider):
    """Provides hierarchical data with repeated field names across branches."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Top-level occurrences
        res["field_to_change"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        res["top_other"] = np.array([100, 200], dtype=np.int32)

        # Group "a" (this will be our selected sub-tree for application)
        res["a"]["field_to_change"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        res["a"]["other_in_a"] = np.array([7, 8, 9], dtype=np.int32)

        # Nested inside "a"
        res["a"]["b"]["field_to_change"] = np.array([5.5, 6.5], dtype=np.float32)
        res["a"]["b"]["unrelated"] = np.array([0, 1, 0], dtype=np.int32)

        # Sibling branch outside selected sub-tree
        res["unrelated"]["field_to_change"] = np.array([9.0], dtype=np.float32)
        res["unrelated"]["x"] = np.array([42], dtype=np.int32)

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 4

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Declare top-level fields
        res.add_data_field("field_to_change", DALIDataType.FLOAT)
        res.add_data_field("top_other", DALIDataType.INT32)

        # Group "a" with nested group "b"
        a = SampleDataGroup()
        a.add_data_field("field_to_change", DALIDataType.FLOAT)
        a.add_data_field("other_in_a", DALIDataType.INT32)
        b = SampleDataGroup()
        b.add_data_field("field_to_change", DALIDataType.FLOAT)
        b.add_data_field("unrelated", DALIDataType.INT32)
        a.add_data_group_field("b", b)
        res.add_data_group_field("a", a)

        # Sibling group outside selected sub-tree
        unrelated = SampleDataGroup()
        unrelated.add_data_field("field_to_change", DALIDataType.FLOAT)
        unrelated.add_data_field("x", DALIDataType.INT32)
        res.add_data_group_field("unrelated", unrelated)

        return res


@pytest.mark.parametrize(
    "path_to_apply_to, expect_change_in_a",
    [
        ("a", True),
        (("a", "b"), False),
    ],
    ids=["apply_to_a", "apply_to_a_b"],
)
def test_data_group_in_path_applied_step_replaces_only_in_subtree(path_to_apply_to, expect_change_in_a):
    provider = _TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    replacement_value = 42.0
    contained = _ReplaceFieldsByName("field_to_change", replacement_value)
    step = DataGroupInPathAppliedStep(processing_step_to_apply=contained, path_to_apply_to=path_to_apply_to)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        prefetch_queue_depth=3,
        use_parallel_external_source=True,
        copy_external_source_passthrough_outputs=True,
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=2,
        prefetch_queue_depth=2,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(2, pipeline, pipeline_def.check_and_get_output_data_structure())
    res = next(iter(iterator))

    # Build expected originals from a provider instance
    expected = _TestProvider().get_data(0)

    # Changed only inside selected sub-tree
    a_field = res["a"]["field_to_change"][0]
    a_b_field = res["a"]["b"]["field_to_change"][0]
    if expect_change_in_a:
        assert torch.all(a_field == torch.full_like(a_field, fill_value=replacement_value))
    else:
        # unchanged when applying only to ["a", "b"]
        assert torch.equal(a_field, torch.from_numpy(expected["a"]["field_to_change"]))
    # a/b is changed in both parameterizations
    assert torch.all(a_b_field == torch.full_like(a_b_field, fill_value=replacement_value))

    # Unchanged outside the sub-tree
    top_field = res["field_to_change"][0]
    unrelated_field = res["unrelated"]["field_to_change"][0]
    assert torch.equal(top_field, torch.from_numpy(expected["field_to_change"]))
    assert torch.equal(unrelated_field, torch.from_numpy(expected["unrelated"]["field_to_change"]))

    # Other fields remain intact
    assert torch.equal(res["top_other"][0], torch.from_numpy(expected["top_other"]))
    assert torch.equal(res["a"]["other_in_a"][0], torch.from_numpy(expected["a"]["other_in_a"]))
    assert torch.equal(res["a"]["b"]["unrelated"][0], torch.from_numpy(expected["a"]["b"]["unrelated"]))
    assert torch.equal(res["unrelated"]["x"][0], torch.from_numpy(expected["unrelated"]["x"]))


if __name__ == "__main__":
    pytest.main([__file__])
