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
Test that `DataGroupArrayInPathElementsAppliedStep` applies a contained step to
each child element of an array group independently.

We create an array data group field (e.g. `arr`) with multiple children. The
contained step replaces all occurrences of `"field_to_change"` within the input
sub-tree with a single random value sampled via DALI's `fn.random.uniform`. Since
the wrapper applies the contained step to each child independently, each child is
expected to receive a different replacement value. We use the fake random number
generator to make the sequence deterministic while ensuring distinct values.
"""

import pytest

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType

from _dali_fake_random_generator import DaliFakeRandomGenerator

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import (
    PipelineStepBase,
    DataGroupArrayInPathElementsAppliedStep,
)


def set_dali_uniform_generator_and_get_orig_and_replacement(sequences):
    generator = DaliFakeRandomGenerator(sequences)
    original_generator = fn.random.uniform
    fn.random.uniform = generator.get_generator()
    return original_generator, generator


def restore_generator(generator):
    fn.random.uniform = generator


class _ReplaceFieldsByNameWithRandom(PipelineStepBase):
    """Replace all occurrences of a field name with a single random value per sub-tree."""

    def __init__(self, field_name_to_replace: str, rand_range=(0.0, 1.0)):
        self._field_name = field_name_to_replace
        self._rand_range = rand_range

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        paths = data.find_all_occurrences(self._field_name)
        if len(paths) == 0:
            return data

        rand_val = fn.random.uniform(range=list(self._rand_range))

        for p in paths:
            parent = data.get_parent_of_path(p)
            field_name = p[-1]
            curr = parent[field_name]
            curr_dtype = parent.get_type_of_field(field_name)

            replaced = curr * 0 + rand_val
            replaced = fn.cast(replaced, dtype=curr_dtype)
            parent[field_name] = replaced
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        return data_empty


class _TestProvider(DataProvider):
    """Provides data with an array group `arr` that contains two children."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Field outside the array group, should remain unchanged
        res["field_to_change"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        res["top_other"] = np.array([100, 200], dtype=np.int32)

        # Array elements
        res["container"]["arr"][0]["field_to_change"] = np.array([10.0, 20.0], dtype=np.float32)
        res["container"]["arr"][0]["inner_other"] = np.array([7, 8, 9], dtype=np.int32)

        res["container"]["arr"][1]["field_to_change"] = np.array([5.0], dtype=np.float32)
        res["container"]["arr"][1]["inner_other"] = np.array([0, 1, 0], dtype=np.int32)
        res["container"]["unrelated"] = np.array([100, 200, 300], dtype=np.int32)

        # Unrelated sibling
        res["unrelated"] = np.array([42], dtype=np.int32)

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 4

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Top-level fields
        res.add_data_field("field_to_change", DALIDataType.FLOAT)
        res.add_data_field("top_other", DALIDataType.INT32)

        # Blueprint for array element
        elem_bp = SampleDataGroup()
        elem_bp.add_data_field("field_to_change", DALIDataType.FLOAT)
        elem_bp.add_data_field("inner_other", DALIDataType.INT32)

        # Add a container that has another array
        container = SampleDataGroup()
        container.add_data_group_field_array("arr", elem_bp, 2)
        container.add_data_field("unrelated", DALIDataType.INT32)
        res.add_data_group_field("container", container)

        # Unrelated
        res.add_data_field("unrelated", DALIDataType.INT32)

        return res


def test_data_group_array_in_path_elements_applied_step_independent_processing():
    # Configure fake random generator so subsequent calls yield distinct values for the same range
    sequences = [
        DaliFakeRandomGenerator.RangeReplacement([0.0, 1.0], [0.11, 0.77, 0.33, 0.55]),
        DaliFakeRandomGenerator.RangeReplacement(None, [0.25, 0.75]),
    ]
    orig_gen, _ = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        provider = _TestProvider()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        contained = _ReplaceFieldsByNameWithRandom("field_to_change", rand_range=(0.0, 1.0))
        step = DataGroupArrayInPathElementsAppliedStep(
            processing_step_to_apply=contained, path_to_array_to_apply_to=("container", "arr")
        )

        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[step],
            copy_external_source_passthrough_outputs=True,
        )

        pipeline = pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=2,
            prefetch_queue_depth=2,
            num_threads=1,
            py_start_method="spawn",
        )

        iterator = DALIStructuredOutputIterator(
            2, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        res = next(iter(iterator))

        # Expected originals from provider
        expected = _TestProvider().get_data(0)

        # Unchanged outside selected array elements
        assert torch.equal(
            res["field_to_change"][0], torch.from_numpy(expected["field_to_change"])
        ), "fields outside the selected array elements should remain unchanged"
        assert torch.equal(
            res["top_other"][0], torch.from_numpy(expected["top_other"])
        ), "top_other should remain unchanged"
        assert torch.equal(
            res["unrelated"][0], torch.from_numpy(expected["unrelated"])
        ), "unrelated should remain unchanged"

        # Both array children should be changed from originals
        arr0 = res["container"]["arr"][0]["field_to_change"][0]
        arr1 = res["container"]["arr"][1]["field_to_change"][0]

        assert not torch.equal(
            arr0, torch.from_numpy(expected["container"]["arr"][0]["field_to_change"])
        ), "container/arr[0]/field_to_change should be different from the original"
        assert not torch.equal(
            arr1, torch.from_numpy(expected["container"]["arr"][1]["field_to_change"])
        ), "container/arr[1]/field_to_change should be different from the original"

        # Ensure independent processing: replacement values differ between children
        val0 = arr0.float().mean().item()
        val1 = arr1.float().mean().item()
        assert (
            abs(val0 - val1) > 0.1
        ), "The two array elements should be different from each other (processed independently)"

        # Non-replaced fields inside the array children remain intact
        assert torch.equal(
            res["container"]["arr"][0]["inner_other"][0],
            torch.from_numpy(expected["container"]["arr"][0]["inner_other"]),
        ), "container/arr[0]/inner_other should remain unchanged"
        assert torch.equal(
            res["container"]["arr"][1]["inner_other"][0],
            torch.from_numpy(expected["container"]["arr"][1]["inner_other"]),
        ), "container/arr[1]/inner_other should remain unchanged"

    finally:
        restore_generator(orig_gen)


if __name__ == "__main__":
    pytest.main([__file__])
