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
Test that `DataGroupsWithNameAppliedStep` applies a contained step to all sub-trees
with a specific name and that each sub-tree is processed independently.

We create two sub-trees with the same name (`"target_group"`). The contained step
replaces all occurrences of `"field_to_change"` within its sub-tree with a single
random value sampled via DALI's `fn.random.uniform`. Using a fake random generator,
we ensure deterministic but distinct values for subsequent calls so that each
sub-tree receives a different replacement value. We do not rely on ordering;
instead, we assert that the resulting values differ between the two sub-trees and
that fields outside the selected sub-trees remain unchanged.
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
from accvlab.dali_pipeline_framework.processing_steps import PipelineStepBase, DataGroupsWithNameAppliedStep
from accvlab.dali_pipeline_framework.internal_helpers.helper_functions import get_as_data_node


def set_dali_uniform_generator_and_get_orig_and_replacement(sequences):
    generator = DaliFakeRandomGenerator(sequences)
    original_generator = fn.random.uniform
    fn.random.uniform = generator.get_generator()
    return original_generator, generator


def restore_generator(generator):
    fn.random.uniform = generator


class _ReplaceFieldsByNameWithRandom(PipelineStepBase):
    """Replace all occurrences of a field name with a single random value per sub-tree.

    One random scalar is sampled once per `process()` call and used to replace all
    matching fields within the provided sub-tree. This ensures independent values
    for each sub-tree when wrapped by `DataGroupsWithNameAppliedStep`.
    """

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

            # Broadcast scalar to current shape via arithmetic
            replaced = curr * 0 + rand_val

            # Cast back to original dtype to keep data format identical
            replaced = fn.cast(replaced, dtype=curr_dtype)

            parent[field_name] = replaced
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        return data_empty


class _TestProvider(DataProvider):
    """Provides a data structure with two sub-trees named `target_group`."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Top-level field that should remain unchanged
        res["field_to_change"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        res["top_other"] = np.array([100, 200], dtype=np.int32)

        # First target group (top-level)
        res["target_group"]["field_to_change"] = np.array([10.0, 20.0], dtype=np.float32)
        res["target_group"]["inner_other"] = np.array([7, 8, 9], dtype=np.int32)

        # Second target group, nested under a container
        res["container"]["target_group"]["field_to_change"] = np.array([5.0], dtype=np.float32)
        res["container"]["target_group"]["inner_other"] = np.array([0, 1, 0], dtype=np.int32)

        # Some unrelated sibling data
        res["container"]["unrelated"] = np.array([42], dtype=np.int32)

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

        # Target group blueprint
        target_group_bp = SampleDataGroup()
        target_group_bp.add_data_field("field_to_change", DALIDataType.FLOAT)
        target_group_bp.add_data_field("inner_other", DALIDataType.INT32)

        # Add a top-level target group
        res.add_data_group_field("target_group", target_group_bp)

        # Add a container that has another target group
        container = SampleDataGroup()
        container.add_data_group_field("target_group", target_group_bp)
        container.add_data_field("unrelated", DALIDataType.INT32)
        res.add_data_group_field("container", container)

        return res


def test_data_groups_with_name_applied_step_independent_processing():
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
        step = DataGroupsWithNameAppliedStep(
            processing_step_to_apply=contained, names_of_groups_to_apply_to="target_group"
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

        # Unchanged outside selected sub-trees
        assert torch.equal(
            res["field_to_change"][0], torch.from_numpy(expected["field_to_change"])
        ), "fields outside the selected sub-trees should remain unchanged"
        assert torch.equal(
            res["top_other"][0], torch.from_numpy(expected["top_other"])
        ), "top_other should remain unchanged"
        assert torch.equal(
            res["container"]["unrelated"][0], torch.from_numpy(expected["container"]["unrelated"])
        ), "unrelated should remain unchanged"

        # Both target_group sub-trees should be changed from originals
        tg1 = res["target_group"]["field_to_change"][0]
        tg2 = res["container"]["target_group"]["field_to_change"][0]

        # The two target_group sub-trees should be different from the originals
        assert not torch.equal(
            tg1, torch.from_numpy(expected["target_group"]["field_to_change"])
        ), "target_group should be different from the original"
        assert not torch.equal(
            tg2, torch.from_numpy(expected["container"]["target_group"]["field_to_change"])
        ), "container/target_group should be different from the original"

        # Ensure the two sub-trees were processed independently: their replacement values differ
        val1 = tg1.float().mean().item()
        val2 = tg2.float().mean().item()
        assert (
            abs(val1 - val2) > 0.1
        ), "The two target_group sub-trees should be different from each other (as they should be processed independently)"

        # Non-replaced fields inside the sub-trees remain intact
        assert torch.equal(
            res["target_group"]["inner_other"][0], torch.from_numpy(expected["target_group"]["inner_other"])
        ), "inner_other should remain unchanged"
        assert torch.equal(
            res["container"]["target_group"]["inner_other"][0],
            torch.from_numpy(expected["container"]["target_group"]["inner_other"]),
        ), "container/target_group/inner_other should remain unchanged"

    finally:
        restore_generator(orig_gen)


if __name__ == "__main__":
    pytest.main([__file__])
