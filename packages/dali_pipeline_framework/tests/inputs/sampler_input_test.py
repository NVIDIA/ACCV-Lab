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

from accvlab.dali_pipeline_framework.pipeline import DALIStructuredOutputIterator, PipelineDefinition

from accvlab.dali_pipeline_framework.inputs.sampler_base import SamplerBase
from accvlab.dali_pipeline_framework.inputs.sampler_input_callable import SamplerInputCallable
from accvlab.dali_pipeline_framework.inputs.sampler_input_iterable import SamplerInputIterable

from _sample_selection_test_helpers import (
    build_no_processing_pipeline_and_iterator,
    SimpleDataProvider,
)

try:
    from typing import override
except ImportError:
    from typing_extensions import override

# --------------------------------------------------------------------------------------------------
# Definitions of classes used in tests
# --------------------------------------------------------------------------------------------------


class _SequentialSampler(SamplerBase):
    """Non-epoch sampler that yields increasing indices in fixed-size batches."""

    def __init__(self, batch_size: int, start: int = 0):
        self._batch_size = batch_size
        self._next = start

    @property
    @override
    def is_epoch_based(self) -> bool:
        return False

    @override
    def reset(self):
        raise AssertionError("reset() should not be called for non-epoch sampler")

    @override
    def get_next_batch_indices(self):
        batch = [self._next + i for i in range(self._batch_size)]
        self._next += self._batch_size
        return batch

    @property
    @override
    def length(self):
        return None


class _EpochSampler(SamplerBase):
    """Epoch-based sampler that yields fixed number of batches per epoch and advances epoch on reset."""

    def __init__(self, batch_size: int, epoch_num_batches: int, epoch_stride: int = 1000):
        self._batch_size = batch_size
        self._epoch_num_batches = epoch_num_batches
        self._epoch_stride = epoch_stride
        self._curr_epoch = 0
        self._b_in_epoch = 0

    @property
    @override
    def is_epoch_based(self) -> bool:
        return True

    @override
    def reset(self):
        self._curr_epoch += 1
        self._b_in_epoch = 0

    @override
    def get_next_batch_indices(self):
        if self._b_in_epoch >= self._epoch_num_batches:
            raise StopIteration
        base = self._curr_epoch * self._epoch_stride + self._b_in_epoch * self._batch_size
        batch = [base + i for i in range(self._batch_size)]
        self._b_in_epoch += 1
        return batch

    @property
    @override
    def length(self):
        return self._epoch_num_batches


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("input_kind", ["callable", "iterable"])
def test_sampler_inputs_basic_flow_and_ids_match(input_kind: str):
    batch_size = 4
    num_test_batches = 8

    data_provider = SimpleDataProvider()
    sampler = _SequentialSampler(batch_size=batch_size, start=0)

    # Build input according to the kind
    if input_kind == "callable":
        input_obj = SamplerInputCallable(
            data_provider=data_provider,
            sampler=sampler,
            max_num_iterations=num_test_batches,
            pre_fetch_queue_length=2,
            shard_id=0,
            num_shards=1,
        )
    else:
        input_obj = SamplerInputIterable(
            data_provider=data_provider,
            sampler=sampler,
            shard_id=0,
            num_shards=1,
        )

    _, pipe, iterator = build_no_processing_pipeline_and_iterator(input_obj, batch_size, num_test_batches)

    try:
        it = iter(iterator)
        for b in range(num_test_batches):
            out = next(it)
            ids = out["id"].tolist()
            expected = [b * batch_size + i for i in range(batch_size)]
            assert ids == expected, f"Batch {b}: got {ids}, expected {expected}"
    finally:
        del iterator
        del pipe


@pytest.mark.parametrize("input_kind", ["iterable", "callable"])
def test_sampler_across_epoch_boundaries(input_kind: str):
    batch_size = 2
    epoch_batches = 3
    stride = 500

    provider = SimpleDataProvider()
    sampler = _EpochSampler(batch_size=batch_size, epoch_num_batches=epoch_batches, epoch_stride=stride)

    total_batches = epoch_batches * 2  # cross one boundary

    if input_kind == "iterable":
        input_obj = SamplerInputIterable(
            data_provider=provider,
            sampler=sampler,
            shard_id=0,
            num_shards=1,
        )
    else:
        input_obj = SamplerInputCallable(
            data_provider=provider,
            sampler=sampler,
            max_num_iterations=total_batches,
            pre_fetch_queue_length=2,
            shard_id=0,
            num_shards=1,
        )

    pipeline_def, pipe, it = build_no_processing_pipeline_and_iterator(input_obj, batch_size, total_batches)
    iter_it = iter(it)

    # Determine the actual starting epoch base from the first batch. Note that this could be > 0 as
    # in the example sampler, the epoch is counted up every time the reset() method is called.
    out0 = next(iter_it)
    ids0 = out0["id"].tolist()
    base0 = ids0[0]  # first id is the base
    expected0 = [base0 + i for i in range(batch_size)]
    assert ids0 == expected0, f"Batch 0: got {ids0}, expected {expected0}"

    # Remaining batches
    num_epoch_borders = 0
    for b in range(1, total_batches):
        try:
            out = next(iter_it)
        except StopIteration:
            num_epoch_borders += 1
            iter_it.reset()
            out = next(iter_it)
        ids = out["id"].tolist()
        if b < epoch_batches:
            expected = [base0 + b * batch_size + i for i in range(batch_size)]
        else:
            base1 = base0 + stride
            b_in_epoch = b - epoch_batches
            expected = [base1 + b_in_epoch * batch_size + i for i in range(batch_size)]
        assert ids == expected, f"Batch {b}: got {ids}, expected {expected}"
    assert num_epoch_borders == 1, f"Expected 1 epoch border, got {num_epoch_borders}"


@pytest.mark.parametrize("input_kind", ["callable", "iterable"])
def test_sampler_inputs_sharding_splits_total_batch(input_kind: str):
    total_batch_size = 4
    num_shards = 2
    local_batch_size = total_batch_size // num_shards
    num_test_batches = 6

    data_provider0 = SimpleDataProvider()
    data_provider1 = SimpleDataProvider()

    # Each shard uses its own sampler instance generating the same total batches
    sampler0 = _SequentialSampler(batch_size=total_batch_size, start=0)
    sampler1 = _SequentialSampler(batch_size=total_batch_size, start=0)

    if input_kind == "callable":
        input0 = SamplerInputCallable(
            data_provider=data_provider0,
            sampler=sampler0,
            max_num_iterations=num_test_batches,
            pre_fetch_queue_length=2,
            shard_id=0,
            num_shards=num_shards,
        )
        input1 = SamplerInputCallable(
            data_provider=data_provider1,
            sampler=sampler1,
            max_num_iterations=num_test_batches,
            pre_fetch_queue_length=2,
            shard_id=1,
            num_shards=num_shards,
        )
    else:
        input0 = SamplerInputIterable(
            data_provider=data_provider0,
            sampler=sampler0,
            shard_id=0,
            num_shards=num_shards,
        )
        input1 = SamplerInputIterable(
            data_provider=data_provider1,
            sampler=sampler1,
            shard_id=1,
            num_shards=num_shards,
        )

    # Build two pipelines (one per shard)
    pd0 = PipelineDefinition(
        data_loading_callable_iterable=input0,
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
    )
    pd1 = PipelineDefinition(
        data_loading_callable_iterable=input1,
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
    )

    pipe0 = pd0.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=local_batch_size,  # local batch per shard
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )
    pipe1 = pd1.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=local_batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    it0 = DALIStructuredOutputIterator(num_test_batches, pipe0, pd0.check_and_get_output_data_structure())
    it1 = DALIStructuredOutputIterator(num_test_batches, pipe1, pd1.check_and_get_output_data_structure())

    try:
        iter0 = iter(it0)
        iter1 = iter(it1)
        for b in range(num_test_batches):
            out0 = next(iter0)
            out1 = next(iter1)

            ids0 = out0["id"].tolist()
            ids1 = out1["id"].tolist()

            # Expect disjoint partition of the total batch
            expected_total = [b * total_batch_size + i for i in range(total_batch_size)]
            expected0 = expected_total[:local_batch_size]
            expected1 = expected_total[local_batch_size:]

            assert ids0 == expected0, f"Shard 0 batch {b}: got {ids0}, expected {expected0}"
            assert ids1 == expected1, f"Shard 1 batch {b}: got {ids1}, expected {expected1}"

            # Union equals total, intersection empty
            assert set(ids0).isdisjoint(ids1)
            assert sorted(ids0 + ids1) == expected_total
    finally:
        del it0
        del it1
        del pipe0
        del pipe1


@pytest.mark.parametrize("input_kind", ["callable", "iterable"])
def test_sampler_inputs_sharding_across_epoch_boundaries(input_kind: str):
    total_batch_size = 6
    num_shards = 2
    local_batch_size = total_batch_size // num_shards
    epoch_batches = 3
    stride = 100

    provider0 = SimpleDataProvider()
    provider1 = SimpleDataProvider()

    sampler0 = _EpochSampler(
        batch_size=total_batch_size, epoch_num_batches=epoch_batches, epoch_stride=stride
    )
    sampler1 = _EpochSampler(
        batch_size=total_batch_size, epoch_num_batches=epoch_batches, epoch_stride=stride
    )

    if input_kind == "callable":
        input0 = SamplerInputCallable(
            data_provider=provider0,
            sampler=sampler0,
            max_num_iterations=epoch_batches * 2,
            pre_fetch_queue_length=2,
            shard_id=0,
            num_shards=num_shards,
        )
        input1 = SamplerInputCallable(
            data_provider=provider1,
            sampler=sampler1,
            max_num_iterations=epoch_batches * 2,
            pre_fetch_queue_length=2,
            shard_id=1,
            num_shards=num_shards,
        )
    else:
        input0 = SamplerInputIterable(
            data_provider=provider0,
            sampler=sampler0,
            shard_id=0,
            num_shards=num_shards,
        )
        input1 = SamplerInputIterable(
            data_provider=provider1,
            sampler=sampler1,
            shard_id=1,
            num_shards=num_shards,
        )

    pd0 = PipelineDefinition(
        data_loading_callable_iterable=input0,
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
    )
    pd1 = PipelineDefinition(
        data_loading_callable_iterable=input1,
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
    )

    pipe0 = pd0.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=local_batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )
    pipe1 = pd1.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=local_batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    it0 = DALIStructuredOutputIterator(epoch_batches, pipe0, pd0.check_and_get_output_data_structure())
    it1 = DALIStructuredOutputIterator(epoch_batches, pipe1, pd1.check_and_get_output_data_structure())

    try:
        iter0 = iter(it0)
        iter1 = iter(it1)

        # Epoch 0 (determine actual base from first outputs to accommodate iterable init resets)
        base0 = None
        for b in range(epoch_batches):
            out0 = next(iter0)
            out1 = next(iter1)
            ids0 = out0["id"].tolist()
            ids1 = out1["id"].tolist()
            if b == 0:
                base0 = min(ids0[0], ids1[0])
            expected_total = [base0 + b * total_batch_size + i for i in range(total_batch_size)]
            assert ids0 == expected_total[:local_batch_size]
            assert ids1 == expected_total[local_batch_size:]

        # Next call should raise StopIteration simultaneously for both shards
        with pytest.raises(StopIteration):
            _ = next(iter0)
        with pytest.raises(StopIteration):
            _ = next(iter1)

        # Reset both and verify epoch 1 partitioning as well
        iter0.reset()
        iter1.reset()

        # First batch of epoch 1: determine base1 from outputs
        out0 = next(iter0)
        out1 = next(iter1)
        ids0 = out0["id"].tolist()
        ids1 = out1["id"].tolist()
        base1 = min(ids0[0], ids1[0])
        assert base1 == base0 + stride, "Base should be advanced by stride"
        expected_total = [base1 + 0 * total_batch_size + i for i in range(total_batch_size)]
        assert ids0 == expected_total[:local_batch_size]
        assert ids1 == expected_total[local_batch_size:]

        # Remaining batches of epoch 1
        for b in range(1, epoch_batches):
            out0 = next(iter0)
            out1 = next(iter1)
            ids0 = out0["id"].tolist()
            ids1 = out1["id"].tolist()
            expected_total = [base1 + b * total_batch_size + i for i in range(total_batch_size)]
            assert ids0 == expected_total[:local_batch_size]
            assert ids1 == expected_total[local_batch_size:]
    finally:
        del it0
        del it1
        del pipe0
        del pipe1


if __name__ == "__main__":
    # test_sampler_input_iterable_continuous_across_epoch_boundaries()
    pytest.main([__file__])
