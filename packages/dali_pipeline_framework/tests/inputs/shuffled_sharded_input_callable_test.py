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

from accvlab.dali_pipeline_framework.inputs.sfuffled_sharded_input_callable import (
    ShuffledShardedInputCallable,
)

from _sample_selection_test_helpers import (
    build_no_processing_pipeline_and_iterator,
    get_next_batch_ids,
    SimpleDataProvider,
)


@pytest.mark.parametrize("shuffle", [True, False])
def test_single_epoch_no_duplicates_and_expected_count(shuffle: bool):
    provider = SimpleDataProvider()
    batch_size = 8
    num_shards = 1
    shard_id = 0

    # Define dataset and epoch sizes
    num_samples = provider.get_number_of_samples()
    shard_size = num_samples // num_shards
    full_iterations = shard_size // batch_size  # number of full batches used

    callable_obj = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=batch_size,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle=shuffle,
        seed=123,
    )

    _, pipe, iterator = build_no_processing_pipeline_and_iterator(callable_obj, batch_size, full_iterations)
    try:
        it = iter(iterator)
        ids = []
        for _ in range(full_iterations):
            ids.extend(get_next_batch_ids(it))

        # Correct total unique count (only full batches, discard potential duplicates)
        assert len(set(ids)) == full_iterations * batch_size

        # No duplicates in a single epoch
        assert len(set(ids)) == len(ids)

        # If shuffle=True, order should not be the original sequential order
        if shuffle:
            assert ids != list(range(len(ids)))
        else:
            # When not shuffled, shard is sequential starting at 0
            assert ids == list(range(len(ids)))
    finally:
        del iterator
        del pipe


def test_two_shards_no_overlap_and_complete_partition():
    provider0 = SimpleDataProvider()
    provider1 = SimpleDataProvider()
    batch_size = 4
    num_shards = 2
    shard_ids = [0, 1]

    num_samples = provider0.get_number_of_samples()
    shard_size = num_samples // num_shards
    full_iterations = shard_size // batch_size

    call0 = ShuffledShardedInputCallable(
        data_provider=provider0,
        batch_size=batch_size,
        shard_id=shard_ids[0],
        num_shards=num_shards,
        shuffle=True,
        seed=999,
    )
    call1 = ShuffledShardedInputCallable(
        data_provider=provider1,
        batch_size=batch_size,
        shard_id=shard_ids[1],
        num_shards=num_shards,
        shuffle=True,
        seed=999,
    )

    _, pipe0, it0 = build_no_processing_pipeline_and_iterator(call0, batch_size, full_iterations)
    _, pipe1, it1 = build_no_processing_pipeline_and_iterator(call1, batch_size, full_iterations)

    try:
        it0_iter = iter(it0)
        it1_iter = iter(it1)
        acc0 = []
        acc1 = []
        for _ in range(full_iterations):
            acc0.extend(get_next_batch_ids(it0_iter))
            acc1.extend(get_next_batch_ids(it1_iter))
        ids0 = set(acc0)
        ids1 = set(acc1)

        # Disjoint shards
        assert ids0.isdisjoint(ids1)

        # Combined should cover exactly shard_size unique samples
        combined = ids0 | ids1
        assert len(combined) == 2 * full_iterations * batch_size
    finally:
        del it0
        del it1
        del pipe0
        del pipe1


def test_two_epochs_shuffle_changes_order_and_each_epoch_valid():
    provider = SimpleDataProvider()
    batch_size = 16
    num_shards = 1
    shard_id = 0

    num_samples = provider.get_number_of_samples()
    shard_size = num_samples // num_shards
    full_iterations = shard_size // batch_size

    call = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=batch_size,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle=True,
        seed=2025,
    )

    # First epoch
    _, pipe, it = build_no_processing_pipeline_and_iterator(call, batch_size, full_iterations)
    it_iter = iter(it)
    ids_epoch0 = []
    for _ in range(full_iterations):
        ids_epoch0.extend(get_next_batch_ids(it_iter))

    with pytest.raises(StopIteration):
        get_next_batch_ids(it_iter)

    # Reset iterator (new epoch)
    it_iter.reset()
    ids_epoch1 = []
    for _ in range(full_iterations):
        ids_epoch1.extend(get_next_batch_ids(it_iter))

    try:
        # Each epoch internally: no duplicates, no overlaps within epoch, correct counts
        assert len(set(ids_epoch0)) == full_iterations * batch_size
        assert len(set(ids_epoch1)) == full_iterations * batch_size
        assert len(set(ids_epoch0)) == len(ids_epoch0)
        assert len(set(ids_epoch1)) == len(ids_epoch1)

        # Across epochs: order should differ (very high probability with shuffle+different epoch_idx)
        assert ids_epoch0 != ids_epoch1
    finally:
        del it
        del pipe


if __name__ == "__main__":
    pytest.main([__file__])
