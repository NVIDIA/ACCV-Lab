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

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)


class _StringProvider(DataProvider):
    """Provides simple string data to pass through the pipeline."""

    def __init__(self):
        self._samples = [
            {"token": "hello world", "meta": "a b c"},
            {"token": "foo bar", "meta": "x y"},
        ]

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        curr = self._samples[sample_id % len(self._samples)]
        # Assign string arrays directly; implementation converts to uint8 arrays inside pipeline
        res["token"] = curr["token"]
        res["meta"] = curr["meta"]
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("token", DALIDataType.STRING)
        res.add_data_field("meta", DALIDataType.STRING)
        return res


def test_strings_roundtrip_through_pipeline():

    batch_size = 2
    provider = _StringProvider()

    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=batch_size,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[],  # No processing; pass-through
        copy_external_source_passthrough_outputs=True,
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(
        batch_size, pipeline, pipeline_def.check_and_get_output_data_structure()
    )
    res = next(iter(iterator))

    # Reference from provider
    expected = [provider.get_data(0), provider.get_data(1)]

    # Verify strings match what was provided originally
    for i in range(batch_size):
        assert res["token"][i] == expected[i]["token"], "token strings should roundtrip unchanged"
        assert res["meta"][i] == expected[i]["meta"], "meta strings should roundtrip unchanged"


if __name__ == "__main__":
    pytest.main([__file__])
