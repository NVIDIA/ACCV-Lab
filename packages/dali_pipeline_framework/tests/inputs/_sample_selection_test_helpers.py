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

from nvidia.dali import types as dali_types

from accvlab.dali_pipeline_framework.pipeline import (
    PipelineDefinition,
    DALIStructuredOutputIterator,
    SampleDataGroup,
)
from accvlab.dali_pipeline_framework.inputs import DataProvider

try:
    from typing import override
except ImportError:
    from typing_extensions import override


def build_no_processing_pipeline_and_iterator(input_obj, batch_size: int, num_batches: int):
    """Build a DALI pipeline that forwards input samples without processing steps."""

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_obj,
        preprocess_functors=[],
        copy_external_source_passthrough_outputs=True,
    )
    pipe = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=batch_size,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )
    iterator = DALIStructuredOutputIterator(
        num_batches, pipe, pipeline_def.check_and_get_output_data_structure()
    )
    return pipeline_def, pipe, iterator


def get_next_batch_ids(iterator_iter):
    """Advance a structured-output iterator and return the current batch IDs."""

    out = next(iterator_iter)
    return out["id"].tolist()


class SimpleDataProvider(DataProvider):
    """Data provider used by input tests to produce deterministic sample IDs."""

    def __init__(self):
        # Blueprint with a single INT32 field 'id' and some additional (constant) fields
        self._blueprint = SampleDataGroup()
        self._blueprint.add_data_field("id", dali_types.DALIDataType.INT32)
        additional = SampleDataGroup()
        additional.add_data_field("some_data", dali_types.DALIDataType.INT32)
        additional.add_data_field("some_other_data", dali_types.DALIDataType.FLOAT)
        self._blueprint.add_data_group_field("additional", additional)

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        """Return one sample with ``id`` equal to ``sample_id``."""

        s = self.sample_data_structure
        s["id"] = int(sample_id)
        s["additional"]["some_data"] = [1, 2, 3]
        s["additional"]["some_other_data"] = [4.0, 5.0, 6.0]
        return s

    @override
    def get_number_of_samples(self) -> int:
        """Return a large fixed sample count for sharding and epoch tests."""

        return 10000

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        """Return an empty sample structure matching this provider's output."""

        # Return an empty-like blueprint each time to avoid sharing state
        return self._blueprint.get_empty_like_self()
