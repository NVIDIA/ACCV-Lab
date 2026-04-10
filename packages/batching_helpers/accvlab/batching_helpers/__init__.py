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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("accvlab.batching_helpers")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import the relevant data type & functions directly into the top-level package
from .data_format import RaggedBatch
from .batched_indexing_ops import (
    batched_indexing_access,
    batched_inverse_indexing_access,
    batched_indexing_write,
)
from .batched_index_mapping_op import batched_index_mapping
from .batched_mask_from_indices import get_mask_from_indices
from .batched_bool_indexing import batched_bool_indexing, batched_bool_indexing_write
from .batched_processing_py import (
    average_over_targets,
    sum_over_targets,
    apply_mask_to_tensor,
    squeeze_except_batch_and_sample,
    get_compact_from_named_tuple,
    get_compact_lists,
    combine_data,
    get_indices_from_mask,
)

# The order of the elements as defined here will be maintained by the Sphinx-generated documentation
__all__ = [
    "__version__",
    'RaggedBatch',
    *sorted(
        [
            'batched_indexing_access',
            'batched_inverse_indexing_access',
            "batched_indexing_write",
            'batched_index_mapping',
            'batched_bool_indexing',
            'batched_bool_indexing_write',
            'average_over_targets',
            'sum_over_targets',
            'apply_mask_to_tensor',
            'squeeze_except_batch_and_sample',
            'get_mask_from_indices',
            'get_indices_from_mask',
            'get_compact_from_named_tuple',
            'get_compact_lists',
            'combine_data',
        ]
    ),
]
