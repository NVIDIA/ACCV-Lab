# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
import sys

import pytest
import torch


pytest.importorskip("mmdet")
bhl = pytest.importorskip("accvlab.batching_helpers")

STREAM_PETR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM_PETR_ROOT))

from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_2d_batched import (  # noqa: E402
    HungarianAssigner2DBatched,
)
from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d_batched import (  # noqa: E402
    HungarianAssigner3DBatched,
)


def _ragged_batch(batch_shape, sample_size, data_shape, dtype):
    data = torch.zeros((*batch_shape, sample_size, *data_shape), dtype=dtype)
    sample_sizes = torch.full(batch_shape, sample_size, dtype=torch.int64)
    return bhl.RaggedBatch(data, sample_sizes=sample_sizes)


def _assert_empty_matches(matches, batch_shape):
    pred_indices, gt_indices = matches
    for indices in (pred_indices, gt_indices):
        assert indices.batch_shape == batch_shape
        assert indices.tensor.shape == (*batch_shape, 0)
        assert indices.tensor.dtype == torch.int64
        assert indices.mask.shape == (*batch_shape, 0)
        assert indices.sample_sizes.shape == batch_shape
        assert torch.count_nonzero(indices.sample_sizes) == 0


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
@pytest.mark.parametrize("empty_input", ["ground_truth", "predictions"])
def test_2d_assigner_preserves_empty_batch_shape(batch_shape, empty_input):
    num_gts = 0 if empty_input == "ground_truth" else 2
    num_preds = 0 if empty_input == "predictions" else 4
    assigner = HungarianAssigner2DBatched.__new__(HungarianAssigner2DBatched)

    matches = assigner.assign(
        bbox_pred=torch.zeros((*batch_shape, num_preds, 4)),
        cls_pred=torch.zeros((*batch_shape, num_preds, 10)),
        pred_centers2d=torch.zeros((*batch_shape, num_preds, 2)),
        gt_bboxes=_ragged_batch(batch_shape, num_gts, (4,), torch.float32),
        gt_labels=_ragged_batch(batch_shape, num_gts, (), torch.int64),
        centers2d=_ragged_batch(batch_shape, num_gts, (2,), torch.float32),
        img_meta=None,
    )

    _assert_empty_matches(matches, batch_shape)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
@pytest.mark.parametrize("empty_input", ["ground_truth", "predictions"])
def test_3d_assigner_preserves_empty_batch_shape(batch_shape, empty_input):
    num_gts = 0 if empty_input == "ground_truth" else 2
    num_preds = 0 if empty_input == "predictions" else 4
    assigner = HungarianAssigner3DBatched.__new__(HungarianAssigner3DBatched)

    matches = assigner.assign(
        bbox_pred=torch.zeros((*batch_shape, num_preds, 10)),
        cls_pred=torch.zeros((*batch_shape, num_preds, 10)),
        gt_bboxes=_ragged_batch(batch_shape, num_gts, (9,), torch.float32),
        gt_labels=_ragged_batch(batch_shape, num_gts, (), torch.int64),
    )

    _assert_empty_matches(matches, batch_shape)
