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
from types import SimpleNamespace
import sys

import pytest
import torch


pytest.importorskip("mmdet")
pytest.importorskip("accvlab.batching_helpers")

STREAM_PETR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM_PETR_ROOT))

from projects.mmdet3d_plugin.models.dense_heads.streampetr_head_batched import (  # noqa: E402
    StreamPETRHeadBatched,
)


def test_empty_denoising_fallback_reuses_detached_losses():
    num_dec_layers = 3
    batch_size = 2
    num_queries = 4

    head = StreamPETRHeadBatched.__new__(StreamPETRHeadBatched)
    torch.nn.Module.__init__(head)
    head.with_dn = True

    losses_cls = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    losses_bbox = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    head.loss_batched = lambda *_args: (losses_cls, losses_bbox)

    empty_bboxes = [
        SimpleNamespace(gravity_center=torch.empty((0, 3)), tensor=torch.empty((0, 9)))
        for _ in range(batch_size)
    ]
    empty_labels = [torch.empty((0,), dtype=torch.int64) for _ in range(batch_size)]
    predictions = {
        "all_cls_scores": torch.empty((num_dec_layers, batch_size, num_queries, 10)),
        "all_bbox_preds": torch.empty((num_dec_layers, batch_size, num_queries, 10)),
        "dn_mask_dict": None,
    }

    result = head.loss(empty_bboxes, empty_labels, predictions)

    expected_keys = {
        "loss_cls",
        "loss_bbox",
        "d0.loss_cls",
        "d0.loss_bbox",
        "d1.loss_cls",
        "d1.loss_bbox",
        "dn_loss_cls",
        "dn_loss_bbox",
        "d0.dn_loss_cls",
        "d0.dn_loss_bbox",
        "d1.dn_loss_cls",
        "d1.dn_loss_bbox",
    }
    assert result.keys() == expected_keys
    assert result["dn_loss_cls"] == losses_cls[-1]
    assert result["dn_loss_bbox"] == losses_bbox[-1]
    assert result["d0.dn_loss_cls"] == losses_cls[0]
    assert result["d0.dn_loss_bbox"] == losses_bbox[0]
    assert result["d1.dn_loss_cls"] == losses_cls[1]
    assert result["d1.dn_loss_bbox"] == losses_bbox[1]
    assert all(not value.requires_grad for key, value in result.items() if "dn_loss" in key)
