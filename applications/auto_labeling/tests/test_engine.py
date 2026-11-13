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

"""Exercise the model boundary without a checkpoint; demo tests use its environment."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from applications.auto_labeling.inference_engine import InferenceEngine
from applications.auto_labeling.gpu_image_transforms import GPUImageTransforms
from applications.auto_labeling.batch_utils import update_metadata_for_transform


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0
        self.pts_bbox_head = SimpleNamespace(reset_memory=self.reset_memory)
        self.calls = []

    def reset_memory(self):
        self.reset_calls += 1

    def simple_test(self, **batch):
        self.calls.append(batch)
        return [
            dict(
                pts_bbox=dict(
                    boxes_3d=SimpleNamespace(tensor=torch.zeros(1, 9)),
                    scores_3d=torch.ones(1),
                    labels_3d=torch.zeros(1, dtype=torch.long),
                )
            )
        ] * 2


def test_engine_preserves_independent_first_frame_flags(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(InferenceEngine, '_build_model', lambda self, *paths: model)
    engine = InferenceEngine('unused', 'unused', device='cpu', batch_size=2)
    batch = dict(img=torch.zeros(2), img_metas=[{'scene_token': 'a'}, {'scene_token': 'b'}])
    results = engine.infer_multi_video(batch, [True, False])
    torch.testing.assert_close(model.calls[-1]['prev_exists'], torch.tensor([0.0, 1.0]))
    assert InferenceEngine.parse_results(results)[0]['boxes_3d'].shape == (1, 9)
    engine.reset_multi_video_state()
    assert model.reset_calls == 1
    with pytest.raises(ValueError):
        engine.infer_multi_video(batch, [True])


def test_transform_and_calibration_match_one_resize():
    import numpy as np

    norm = dict(mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 3.0], to_rgb=True)
    aug = dict(H=4, W=8, final_dim=(2, 4), bot_pct_lim=(0.0, 0.0))
    transform = GPUImageTransforms(norm, aug, device='cpu')
    image = torch.tensor([10.0, 20.0, 30.0]).expand(4, 8, 3)
    result = transform([image])
    assert result.shape == (1, 3, 32, 32)
    torch.testing.assert_close(result[0, :, 0, 0], torch.tensor([29.0, 9.0, 7.0 / 3.0]))
    metadata = dict(intrinsics=[np.eye(4)], extrinsics=[np.eye(4)])
    updated = update_metadata_for_transform(
        metadata, transform.resize, transform.crop_h, transform.crop_w, transform.final_dim
    )
    assert updated['intrinsics'][0][0, 0] == pytest.approx(0.5)
    assert updated['intrinsics'][0][1, 1] == pytest.approx(0.5)


def test_demo_respects_external_temporal_flags_and_keeps_legacy_scene_reset():
    pytest.importorskip('mmdet3d')
    demo = str(Path(__file__).resolve().parents[3] / 'demos/streampetr')
    if demo not in sys.path:
        sys.path.insert(0, demo)
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    class Head:
        resets = 0

        def reset_memory(self):
            self.resets += 1

        def __call__(self, location, metas, topk, **data):
            self.flags = data['prev_exists']
            return None

        def get_bboxes(self, outs, metas):
            return []

    head = Head()
    model = SimpleNamespace(
        prepare_location=lambda *args, **kw: None,
        forward_roi_head=lambda *args, **kw: {'topk_indexes': None},
        pts_bbox_head=head,
        prev_scene_token=None,
    )
    flags = torch.tensor([0.0, 1.0])
    Petr3D.simple_test_pts(
        model, [{'scene_token': 'a'}, {'scene_token': 'b'}], img=torch.zeros(2), prev_exists=flags
    )
    assert head.flags is flags
    assert head.resets == 0
    Petr3D.simple_test_pts(model, [{'scene_token': 'a'}], img=torch.zeros(1))
    assert head.resets == 1
    torch.testing.assert_close(head.flags, torch.zeros(1))
    Petr3D.simple_test_pts(model, [{'scene_token': 'a'}], img=torch.zeros(1))
    assert head.resets == 1
    torch.testing.assert_close(head.flags, torch.ones(1))
