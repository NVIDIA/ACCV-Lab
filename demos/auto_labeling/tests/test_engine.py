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

from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')
from demos.auto_labeling.inference_engine import InferenceEngine
from demos.auto_labeling.gpu_image_transforms import GPUImageTransforms
from demos.auto_labeling.batch_utils import update_metadata_for_transform


@pytest.fixture
def model_factory(monkeypatch, tmp_path):
    models = pytest.importorskip('mmdet3d.models')

    config = tmp_path / 'model.py'
    config.write_text(
        "model = dict(img_backbone=dict(pretrained='unused', init_cfg=dict(type='Pretrained')))"
    )
    built = []

    def build_model(cfg, **kwargs):
        assert cfg.pretrained is None
        assert cfg.img_backbone.pretrained is None
        assert cfg.img_backbone.init_cfg is None
        model = torch.nn.Linear(2, 2)
        model.init_weights = Mock(side_effect=model.reset_parameters)
        built.append(model)
        return model

    monkeypatch.setattr(models, 'build_model', build_model)
    return str(config), built


def test_random_initialization_is_repeatable_and_does_not_load_a_checkpoint(monkeypatch, model_factory):
    config, built = model_factory
    load = Mock(side_effect=AssertionError('Random initialization must not load a checkpoint'))
    monkeypatch.setattr(torch, 'load', load)
    rng = torch.get_rng_state().clone()
    first = InferenceEngine(config, device='cpu')
    second = InferenceEngine(config, device='cpu')
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name])
    assert not first.model.training
    for model in built:
        model.init_weights.assert_called_once_with()
    load.assert_not_called()


@pytest.mark.parametrize('checkpoint_kind', ['valid', 'missing', 'incompatible'])
def test_supplied_checkpoints_load_strictly_without_random_fallback(model_factory, tmp_path, checkpoint_kind):
    config, built = model_factory
    checkpoint = tmp_path / 'user_checkpoint.pth'
    expected = {'module.weight': torch.ones(2, 2), 'module.bias': torch.zeros(2)}
    if checkpoint_kind != 'missing':
        if checkpoint_kind == 'incompatible':
            expected['module.weight'] = torch.zeros(3, 3)
        torch.save({'state_dict': expected}, checkpoint)
    if checkpoint_kind == 'valid':
        engine = InferenceEngine(config, checkpoint, device='cpu')
        assert torch.equal(engine.model.weight, expected['module.weight'])
        assert torch.equal(engine.model.bias, expected['module.bias'])
    else:
        error = FileNotFoundError if checkpoint_kind == 'missing' else RuntimeError
        with pytest.raises(error):
            InferenceEngine(config, checkpoint, device='cpu')
    built[0].init_weights.assert_not_called()


class RecordingModel(torch.nn.Module):
    def __init__(self, boxes):
        super().__init__()
        self.reset_calls = 0
        self.boxes = boxes
        self.pts_bbox_head = Mock(side_effect=self.record_head_call)
        self.pts_bbox_head.reset_memory = self.reset_memory
        self.pts_bbox_head.get_bboxes = lambda outputs, metas: outputs
        self.calls = []

    def reset_memory(self):
        self.reset_calls += 1

    def extract_img_feat(self, img, len_queue):
        return img

    def prepare_location(self, img_metas, **batch):
        return None

    def forward_roi_head(self, location, **batch):
        return {'topk_indexes': None}

    def record_head_call(self, location, img_metas, topk, **batch):
        self.calls.append(batch)
        return [(self.boxes, torch.ones(1), torch.zeros(1, dtype=torch.long))] * len(img_metas)


def test_engine_preserves_independent_first_frame_flags(monkeypatch):
    bbox = pytest.importorskip('mmdet3d.core.bbox')

    model = RecordingModel(bbox.LiDARInstance3DBoxes(torch.zeros(1, 9), box_dim=9))
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

    pytest.importorskip('mmdet3d')
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
