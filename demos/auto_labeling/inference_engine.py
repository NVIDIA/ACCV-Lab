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

"""Load the existing StreamPETR demo and run per-slot temporal inference."""

from collections import OrderedDict
import importlib

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from . import add_streampetr_import_path


class InferenceEngine:
    def __init__(self, config_path, checkpoint_path=None, device='cuda:0', batch_size=1):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = self._build_model(config_path, checkpoint_path)
        self.model.eval()

    def _build_model(self, config_path, checkpoint_path):
        from mmcv import Config
        from mmdet3d.models import build_model

        add_streampetr_import_path()
        importlib.import_module('projects.mmdet3d_plugin')
        cfg = Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.model.img_backbone.pretrained = None
        cfg.model.img_backbone.init_cfg = None
        cfg.model.train_cfg = None
        if checkpoint_path is None:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
                model.init_weights()
            print('Using random initialization (seed=0) for performance testing; predictions are untrained.')
        else:
            model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
            # User-supplied checkpoints may contain metadata as well as tensors.
            # Only load trusted files, as documented in the demo README.
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('state_dict', checkpoint)
            if not isinstance(state, OrderedDict):
                state = OrderedDict(state)
            consume_prefix_in_state_dict_if_present(state, 'module.')
            # Preserve canonical FocalHead names in plain state dictionaries.
            if not hasattr(state, '_metadata'):
                state._metadata = {}
            state._metadata.setdefault('img_roi_head', {}).setdefault('version', 1)
            model.load_state_dict(state, strict=True)
        return model.to(self.device)

    def infer_multi_video(self, batch_data, is_first_frame):
        if len(batch_data['img_metas']) != self.batch_size or len(is_first_frame) != self.batch_size:
            raise ValueError('Batch size and temporal flags must match the configured batch size')
        from mmdet3d.core import bbox3d2result

        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch_data.items()
        }
        img_metas = batch.pop('img_metas')
        batch['prev_exists'] = torch.tensor(
            [not first for first in is_first_frame], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            # Petr3D.simple_test uses one temporal flag for the entire input.
            # Run its existing stages with the sampler's flag for each slot.
            batch['img_feats'] = self.model.extract_img_feat(batch['img'], 1)
            location = self.model.prepare_location(img_metas, **batch)
            topk = self.model.forward_roi_head(location, **batch)['topk_indexes']
            head = self.model.pts_bbox_head
            outputs = head(location, img_metas, topk, **batch)
            return [
                {'pts_bbox': bbox3d2result(boxes, scores, labels)}
                for boxes, scores, labels in head.get_bboxes(outputs, img_metas)
            ]

    def reset_multi_video_state(self):
        self.model.pts_bbox_head.reset_memory()

    @staticmethod
    def parse_results(results):
        """Transfer LiDAR boxes, scores and labels to CPU for JSON serialization."""
        parsed = []
        for result in results:
            boxes = result['pts_bbox']
            parsed.append(
                dict(
                    boxes_3d=boxes['boxes_3d'].tensor.cpu().numpy(),
                    scores_3d=boxes['scores_3d'].cpu().numpy(),
                    labels_3d=boxes['labels_3d'].cpu().numpy(),
                )
            )
        return parsed
