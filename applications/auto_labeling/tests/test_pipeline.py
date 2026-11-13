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

"""Public output, temporal sampling and failure-handling regressions."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from applications.auto_labeling.data_index import DataIndex
from applications.auto_labeling.inference import run_multi_video_inference, save_results_json
from applications.auto_labeling.multi_video_sampler import MultiVideoSampler


def test_cli_help_needs_no_model_dependencies():
    root = Path(__file__).resolve().parents[3]
    for command in [
        [sys.executable, '-m', 'applications.auto_labeling.inference', '--help'],
        [sys.executable, str(root / 'applications/auto_labeling/inference.py'), '--help'],
    ]:
        result = subprocess.run(
            command, cwd='/tmp', env={'PYTHONPATH': str(root)}, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert '--checkpoint' in result.stdout
        assert '--data-mode' not in result.stdout


@pytest.mark.parametrize('batch_size', [0, -1])
def test_sampler_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        MultiVideoSampler({'scene': [0]}, batch_size)


def test_uneven_scenes_hold_the_last_frame_without_repeating_labels():
    sampler = MultiVideoSampler({'short': [0], 'long': [1, 2, 3]}, 2)
    assert sampler.get_batch_indices() == ([0, 1], [True, True])
    assert sampler.peek_next_indices() == [0, 2]
    sampler.advance_frame()
    assert sampler.get_batch_indices() == ([0, 2], [False, False])
    sampler.advance_frame()
    assert sampler.get_batch_indices() == ([0, 3], [False, False])
    assert sampler.peek_next_indices() is None
    assert not sampler.advance_frame()


def test_partial_round_marks_padding():
    sampler = MultiVideoSampler({'a': [0], 'b': [1], 'c': [2]}, 2)
    sampler.set_round(1)
    assert sampler.active_mask == [True, False]
    assert sampler.get_batch_indices() == ([2, 2], [True, True])
    with pytest.raises(ValueError):
        sampler.set_round(2)


@pytest.mark.parametrize('limit, expected', [(None, [0, 1, 2, 3, 4]), (3, [0, 1, 2]), (1, [0])])
def test_inference_emits_each_real_sample_once_and_honors_limit(limit, expected):
    class Provider:
        def __len__(self):
            return 5

        def create_multi_video_sampler(self, batch_size):
            self.sampler = MultiVideoSampler({'a': [0], 'b': [1, 2, 3], 'c': [4]}, batch_size)
            return self.sampler

        def get_multi_video_batch(self):
            indices, first = self.sampler.get_batch_indices()
            return {'img_metas': [{'sample_idx': i} for i in indices]}, first

        def prefetch_next_batch(self):
            pass

        def advance_multi_video_frame(self):
            return self.sampler.advance_frame()

    class Engine:
        def reset_multi_video_state(self):
            pass

        def infer_multi_video(self, batch, first):
            return batch['img_metas']

        @staticmethod
        def parse_results(results):
            return results

    batches = run_multi_video_inference(
        SimpleNamespace(batch_size=2, num_samples=limit), Provider(), Engine()
    )
    assert [meta['sample_idx'] for _, metadata in batches for meta in metadata] == expected


def results_batch():
    result = dict(
        boxes_3d=np.array([[1, 2, 3, 4, 5, 6, 0.2, 7, 8]] * 2),
        scores_3d=np.array([0.8, 0.1]),
        labels_3d=np.array([0, 1]),
    )
    return [result], [dict(sample_idx='sample', scene_token='scene', filename=[])]


def test_saved_detections_use_config_classes_and_keep_geometry(tmp_path):
    dest = tmp_path / 'detections.json'
    assert save_results_json(iter([results_batch()]), dest, ['configured_car', 'truck'], 0.3) == 1
    row = json.loads(dest.read_text())[0]
    assert row['num_detections'] == 1
    assert row['detections'][0]['translation'] == [1, 2, 3]
    assert row['detections'][0]['velocity'] == [7, 8]
    assert row['detections'][0]['label_name'] == 'configured_car'


def test_stream_failure_keeps_the_previous_complete_file(tmp_path):
    dest = tmp_path / 'detections.json'
    dest.write_text('previous result')

    def batches():
        yield results_batch()
        raise RuntimeError('decode failed')

    with pytest.raises(RuntimeError, match='decode failed'):
        save_results_json(batches(), dest, ['car', 'truck'])
    assert dest.read_text() == 'previous result'
    assert list(tmp_path.iterdir()) == [dest]


@pytest.mark.parametrize('bad_value', ['label', 'geometry'])
def test_invalid_detections_do_not_publish_invalid_json(tmp_path, bad_value):
    batch = results_batch()
    if bad_value == 'label':
        batch[0][0]['labels_3d'][0] = -1
    else:
        batch[0][0]['boxes_3d'][0, 0] = np.inf
    with pytest.raises(ValueError):
        save_results_json(iter([batch]), tmp_path / 'out.json', ['car', 'truck'])
    assert list(tmp_path.iterdir()) == []


def test_missing_frame_indices_fall_back_to_timestamp_order(tmp_path):
    path = tmp_path / 'index.pkl'
    path.write_bytes(pickle.dumps(dict(infos=[dict(scene_token='a', timestamp=t) for t in [30, 10, 20]])))
    index = DataIndex(path, tmp_path)
    assert index.video_indices['a'] == [1, 2, 0]
