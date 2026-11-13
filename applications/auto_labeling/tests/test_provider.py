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

"""Decoder failures and invalid metadata must fail before producing labels."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')
from applications.auto_labeling.data_provider import VideoDataProvider


def test_incomplete_camera_metadata_fails_before_decode():
    provider = object.__new__(VideoDataProvider)
    provider.batch_size = 1
    provider.num_cameras = 6
    provider._pending = None
    provider._decoder = None
    provider.video_base_path = Path('/videos')
    provider.data_index = SimpleNamespace(
        get_data_info=lambda idx: dict(
            sample_idx='sample',
            img_filename=['image'] * 6,
            video_paths=['video'] * 5,
            video_frame_ids=[0] * 5,
        )
    )
    with pytest.raises(ValueError, match='all 6 cameras'):
        provider.get_batch([0])
    assert provider._decoder is None


def test_failed_prefetch_get_is_not_retried_during_close():
    class Decoder:
        calls = 0

        def DecodeN12ToRGBAsyncGetBuffer(self, *args):
            self.calls += 1
            raise RuntimeError('decode failed')

    decoder = Decoder()
    provider = object.__new__(VideoDataProvider)
    provider.batch_size = 1
    provider._decoder = decoder
    provider._pending = ((0,), [{}], ['video'], [0])
    with pytest.raises(RuntimeError, match='decode failed'):
        provider.get_batch([0])
    provider.close()
    assert decoder.calls == 1
    assert provider._decoder is None
