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
import sys

import random

import utils
import accvlab.on_demand_video_decoder as nvc


def test_random_access_fast_init_single():
    max_num_files_to_use = 6
    iter_num = 10
    path_base = utils.get_data_dir()

    nv_gop_dec = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )

    frame_min = 0
    frame_max = 200

    files = utils.select_random_clip(path_base)
    assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"
    fast_stream_infos = nvc.GetFastInitInfo(files)

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        gop_decoded = utils.gop_decode_bgr_with_fast_init(nv_gop_dec, files, frames, fast_stream_infos)
        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


def test_separate_access_fast_init_single():
    max_num_files_to_use = 6
    iter_num = 10
    path_base = utils.get_data_dir()

    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )

    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )

    frame_min = 0
    frame_max = 200

    files = utils.select_random_clip(path_base)
    assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"
    fast_stream_infos = nvc.GetFastInitInfo(files)

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        gop_decoded = utils.gop_decode_bgr_ddseparate_with_fast_init(
            nv_gop_dec1, nv_gop_dec2, files, frames, fast_stream_infos
        )
        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
