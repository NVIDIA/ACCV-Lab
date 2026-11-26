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

from pathlib import Path

import pytest
import torch

import accvlab.on_demand_video_decoder as nvc


@pytest.mark.parametrize("device_idx", range(torch.cuda.device_count()))
def test_nv12_dlpack_device(device_idx):
    device = torch.device("cuda", device_idx)
    path = str(Path(__file__).resolve().parents[1] / "data/pix_fmt_variants/h264_avc1_yuv420p.mp4")
    frame_id = 33
    with torch.cuda.device(device):
        decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=device_idx)
        gop, _, _ = decoder.GetGOPList([path], [frame_id], useGOPCache=True)[0]
        (frame,) = decoder.DecodeFromGOPList([gop], [path], [frame_id])

        assert frame.__dlpack_device__() == (2, device_idx)  # kDLCUDA
        packed = torch.from_dlpack(frame)
        assert packed.device == device
        assert packed.data_ptr() == frame.GetPtrToPlane(0)
