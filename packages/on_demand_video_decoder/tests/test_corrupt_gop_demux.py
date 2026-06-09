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
from tempfile import TemporaryDirectory

import pytest


def test_corrupt_hevc_demux_raises_actionable_error():
    """Regression guard for corrupt HEVC mdat packets in GetGOPList."""
    video = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pix_fmt_variants"
        / "hevc_hvc1_yuv420p10le.mp4"
    )

    from accvlab.on_demand_video_decoder import CreateGopDecoder

    raw = bytearray(video.read_bytes())
    mdat_pos = raw.find(b"mdat")
    if mdat_pos < 0:
        raise RuntimeError("mdat box not found")

    start = mdat_pos + 16
    end = min(start + 512, len(raw))
    for idx in range(start, end):
        raw[idx] ^= 0xFF

    with TemporaryDirectory() as tmpdir:
        bad_video = Path(tmpdir) / "bad_hevc.mp4"
        bad_video.write_bytes(raw)
        decoder = CreateGopDecoder(maxfiles=8, iGpu=0)
        for frame_id in (0, 1, 5, 10, 20, 30):
            with pytest.raises(RuntimeError) as exc_info:
                decoder.GetGOPList([str(bad_video)], [frame_id], useGOPCache=False)
            output = str(exc_info.value)
            assert "General error -1094995529" not in output
            assert "FFmpeg bitstream filter receive failed" in output
            assert "GOP demux failed" in output
