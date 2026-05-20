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

"""
Pix_fmt detection coverage across the codec_tag / bit-depth matrix.

Regression target: a previously-shipped patent-free FFmpeg build cannot populate
`AVCodecParameters::format` for HEVC / H.264 streams (no decoder is linked in to
probe the SPS), and the demuxer silently fell back to 8-bit yuv420p. For a real
10-bit HEVC stream this mis-sized the GPU output buffer by a factor of 2 and
crashed `DecodeFromGOP` with `CUDA_ERROR_INVALID_VALUE`.

These tests drive `GetGOPList` + `DecodeFromGOP` over the matrix of stream
shapes the package supports on NVDEC (HEVC at both 8-bit and 10-bit with both
hev1 and hvc1 sample-entry tags, plus H.264 8-bit), and assert the decoded
planes carry the dtype implied by the stream's real bit-depth (uint8 for
8-bit, uint16 for 10-bit). This exercises the SPS-extradata fallback added to
FFmpegDemuxer.h when `codecpar->format == AV_PIX_FMT_NONE`.
"""

import os
import pytest

import accvlab.on_demand_video_decoder as nvc
import utils

VARIANTS_DIR = os.path.join(utils.get_data_dir(), "pix_fmt_variants")


# Each variant lists the filename, the codec_tag carried by the container, the
# bit-depth the stream actually encodes, and the dtype that should appear on the
# decoded Y plane. All variants here are NVDEC-decodable on the GPUs this
# package targets — see the support matrix for codec/bit-depth coverage:
# https://developer.nvidia.com/video-encode-decode-support-matrix
VARIANTS = [
    ("hevc_hev1_yuv420p.mp4", "hev1", 8, "|u1"),
    ("hevc_hev1_yuv420p10le.mp4", "hev1", 10, "|u2"),
    ("hevc_hvc1_yuv420p.mp4", "hvc1", 8, "|u1"),
    ("hevc_hvc1_yuv420p10le.mp4", "hvc1", 10, "|u2"),
    ("h264_avc1_yuv420p.mp4", "avc1", 8, "|u1"),
]


def _video_path(name):
    path = os.path.join(VARIANTS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"test asset missing: {path}")
    return path


@pytest.mark.parametrize(
    "filename, codec_tag, bit_depth, expected_dtype",
    VARIANTS,
    ids=[v[0] for v in VARIANTS],
)
def test_decode_from_gop_round_trip(filename, codec_tag, bit_depth, expected_dtype):
    """End-to-end: GetGOPList -> DecodeFromGOP must produce a plane of the
    correct dtype for the stream's actual bit-depth."""
    path = _video_path(filename)

    demuxer = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)

    gop_list = demuxer.GetGOPList([path], [0], useGOPCache=True)
    assert gop_list, f"GetGOPList returned empty for {filename}"
    gop_data, first_ids, gop_lens = gop_list[0]
    assert gop_data.size > 0, f"GOP data is empty for {filename}"
    assert first_ids == [0], f"unexpected first_ids={first_ids} for {filename}"
    assert gop_lens and gop_lens[0] > 0, f"unexpected gop_lens={gop_lens} for {filename}"

    frames = decoder.DecodeFromGOP(gop_data, [path], [0])
    assert len(frames) == 1, f"expected 1 frame, got {len(frames)} for {filename}"

    planes = frames[0].cuda()
    assert len(planes) >= 1, f"no planes returned for {filename}"

    cai = planes[0].__cuda_array_interface__
    assert cai["typestr"] == expected_dtype, (
        f"Y plane dtype mismatch for {filename}: got {cai['typestr']!r}, "
        f"expected {expected_dtype!r} for {bit_depth}-bit"
    )

    expected_bytes_per_sample = 2 if bit_depth >= 10 else 1
    actual_bytes_per_sample = int(cai["typestr"][-1])
    assert actual_bytes_per_sample == expected_bytes_per_sample, (
        f"Y plane element size mismatch for {filename}: got "
        f"{actual_bytes_per_sample}B, expected {expected_bytes_per_sample}B"
    )


@pytest.mark.parametrize(
    "filename, codec_tag, bit_depth, expected_dtype",
    VARIANTS,
    ids=[v[0] for v in VARIANTS],
)
def test_decode_does_not_raise_invalid_value(filename, codec_tag, bit_depth, expected_dtype):
    """Focused regression: the specific failure mode we fixed was
    `CUDA_ERROR_INVALID_VALUE` thrown from DecodeFromGOP because the GPU buffer
    was half the size NVDEC writes. Reproduce the exact call sequence the
    customer used and assert it does not raise that error."""
    path = _video_path(filename)
    demuxer = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)

    gop_data, _, _ = demuxer.GetGOPList([path], [0], useGOPCache=True)[0]
    # No prior RGB call to "prime" the GPU pool — exercise the raw YUV path
    # directly, which was the broken path before the SPS fallback.
    frames = decoder.DecodeFromGOP(gop_data, [path], [0])
    assert frames, f"DecodeFromGOP returned no frames for {filename}"
