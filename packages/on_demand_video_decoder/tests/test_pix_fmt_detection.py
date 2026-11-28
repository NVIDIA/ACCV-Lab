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
crashed the GOP decode path with `CUDA_ERROR_INVALID_VALUE`.

This parametrized test drives `GetGOPList` + `DecodeFromGOPList` over the matrix of stream
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

PIXEL_FORMAT_NV12 = 3
PIXEL_FORMAT_YUV444 = 4
PIXEL_FORMAT_P016 = 5
PIXEL_FORMAT_YUV444_16BIT = 6


# Each variant lists the filename, the codec_tag carried by the container, the
# bit-depth the stream actually encodes, and the dtype that should appear on the
# decoded Y plane. All variants here are NVDEC-decodable on the GPUs this
# package targets — see the support matrix for codec/bit-depth coverage:
# https://developer.nvidia.com/video-encode-decode-support-matrix
VARIANTS = [
    ("hevc_hev1_yuv420p.mp4", "hev1", 8, 3, "|u1", ((256, 256, 1), (128, 128, 2))),
    ("hevc_hev1_yuv420p10le.mp4", "hev1", 10, 5, "|u2", ((256, 256, 1), (128, 128, 2))),
    ("hevc_hvc1_yuv420p.mp4", "hvc1", 8, 3, "|u1", ((256, 256, 1), (128, 128, 2))),
    ("hevc_hvc1_yuv420p10le.mp4", "hvc1", 10, 5, "|u2", ((256, 256, 1), (128, 128, 2))),
    ("hevc_hvc1_yuv444p.mp4", "hvc1", 8, 4, "|u1", ((256, 256, 1),) * 3),
    ("h264_avc1_yuv420p.mp4", "avc1", 8, 3, "|u1", ((256, 256, 1), (128, 128, 2))),
]


def _video_path(name):
    path = os.path.join(VARIANTS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"test asset missing: {path}")
    return path


def test_sps_recovery_log_once_per_thread(capfd):
    path = _video_path("h264_avc1_yuv420p.mp4")
    infos = nvc.GetFastInitInfo([path])
    infos[0].format = -1  # AV_PIX_FMT_NONE forces SPS recovery on any FFmpeg build.

    message = "Recovered pix_fmt from SPS extradata: yuv420p"
    # Each single-file decoder reuses its own native demux thread across calls.
    for _ in range(2):
        decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
        for expected_logs in (1, 0):
            gop, _, _ = decoder.GetGOPList([path], [0], fastStreamInfos=infos)[0]
            assert gop.size > 0
            assert capfd.readouterr().out.count(message) == expected_logs


def _expected_plane_strides(pixel_format, luma_width, bytes_per_sample):
    if pixel_format in (PIXEL_FORMAT_NV12, PIXEL_FORMAT_P016):
        # Semi-planar 4:2:0: one Y plane followed by one interleaved UV plane.
        return (
            (
                luma_width * bytes_per_sample,
                bytes_per_sample,
                bytes_per_sample,
            ),
            (
                luma_width * bytes_per_sample,
                2 * bytes_per_sample,
                bytes_per_sample,
            ),
        )

    if pixel_format in (PIXEL_FORMAT_YUV444, PIXEL_FORMAT_YUV444_16BIT):
        # Planar 4:4:4: Y, U, and V are three full-resolution planes.
        plane_strides = (
            luma_width * bytes_per_sample,
            bytes_per_sample,
            bytes_per_sample,
        )
        return (plane_strides,) * 3

    raise ValueError(f"unsupported pixel format in stride check: {pixel_format}")


@pytest.mark.parametrize(
    "filename, codec_tag, bit_depth, expected_format, expected_dtype, expected_shapes",
    VARIANTS,
    ids=[v[0] for v in VARIANTS],
)
def test_decode_from_gop_round_trip(
    filename,
    codec_tag,
    bit_depth,
    expected_format,
    expected_dtype,
    expected_shapes,
):
    """End-to-end: GetGOPList -> DecodeFromGOPList must produce a plane of the
    correct dtype for the stream's actual bit-depth."""
    path = _video_path(filename)

    demuxer = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)

    frame_id = 33
    gop_list = demuxer.GetGOPList([path], [frame_id], useGOPCache=True)
    assert gop_list, f"GetGOPList returned empty for {filename}"
    gop_data, first_ids, gop_lens = gop_list[0]
    assert gop_data.size > 0, f"GOP data is empty for {filename}"
    assert first_ids[0] <= frame_id, f"unexpected first_ids={first_ids} for {filename}"
    assert gop_lens and gop_lens[0] > 0, f"unexpected gop_lens={gop_lens} for {filename}"
    assert frame_id < first_ids[0] + gop_lens[0]

    frames = decoder.DecodeFromGOPList([gop_data], [path], [frame_id])
    assert len(frames) == 1, f"expected 1 frame, got {len(frames)} for {filename}"

    assert frames[0].format == expected_format
    planes = frames[0].cuda()
    assert tuple(tuple(plane.shape) for plane in planes) == expected_shapes
    expected_frame_bytes = {
        3: 256 * 256 * 3 // 2,  # NV12
        4: 256 * 256 * 3,  # YUV444
        5: 256 * 256 * 3,  # P016
    }[expected_format]
    assert frames[0].framesize() == expected_frame_bytes

    expected_bytes_per_sample = 2 if bit_depth >= 10 else 1
    luma_width = expected_shapes[0][1]
    expected_strides = _expected_plane_strides(
        expected_format,
        luma_width,
        expected_bytes_per_sample,
    )
    actual_strides = tuple(tuple(plane.__cuda_array_interface__["strides"]) for plane in planes)
    assert (
        actual_strides == expected_strides
    ), f"plane strides mismatch for {filename}: got {actual_strides}, expected {expected_strides}"

    for plane_index, plane in enumerate(planes):
        cai = plane.__cuda_array_interface__
        assert cai["typestr"] == expected_dtype, (
            f"plane {plane_index} dtype mismatch for {filename}: got {cai['typestr']!r}, "
            f"expected {expected_dtype!r} for {bit_depth}-bit"
        )

    actual_bytes_per_sample = int(planes[0].__cuda_array_interface__["typestr"][-1])
    assert actual_bytes_per_sample == expected_bytes_per_sample, (
        f"Y plane element size mismatch for {filename}: got "
        f"{actual_bytes_per_sample}B, expected {expected_bytes_per_sample}B"
    )

    if expected_format in (3, 4):
        nvcv_views = frames[0].nvcv_image()
        expected_nvcv_shape = (256 + 128, 256, 1) if expected_format == 3 else (256 * 3, 256, 1)
        expected_nvcv_stride = (256, 2, 1) if expected_format == 3 else (256, 3, 1)
        assert len(nvcv_views) == 1
        assert tuple(nvcv_views[0].shape) == expected_nvcv_shape
        assert tuple(nvcv_views[0].stride) == expected_nvcv_stride
        assert nvcv_views[0].__cuda_array_interface__["typestr"] == "|u1"
