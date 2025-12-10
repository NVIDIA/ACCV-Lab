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

import torch
import numpy as np
import os
import random


def is_diff_in_range(to_comp_1, to_comp_2, tolerance):
    # support input as YUV tuple or single channel
    if isinstance(to_comp_1, tuple) and isinstance(to_comp_2, tuple):
        results = [is_diff_in_range(c1, c2, tolerance) for c1, c2 in zip(to_comp_1, to_comp_2)]
        in_range = all(r[0] for r in results)
        max_diff = max(r[1] for r in results)
        count_in_range = sum(r[2] for r in results)
        return in_range, max_diff, count_in_range

    # support single channel
    if torch.is_tensor(to_comp_1):
        to_comp_1 = to_comp_1.cpu()
    if torch.is_tensor(to_comp_2):
        to_comp_2 = to_comp_2.cpu()
    to_comp_1 = np.array(to_comp_1)
    to_comp_2 = np.array(to_comp_2)
    use_int = issubclass(to_comp_1.dtype.type, np.integer) and issubclass(to_comp_2.dtype.type, np.integer)
    type_to_use = np.int64 if use_int else np.float64
    diffs = np.abs(np.array(to_comp_1).astype(type_to_use) - np.array(to_comp_2).astype(type_to_use))
    max_diff = diffs.max()
    in_range = max_diff <= tolerance
    count_in_range = np.sum(diffs > tolerance)
    return in_range, max_diff, count_in_range


def diff(gop_decoded, opencv_decoded, file_names, frames, num_files, diff_tolerance=21):
    for file_name, frame, i in zip(file_names, frames, range(num_files)):
        same_with_opencv_dec, max_diff, count_in_range = is_diff_in_range(
            gop_decoded[i], opencv_decoded[i], diff_tolerance
        )
        print(file_name, " frame_id: ", frame, " pass: ", same_with_opencv_dec, " max_diff: ", max_diff)
        if not same_with_opencv_dec:
            print(f"Error: {file_name} {frame} is not same with opencv_decoded")
            return max_diff
    return 0


def gop_decode_bgr(nv_gop_dec, file_path_list, frame_id_list):
    try:
        decoded_frames = nv_gop_dec.DecodeN12ToRGB(file_path_list, frame_id_list, True)
        torch.cuda.nvtx.range_push("unsqueeze_tensor_list")
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        torch.cuda.nvtx.range_pop()
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_with_fast_init(nv_gop_dec, file_path_list, frame_id_list, fast_stream_infos):
    try:
        decoded_frames = nv_gop_dec.DecodeN12ToRGB(
            file_path_list, frame_id_list, as_bgr=True, fastStreamInfos=fast_stream_infos
        )
        torch.cuda.nvtx.range_push("unsqueeze_tensor_list")
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        torch.cuda.nvtx.range_pop()
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate(nv_gop_dec1, nv_gop_dec2, file_path_list, frame_id_list):
    try:
        packets, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(file_path_list, frame_id_list)
        decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(packets, file_path_list, frame_id_list, as_bgr=True)
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate_with_fast_init(
    nv_gop_dec1, nv_gop_dec2, file_path_list, frame_id_list, fast_stream_infos
):
    try:
        packets, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
            file_path_list, frame_id_list, fastStreamInfos=fast_stream_infos
        )
        decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(packets, file_path_list, frame_id_list, as_bgr=True)
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate_from_single_packet(nv_gop_dec, file_path_list, frame_id_list, packets):
    try:
        decoded_frames = nv_gop_dec.DecodeFromGOPRGB(packets, file_path_list, frame_id_list, as_bgr=True)
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate_from_multi_packets(nv_gop_dec, file_path_list, frame_id_list, packets_list):
    try:
        decoded_frames = nv_gop_dec.DecodeFromGOPListRGB(
            packets_list, file_path_list, frame_id_list, as_bgr=True
        )
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_data_dir():
    """
    Return absolute path to the test video data directory.

    This is resolved relative to this test package so that tests can be run
    from any current working directory.
    """
    test_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(test_root, "data")


def select_random_clip(path_base):
    subdirs = [d for d in os.listdir(path_base) if os.path.isdir(os.path.join(path_base, d))]
    if not subdirs:
        return None
    clip_dir = os.path.join(path_base, random.choice(subdirs))
    video_names = os.listdir(clip_dir)
    files = [os.path.join(clip_dir, file) for file in video_names]
    return files
