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

import torch
import random
import os

import utils
import accvlab.on_demand_video_decoder as nvc


def test_separate_access_single():
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

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        gop_decoded = utils.gop_decode_bgr_ddseparate(nv_gop_dec1, nv_gop_dec2, files, frames)
        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


def test_separate_access_from_gop_file_single():
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

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        packet_files = []
        for i in range(len(files)):
            numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(files[i : i + 1], frames[i : i + 1])
            packet_file = os.path.join("./", f"packets_{c:02d}_{i:02d}.bin")
            nvc.SavePacketsToFile(numpy_data, packet_file)
            packet_files.append(packet_file)

            # Verify file was created
            assert os.path.exists(packet_file), f"Packet file not created: {packet_file}"
            assert os.path.getsize(packet_file) == numpy_data.size, f"File size mismatch for {packet_file}"

        merged_numpy_data = nv_gop_dec2.LoadGops(packet_files)
        gop_decoded = utils.gop_decode_bgr_ddseparate_from_single_packet(
            nv_gop_dec2, files, frames, merged_numpy_data
        )

        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"

        for packet_file in packet_files:
            os.remove(packet_file)


def test_separate_access_from_gop_file_to_list_api_single():
    """
    Test LoadGopsToList + DecodeFromGOPListRGB API combination with file persistence.

    This test validates the workflow:
    1. Extract GOP data using GetGOP and save to separate files
    2. Load GOP data from files as a list using LoadGopsToList
    3. Decode using DecodeFromGOPListRGB
    4. Compare with OpenCV baseline

    Key difference from SeparateAccessFromGopFileSingleTest:
    - Uses LoadGopsToList (returns list of arrays) instead of LoadGops (returns merged array)
    - Uses DecodeFromGOPListRGB instead of DecodeFromGOPRGB
    """
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

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        num_frames_to_use = len(frames)

        # Step 1: Extract and save GOP data for each video
        packet_files = []
        for i in range(len(files)):
            numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(files[i : i + 1], frames[i : i + 1])
            packet_file = os.path.join("./", f"packets_list_{c:02d}_{i:02d}.bin")
            nvc.SavePacketsToFile(numpy_data, packet_file)
            packet_files.append(packet_file)

            # Verify file was created
            assert os.path.exists(packet_file), f"Packet file not created: {packet_file}"
            assert os.path.getsize(packet_file) == numpy_data.size, f"File size mismatch for {packet_file}"

        # Step 2: Load GOP data as a list using LoadGopsToList
        gop_data_list = nv_gop_dec2.LoadGopsToList(packet_files)

        # Validate the loaded list
        assert isinstance(
            gop_data_list, list
        ), f"LoadGopsToList should return a list, got {type(gop_data_list)}"
        assert len(gop_data_list) == len(
            packet_files
        ), f"LoadGopsToList returned {len(gop_data_list)} items, expected {len(packet_files)}"

        # Step 3: Decode using DecodeFromGOPListRGB
        decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
            gop_data_list,  # List of GOP data arrays
            files,  # List of file paths
            frames,  # List of frame IDs
            as_bgr=True,
        )

        assert decoded_frames is not None and len(decoded_frames) == num_frames_to_use, (
            f"DecodeFromGOPListRGB returned {len(decoded_frames) if decoded_frames else 0} "
            f"frames, expected {num_frames_to_use}"
        )

        # Convert to tensor format for comparison (kept for parity with original test)
        gop_decoded = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        assert len(gop_decoded) == num_frames_to_use

        # Cleanup: Remove temporary packet files
        for packet_file in packet_files:
            os.remove(packet_file)


def test_separate_access_from_multi_packets_merge_on_the_fly():
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

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        packets_list = []
        for i in range(len(files)):
            numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(files[i : i + 1], frames[i : i + 1])
            packets_list.append(numpy_data)

        gop_decoded = utils.gop_decode_bgr_ddseparate_from_multi_packets(
            nv_gop_dec2, files, frames, packets_list
        )

        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


def test_separate_access_from_multi_packets_merge_packets():
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

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        packets_list = []
        for i in range(len(files)):
            numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(files[i : i + 1], frames[i : i + 1])
            packets_list.append(numpy_data)

        merged_numpy_data = nv_gop_dec1.MergePacketDataToOne(packets_list)

        gop_decoded = utils.gop_decode_bgr_ddseparate_from_single_packet(
            nv_gop_dec2, files, frames, merged_numpy_data
        )

        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


def test_separate_access_gop_list_api():
    """
    Test GetGOPList + DecodeFromGOPListRGB API combination.

    This test validates the GetGOPList workflow by:
    1. Extracting separate GOP bundles for each video using GetGOPList
    2. Extracting gop_data from each bundle (ignoring metadata)
    3. Decoding all videos at once using DecodeFromGOPListRGB
    4. Comparing results with OpenCV baseline

    Key difference from other tests:
    - Uses GetGOPList (returns list of per-video bundles) + DecodeFromGOPListRGB
    - Instead of GetGOP (returns single merged bundle) + DecodeFromGOPRGB

    This combination demonstrates the complete workflow for per-video GOP management.
    """
    max_num_files_to_use = 6
    iter_num = 10
    path_base = utils.get_data_dir()

    # Stage 1 decoder: Extract per-video GOP data
    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )

    # Stage 2 decoder: Decode from individual GOP bundles
    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )

    frame_min = 0
    frame_max = 200

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        num_frames_to_use = len(frames)

        # Stage 1: Extract per-video GOP data using GetGOPList
        # Returns: [(gop_data1, first_ids1, gop_lens1), (gop_data2, ...), ...]
        gop_list = nv_gop_dec1.GetGOPList(files, frames)

        # Validate GetGOPList output
        assert gop_list is not None and len(gop_list) == len(files), (
            f"GetGOPList returned invalid data. "
            f"Expected {len(files)} bundles, got {len(gop_list) if gop_list else 0}"
        )

        # Validate each bundle structure
        for i, bundle in enumerate(gop_list):
            assert (
                isinstance(bundle, tuple) and len(bundle) == 3
            ), f"Bundle {i} has invalid structure. Expected tuple of 3 elements, got {type(bundle)}"
            gop_data, first_frame_ids, gop_lens = bundle
            assert gop_data is not None and len(gop_data) > 0, f"Bundle {i} has empty or None gop_data"

        # Stage 2: Decode all videos at once using DecodeFromGOPListRGB
        # Extract only the gop_data from each bundle (ignore first_frame_ids and gop_lens metadata)
        # GetGOPList returns: [(gop_data1, ids1, lens1), (gop_data2, ids2, lens2), ...]
        # DecodeFromGOPListRGB needs: [gop_data1, gop_data2, ...]
        gop_data_list = [gop_data for gop_data, _, _ in gop_list]

        # DecodeFromGOPListRGB: Batch decode multiple GOP bundles in one call
        # This is the optimal way to decode GetGOPList results
        decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
            gop_data_list,  # List of per-video GOP data from GetGOPList
            files,  # List of file paths (one per video)
            frames,  # List of frame IDs (one per video)
            as_bgr=True,  # Output in BGR format for comparison
        )

        assert decoded_frames is not None and len(decoded_frames) == num_frames_to_use, (
            f"DecodeFromGOPListRGB returned {len(decoded_frames) if decoded_frames else 0} "
            f"frames, expected {num_frames_to_use}"
        )

        # Convert to tensor format for comparison
        gop_decoded_list = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        assert len(gop_decoded_list) == num_frames_to_use


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
