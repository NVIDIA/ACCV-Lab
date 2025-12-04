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
``accvlab.on_demand_video_decoder`` - Separation Access with GetGOPList API Sample

This advanced sample demonstrates the GetGOPList API capability of ``accvlab.on_demand_video_decoder``,
which extends the Separation Access architecture by providing per-video GOP data extraction.
Unlike GetGOP which merges all video data into a single bundle, GetGOPList returns separate
GOP data for each video, enabling more granular control and optimized caching strategies.

Key Features Demonstrated:
- Per-video GOP data extraction with GetGOPList
- Independent caching and management of each video's GOP data
- Selective video decoding from cached GOP bundles
- Distributed processing and storage optimization
- Two-stage video processing with enhanced granularity control
- Enable per-video caching, partial loading, and parallel processing

Comparison with GetGOP:
- GetGOP: Merges all videos into one bundle → suitable for batch processing
- GetGOPList: Separate bundle per video → suitable for distributed caching and selective loading
"""

import os
import random
import numpy as np
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleSeparationAccessGOPListAPI():
    """
    Demonstrate separation access video decoding using GetGOPList API.

    This function showcases the advanced GetGOPList capability of accvlab.on_demand_video_decoder:
    1. Stage 1: Extract per-video GOP data using GetGOPList (not merged)
    2. Stage 2: Selective decoding from individual video GOP bundles
    3. Cache Management: Demonstrating per-video caching and retrieval
    4. Flexibility: Processing specific videos without loading all GOP data

    The per-video approach provides enhanced control for distributed systems,
    enabling selective loading, parallel processing, and optimized memory usage.
    """

    # Sample video files from nuScenes multi-camera dataset
    # These represent synchronized camera views from autonomous vehicle sensors
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    file_path_list = [
        os.path.join(sample_clip_dir, "moving_shape_circle_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_ellipse_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_hexagon_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_rect_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_triangle_h265.mp4"),
    ]

    # Camera names for better readability
    camera_names = ["circle", "ellipse", "hexagon", "rect", "triangle"]

    # Configuration: Maximum number of video files for concurrent processing
    max_num_files_to_use = 6

    print("=" * 80)
    print("NVIDIA accvlab.on_demand_video_decoder - GetGOPList API Demonstration")
    print("=" * 80)
    print(f"Configuration: {max_num_files_to_use} video streams")
    print(f"Processing: {len(file_path_list)} camera views")
    print("Architecture: Two-stage processing with per-video GOP separation")
    print("=" * 80)

    # STAGE 1 DECODER: Dedicated to per-video GOP extraction
    print("\n📦 Initializing Stage 1 Decoder (Per-Video GOP Extraction)...")
    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )
    print("✓ Stage 1 decoder initialized - ready for per-video GOP extraction")

    # STAGE 2 DECODER: Dedicated to selective GOP decoding
    print("\n🎬 Initializing Stage 2 Decoder (Selective GOP Decoding)...")
    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )
    print("✓ Stage 2 decoder initialized - ready for frame decoding")

    # STAGE 1: PER-VIDEO GOP EXTRACTION
    print("\n" + "=" * 80)
    print("STAGE 1: Per-Video GOP Data Extraction")
    print("=" * 80)
    print(f"🔄 Extracting GOP data from {len(file_path_list)} video files...")
    print(f"Target frame: 77 for all videos")

    '''
    GetGOPList performs per-video GOP extraction (unlike GetGOP which merges all)
    
    Parameters:
    - filepaths: List of video file paths to process
    - frame_ids: List of target frame indices for GOP extraction
    
    Returns:
    - List of tuples, one per video: (gop_data, first_frame_ids, gop_lens)
    
    Key Difference from GetGOP:
    - GetGOP: Returns single merged bundle → (merged_data, merged_first_ids, merged_gop_lens)
    - GetGOPList: Returns list of separate bundles → [(data1, ids1, lens1), (data2, ids2, lens2), ...]
    '''
    gop_list = nv_gop_dec1.GetGOPList(file_path_list, [77] * len(file_path_list))

    print(f"✓ Successfully extracted GOP data for {len(gop_list)} videos")
    print("\nPer-Video GOP Data Summary:")
    print("-" * 80)

    # Display detailed information for each video's GOP data
    for i, (gop_data, first_frame_ids, gop_lens) in enumerate(gop_list):
        print(f"  Video {i + 1} ({camera_names[i]}):")
        print(f"    GOP data size: {len(gop_data):,} bytes ({len(gop_data) / 1024 / 1024:.2f} MB)")
        print(f"    Number of GOPs: {len(first_frame_ids)}")
        print(f"    First frame IDs: {first_frame_ids}")
        print(f"    GOP lengths: {gop_lens}")
        print(f"    Frame range: [{first_frame_ids[0]}, {first_frame_ids[0] + gop_lens[0] - 1}]")

    # DEMONSTRATE PER-VIDEO CACHING ADVANTAGE
    print("\n" + "=" * 80)
    print("CACHING DEMONSTRATION: Per-Video Storage")
    print("=" * 80)
    print("GetGOPList enables efficient per-video caching strategies:")
    print("  • Each video's GOP data can be cached independently")
    print("  • Selective loading: Load only required videos")
    print("  • Distributed storage: Store in separate cache locations")
    print("  • Memory optimization: Unload unused video GOP data")
    print("\nSimulating cache storage (not actually writing to disk)...")

    # Simulate per-video cache mapping
    gop_cache = {}
    for i, (gop_data, first_frame_ids, gop_lens) in enumerate(gop_list):
        cache_key = f"video_{i}_{camera_names[i]}_frame_77"
        gop_cache[cache_key] = {
            'gop_data': gop_data,
            'first_frame_ids': first_frame_ids,
            'gop_lens': gop_lens,
            'filepath': file_path_list[i],
        }
        print(f"  ✓ Cached: {cache_key} ({len(gop_data):,} bytes)")

    # STAGE 2: SELECTIVE VIDEO DECODING
    print("\n" + "=" * 80)
    print("STAGE 2: Selective Video Decoding from Cached GOP Data")
    print("=" * 80)

    num_iterations = 3
    print(f"Processing {num_iterations} iterations with selective video decoding\n")

    for iteration in range(num_iterations):
        print(f"\n{'─' * 80}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'─' * 80}")

        # Randomly select a subset of videos to decode (simulating selective loading)
        num_videos_to_decode = random.randint(2, len(file_path_list))
        selected_indices = sorted(random.sample(range(len(file_path_list)), num_videos_to_decode))

        print(f"🎯 Selectively decoding {num_videos_to_decode} out of {len(file_path_list)} videos")
        print(f"   Selected cameras: {[camera_names[i] for i in selected_indices]}")

        # Load only selected videos' GOP data from cache
        selected_gop_data_list = []
        selected_filepaths = []
        selected_frame_ids = []

        for idx in selected_indices:
            cache_key = f"video_{idx}_{camera_names[idx]}_frame_77"
            cached_item = gop_cache[cache_key]

            # Generate random frame within GOP range
            first_frame_id = cached_item['first_frame_ids'][0]
            gop_len = cached_item['gop_lens'][0]
            random_frame = random.randint(first_frame_id, first_frame_id + gop_len - 1)

            selected_gop_data_list.append(cached_item['gop_data'])
            selected_filepaths.append(cached_item['filepath'])
            selected_frame_ids.append(random_frame)

        print(f"   Frame IDs to decode: {selected_frame_ids}")
        print(f"   Loading GOP data from cache...")

        try:
            '''
            Decode frames from multiple GOP bundles using DecodeFromGOPListRGB

            Key Advantage: We can decode from any subset of cached GOP data
            without needing to load or process all videos.

            Using DecodeFromGOPListRGB for optimal batch decoding performance:
            - GetGOPList output: [(gop_data1, ids, lens), (gop_data2, ...), ...]
            - DecodeFromGOPListRGB input: [gop_data1, gop_data2, ...]
            '''
            print(
                f"\n   🎬 Batch decoding {len(selected_gop_data_list)} videos using DecodeFromGOPListRGB..."
            )
            for i, idx in enumerate(selected_indices):
                print(
                    f"      Video {i + 1}: {camera_names[idx]} - {len(selected_gop_data_list[i]):,} bytes - Frame {selected_frame_ids[i]}"
                )

            '''
            DecodeFromGOPListRGB: Batch decode multiple GOP bundles in one call
            
            Parameters:
            - gop_data_list: List of GOP data arrays (from GetGOPList)
            - filepaths: List of video file paths
            - frame_ids: List of frame IDs to decode
            - as_bgr: Output format (True=BGR, False=RGB)
            
            This is the optimal way to decode GetGOPList results!
            '''
            decoded_frames_all = nv_gop_dec2.DecodeFromGOPListRGB(
                selected_gop_data_list,  # List of GOP data from selected videos
                selected_filepaths,  # List of file paths
                selected_frame_ids,  # List of frame IDs
                True,  # BGR format
            )

            print(f"\n   ✓ Successfully decoded {len(decoded_frames_all)} frame(s) in one batch call")

            # Convert first decoded frame to PyTorch tensor for analysis
            if decoded_frames_all:
                first_frame = decoded_frames_all[0]
                tensor = torch.as_tensor(first_frame)

                print(f"\n📊 First Decoded Frame Analysis:")
                print(f"   Shape: {tensor.shape} (Height × Width × Channels)")
                print(f"   Data type: {tensor.dtype}")
                print(f"   Value range: [{tensor.min().item()}, {tensor.max().item()}]")
                print(f"   Memory size: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f} MB")

        except Exception as e:
            print(f"\n❌ Error during selective decoding in iteration {iteration + 1}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {e}")
            print("\n   Diagnostic Guide:")
            print("     • Verify GOP data integrity from GetGOPList")
            print("     • Check frame indices are within GOP range")
            print("     • Ensure decoder has sufficient GPU memory")
            print("     • Validate cached data hasn't been corrupted")
            exit(-1)

    # SUMMARY AND BENEFITS
    print("\n" + "=" * 80)
    print("GetGOPList API Benefits Summary")
    print("=" * 80)
    print("✓ Per-Video Granularity:")
    print("  • Each video's GOP data is independent and separately manageable")
    print("  • Enables fine-grained caching strategies")
    print("\n✓ Selective Loading:")
    print("  • Load only required videos from cache")
    print("  • Reduces memory footprint for large video collections")
    print("\n✓ Distributed Processing:")
    print("  • Each GOP bundle can be stored/processed independently")
    print("  • Facilitates parallel processing across multiple workers")
    print("\n✓ Flexible Caching:")
    print("  • Per-video cache invalidation and updates")
    print("  • Different cache policies per video (e.g., by priority)")
    print("\n✓ Scalability:")
    print("  • Better suited for large-scale video processing pipelines")
    print("  • Reduced inter-video dependencies")
    print("=" * 80)


def CompareGetGOPvsGetGOPList():
    """
    Demonstrate the difference between GetGOP and GetGOPList APIs.

    This comparison helps understand when to use each API:
    - GetGOP: Best for batch processing where all videos are processed together
    - GetGOPList: Best for scenarios requiring per-video control and caching
    """

    print("\n" + "=" * 80)
    print("API COMPARISON: GetGOP vs GetGOPList")
    print("=" * 80)

    print("\n📌 GetGOP + DecodeFromGOPRGB (Merged Approach):")
    print("   Usage:")
    print("     merged_data, first_ids, gop_lens = decoder.GetGOP(files, frame_ids)")
    print("     frames = decoder.DecodeFromGOPRGB(merged_data, files, frame_ids)")
    print("   Characteristics:")
    print("     • Returns: Single merged GOP bundle for all videos")
    print("     • Decoding: One call with merged bundle")
    print("     • Best for: Batch processing, when all videos processed together")
    print("     • Memory: One large bundle in memory")
    print("     • Caching: Single cache entry for all videos")

    print("\n📌 GetGOPList + DecodeFromGOPListRGB (Per-Video Approach):")
    print("   Usage:")
    print("     gop_list = decoder.GetGOPList(files, frame_ids)")
    print("     # gop_list = [(data1, ids1, lens1), (data2, ids2, lens2), ...]")
    print("     gop_data_list = [data for data, _, _ in gop_list]")
    print("     frames = decoder.DecodeFromGOPListRGB(gop_data_list, files, frame_ids)")
    print("   Characteristics:")
    print("     • Returns: Separate GOP bundle for each video")
    print("     • Decoding: Batch decode from list of bundles")
    print("     • Best for: Selective loading, distributed caching, parallel processing")
    print("     • Memory: Individual bundles, can load/unload independently")
    print("     • Caching: Per-video cache entries, fine-grained control")

    print("\n🎯 Use Case Recommendations:")
    print("   Use GetGOP when:")
    print("     ✓ Processing all videos together in a batch")
    print("     ✓ Simple caching strategy (all or nothing)")
    print("     ✓ Videos are always processed as a group")

    print("\n   Use GetGOPList when:")
    print("     ✓ Need to cache each video independently")
    print("     ✓ Selective video loading from cache")
    print("     ✓ Distributed storage and processing")
    print("     ✓ Per-video cache management (expiration, priority, etc.)")
    print("     ✓ Large video collections where loading all GOP data is impractical")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main entry point for the GetGOPList API demonstration.

    This sample showcases advanced per-video GOP extraction and management
    capabilities that extend the Separation Access architecture with enhanced
    granularity and control.

    Prerequisites:
    1. NVIDIA GPU with hardware video decoding support
    2. CUDA drivers and runtime properly installed
    3. accvlab.on_demand_video_decoder library with GetGOPList support
    4. Multi-camera video dataset (nuScenes format recommended)
    5. PyTorch for tensor conversion demonstrations

    Performance Characteristics:
    - Stage 1 (GetGOPList): Per-video GOP extraction with independent bundles
    - Stage 2 (DecodeFromGOP): Selective decoding from individual bundles
    - Memory Efficiency: Load only required video GOP data
    - Caching Granularity: Per-video cache management
    - Scalability: Better suited for large video collections

    Architecture Benefits:
    - Enhanced caching strategies with per-video control
    - Reduced memory footprint through selective loading
    - Distributed processing and storage optimization
    - Fine-grained cache invalidation and updates
    - Better scalability for large video collections
    """
    print("\n" + "=" * 80)
    print("NVIDIA accvlab.on_demand_video_decoder - GetGOPList API Sample")
    print("=" * 80)
    print("Demonstrating per-video GOP extraction and selective decoding")
    print("with enhanced caching and memory management capabilities")
    print("=" * 80 + "\n")

    # Run main demonstration
    SampleSeparationAccessGOPListAPI()

    # Show API comparison
    CompareGetGOPvsGetGOPList()

    print("\n✓ Sample completed successfully!")
