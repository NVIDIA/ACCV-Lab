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
``accvlab.on_demand_video_decoder`` - GOP Files List API Decoding Sample

This sample demonstrates the ``LoadGopsToList`` API for per-video GOP file loading
and batch decoding. Unlike ``LoadGops`` which merges all files into one bundle,
``LoadGopsToList`` returns separate GOP data for each file, enabling:

- Per-video cache management and selective loading
- Distributed storage and processing optimization
- Independent video data handling
- Flexible batch decoding workflows

Key Features Demonstrated:
- Per-video GOP file storage and independent loading
- LoadGopsToList API for list-based GOP data management
- DecodeFromGOPListRGB for batch decoding from GOP list
- Selective video loading (load only needed videos)
- GPU-accelerated hardware decoding
- RGB/BGR format output options
- Comparison with merged LoadGops approach
"""

import os
import random
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleDecodeFromGopFilesListAPI():
    """
    Demonstrate GOP files list API for per-video decoding control.

    This function showcases the LoadGopsToList + DecodeFromGOPListRGB workflow:
    1. Phase 1: Extract and store GOP data to separate files (one per video)
    2. Phase 2: Load GOP files as a list (preserves per-video independence)
    3. Phase 3: Batch decode from GOP list
    4. Phase 4: Demonstrate selective loading (partial video set)

    Key Differences from LoadGops:
    - LoadGops: Merges all files → single bundle → DecodeFromGOPRGB
    - LoadGopsToList: Keeps files separate → list of bundles → DecodeFromGOPListRGB

    Benefits:
    - Load only needed videos from cache
    - Per-video cache management (expiration, priority)
    - Better suited for distributed systems
    - Reduced memory footprint for selective loading
    """

    # Set random seed for reproducible results
    random.seed(42)

    # Configuration
    max_num_files_to_use = 6
    frame_min = 0
    frame_max = 200
    num_iterations = 3

    # Sample video files from nuScenes multi-camera dataset
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    file_list = [
        os.path.join(sample_clip_dir, "moving_shape_circle_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_ellipse_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_hexagon_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_rect_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_triangle_h265.mp4"),
    ]

    camera_names = ["circle", "ellipse", "hexagon", "rect", "triangle"]

    print("=" * 80)
    print("NVIDIA accvlab.on_demand_video_decoder - GOP Files List API Sample")
    print("=" * 80)
    print(f"Processing {len(file_list)} video files from multi-camera setup")
    print(f"Demonstrating LoadGopsToList + DecodeFromGOPListRGB workflow")
    print(f"Frame range: {frame_min} to {frame_max}")
    print(f"Number of iterations: {num_iterations}")
    print("=" * 80)

    # Initialize NVIDIA GPU video decoders
    print("\n📦 Initializing NVIDIA GPU video decoders...")

    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )
    print("✓ Packet extraction decoder initialized")

    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,
        iGpu=0,
    )
    print("✓ GOP file decoder initialized")

    # Phase 1: Extract and store GOP packet data
    print("\n" + "=" * 80)
    print("PHASE 1: GOP Data Storage (Per-Video Files)")
    print("=" * 80)
    print("Extracting packet data and storing to separate binary files...")

    stored_gop_files = []
    target_frames = []

    for iteration in range(num_iterations):
        print(f"\n{'─' * 80}")
        print(f"Storage Iteration {iteration + 1}/{num_iterations}")
        print(f"{'─' * 80}")

        # Generate random frame indices
        frames = [random.randint(frame_min, frame_max) for _ in range(len(file_list))]
        target_frames.append(frames)
        print(f"Target frames: {frames}")

        try:
            packet_files = []
            for i in range(len(file_list)):
                print(f"\n  📹 Video {i + 1}/{len(file_list)}: {camera_names[i]}")

                # Extract packet data for single file
                numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
                    file_list[i : i + 1], frames[i : i + 1]
                )

                # Create unique filename for this video's GOP data
                packet_file = f"./gop_list_{iteration:02d}_{camera_names[i]}.bin"
                packet_files.append(packet_file)

                # Save packet data
                nvc.SavePacketsToFile(numpy_data, packet_file)

                # Verify file creation
                if not os.path.exists(packet_file):
                    raise FileNotFoundError(f"Packet file not created: {packet_file}")

                file_size = os.path.getsize(packet_file)
                print(f"     ✓ Saved: {packet_file}")
                print(f"     Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            stored_gop_files.append(packet_files)
            print(f"\n✓ Successfully stored {len(packet_files)} GOP files for iteration {iteration + 1}")

        except Exception as e:
            # Clean up on error
            for packet_file in packet_files:
                if os.path.exists(packet_file):
                    os.remove(packet_file)

            print(f"\n❌ GOP data storage failed")
            print(f"   Error: {type(e).__name__}: {e}")
            return 1

    print(f"\n✓ Phase 1 completed: Stored {len(stored_gop_files)} GOP file sets")

    # Phase 2: Load and decode ALL videos using LoadGopsToList
    print("\n" + "=" * 80)
    print("PHASE 2: Load All Videos and Batch Decode")
    print("=" * 80)
    print("Using LoadGopsToList + DecodeFromGOPListRGB for batch processing")

    for iteration in range(num_iterations):
        print(f"\n{'─' * 80}")
        print(f"Decoding Iteration {iteration + 1}/{num_iterations}")
        print(f"{'─' * 80}")

        packet_files = stored_gop_files[iteration]
        frames = target_frames[iteration]

        print(f"Loading {len(packet_files)} GOP files...")

        try:
            """
            LoadGopsToList: Load GOP files as separate bundles

            Key Difference from LoadGops:
            - LoadGops → single merged numpy array
            - LoadGopsToList → list of numpy arrays (one per file)

            Returns:
            - List of numpy arrays, each containing one video's GOP data
            - Preserves per-video independence for flexible processing
            """
            gop_data_list = nv_gop_dec2.LoadGopsToList(packet_files)

            print(f"✓ Loaded {len(gop_data_list)} GOP bundles")
            for i, gop_data in enumerate(gop_data_list):
                print(f"   Bundle {i + 1} ({camera_names[i]}): {len(gop_data):,} bytes")

            """
            DecodeFromGOPListRGB: Batch decode from GOP list
            
            Parameters:
            - gop_data_list: List of GOP data arrays (from LoadGopsToList)
            - file_list: List of video file paths
            - frames: List of frame IDs
            - as_bgr: Output format (True=BGR, False=RGB)
            
            Returns:
            - List of decoded RGB/BGR frames
            """
            print(f"\n🎬 Decoding {len(gop_data_list)} videos...")
            decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(gop_data_list, file_list, frames, as_bgr=True)

            print(f"✓ Successfully decoded {len(decoded_frames)} frames")

            # Convert to PyTorch tensors
            gop_decoded = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]

            if gop_decoded:
                first_tensor = gop_decoded[0]
                print(f"\n📊 Frame Analysis:")
                print(f"   Shape: {first_tensor.shape}")
                print(f"   Data type: {first_tensor.dtype}")
                print(f"   Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")
                print(f"   Dimensions: {first_tensor.shape[1]}x{first_tensor.shape[2]} (HxW)")

        except Exception as e:
            print(f"\n❌ Decoding failed")
            print(f"   Error: {type(e).__name__}: {e}")
            return 1

    # Phase 3: Demonstrate selective loading
    print("\n" + "=" * 80)
    print("PHASE 3: Selective Loading Demo")
    print("=" * 80)
    print("Loading only a subset of videos (demonstrating key advantage)")

    iteration = 0  # Use first iteration's files
    packet_files = stored_gop_files[iteration]
    frames = target_frames[iteration]

    # Select only front cameras (indices 3, 4, 5)
    selected_indices = [2, 3, 4]
    selected_files = [packet_files[i] for i in selected_indices]
    selected_video_paths = [file_list[i] for i in selected_indices]
    selected_frames = [frames[i] for i in selected_indices]
    selected_cameras = [camera_names[i] for i in selected_indices]

    print(f"\n🎯 Selective loading: Only {len(selected_indices)} out of {len(packet_files)} videos")
    print(f"   Selected cameras: {selected_cameras}")
    print(f"   Target frames: {selected_frames}")

    try:
        # Load only selected GOP files
        print(f"\n📂 Loading selected GOP files...")
        selected_gop_list = nv_gop_dec2.LoadGopsToList(selected_files)

        total_bytes = sum(len(gop) for gop in selected_gop_list)
        print(f"✓ Loaded {len(selected_gop_list)} GOP bundles ({total_bytes:,} bytes)")

        # Decode only selected videos
        print(f"\n🎬 Decoding selected videos...")
        decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
            selected_gop_list, selected_video_paths, selected_frames, as_bgr=True
        )

        print(f"✓ Successfully decoded {len(decoded_frames)} frames from selected videos")

        print(f"\n💡 Key Advantage:")
        print(f"   - Loaded only {len(selected_indices)}/{len(packet_files)} videos")
        print(f"   - Saved memory and I/O by not loading unneeded videos")
        print(f"   - Perfect for distributed caching and selective processing")

    except Exception as e:
        print(f"\n❌ Selective loading failed")
        print(f"   Error: {type(e).__name__}: {e}")
        return 1

    # Phase 4: API Comparison
    print("\n" + "=" * 80)
    print("API COMPARISON: LoadGops vs LoadGopsToList")
    print("=" * 80)

    print("\n📌 LoadGops + DecodeFromGOPRGB (Merged Approach):")
    print("   Usage:")
    print("     merged_data = decoder.LoadGops(file_paths)")
    print("     frames = decoder.DecodeFromGOPRGB(merged_data, videos, frame_ids)")
    print("   Characteristics:")
    print("     • Returns: Single merged numpy array")
    print("     • Decoding: One call with all data")
    print("     • Best for: Batch processing all videos together")
    print("     • Memory: One large bundle")
    print("     • Loading: All or nothing")

    print("\n📌 LoadGopsToList + DecodeFromGOPListRGB (List Approach) ⭐:")
    print("   Usage:")
    print("     gop_list = decoder.LoadGopsToList(file_paths)")
    print("     frames = decoder.DecodeFromGOPListRGB(gop_list, videos, frame_ids)")
    print("   Characteristics:")
    print("     • Returns: List of numpy arrays (one per video)")
    print("     • Decoding: Batch decode from list")
    print("     • Best for: Selective loading, distributed caching")
    print("     • Memory: Individual bundles, load independently")
    print("     • Loading: Can load subset of videos")

    print("\n🎯 Use LoadGopsToList when:")
    print("   ✓ Need to cache each video independently")
    print("   ✓ Want selective video loading from cache")
    print("   ✓ Distributed storage and processing")
    print("   ✓ Per-video cache management (expiration, priority)")
    print("   ✓ Large video collections (avoid loading all at once)")

    # Cleanup
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    print("Removing stored GOP files...")

    for packet_files in stored_gop_files:
        for packet_file in packet_files:
            if os.path.exists(packet_file):
                os.remove(packet_file)
                print(f"  ✓ Removed: {os.path.basename(packet_file)}")

    print("\n" + "=" * 80)
    print("✓ Sample completed successfully!")
    print("=" * 80)
    print(f"Processed {num_iterations} iterations with {len(file_list)} files each")
    print(f"Demonstrated selective loading with {len(selected_indices)} videos")

    return 0


if __name__ == "__main__":
    """
    Main entry point for the GOP Files List API demonstration.

    This sample demonstrates the LoadGopsToList + DecodeFromGOPListRGB workflow
    for per-video GOP management and selective loading capabilities.

    Key Advantages Over LoadGops:
    - Per-video independence: Each video's GOP data is separate
    - Selective loading: Load only needed videos from storage
    - Memory efficiency: Don't load all videos if not needed
    - Distributed caching: Better suited for distributed systems
    - Flexible processing: Process videos independently

    Prerequisites:
    1. NVIDIA GPU with hardware video decoding support
    2. CUDA drivers and runtime properly installed
    3. accvlab.on_demand_video_decoder library with LoadGopsToList support
    4. PyTorch for tensor conversion demonstrations
    5. Sample video files at specified paths (or update paths)
    6. Sufficient disk space for temporary GOP files
    7. Write permissions in current directory

    Workflow Summary:
    1. Extract GOP data and save to separate files (one per video)
    2. Load GOP files as a list using LoadGopsToList
    3. Batch decode using DecodeFromGOPListRGB
    4. Demonstrate selective loading (load only needed videos)
    5. Compare with LoadGops approach
    """
    exit_code = SampleDecodeFromGopFilesListAPI()
    exit(exit_code)
