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
``accvlab.on_demand_video_decoder`` - Separation Access Video Decoding Sample

This advanced sample demonstrates the Separation Access capability of ``accvlab.on_demand_video_decoder``,
which provides a two-stage approach to video processing by separating demuxing and decoding
operations. This architecture enables enhanced control, flexibility, and optimization
opportunities for complex video processing workflows.

Key Features Demonstrated:
- Multi-file concurrent decoding (up to configurable limit)
- Random frame access without sequential decoding overhead
- GPU-accelerated hardware decoding
- RGB/BGR/NV12 format output options
- Device memory output for further processing
- Two-stage video processing: demuxing separation from decoding
- Enable single demuxing and reuse its results for multiple decoding processes.
"""

import os
import random
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleSeparationAccess():
    """
    Demonstrate separation access video decoding using dual-stage processing.

    This function showcases the advanced Separation Access capability of accvlab.on_demand_video_decoder:
    1. Stage 1: Demuxing video files to extract compressed packet data
    2. Stage 2: Decoding packets directly to RGB frames without re-demuxing
    3. Demonstrating the flexibility and control benefits of separated operations
    4. Showing practical applications in complex video processing workflows

    The two-stage approach provides enhanced control over the video processing pipeline,
    enabling optimizations and flexibility not possible with traditional integrated decoding.
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

    # Configuration: Maximum number of video files for concurrent processing
    # This limit applies to both demuxing and decoding stages independently
    max_num_files_to_use = 6

    print("Initializing NVIDIA GPU video decoder with Separation Access capability...")
    print(f"Configuration: {max_num_files_to_use} concurrent video streams")
    print("Architecture: Two-stage processing (demuxing + decoding separation)")

    # STAGE 1 DECODER: Dedicated to packet extraction and demuxing operations
    # This decoder specializes in efficiently parsing video files and extracting compressed packets
    print("\nInitializing Stage 1 Decoder (Packet Extraction)...")
    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files for packet extraction
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )
    print("✓ Stage 1 decoder initialized - ready for packet extraction")

    # STAGE 2 DECODER: Dedicated to packet decoding and frame reconstruction
    # This decoder specializes in converting compressed packets directly to RGB frames
    print("Initializing Stage 2 Decoder (Packet Decoding)...")
    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files for frame decoding
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )
    print("✓ Stage 2 Extracting packets from video files...")
    print("✓ Dual-stage separation access architecture ready")

    # STAGE 2: PACKET EXTRACTION AND DEMUXING
    # Extract compressed packet data without decoding to RGB frames
    print(f"\n🔄 Stage 1: Extracting packets from {len(file_path_list)} video files...")

    '''
    GetGOP performs selective demuxing to extract compressed video GOP data
    
    Parameters:
    - filepaths: List of video file paths to process
    - frame_ids: List of target frame indices for packet extraction
    - useGOPCache: If True, enables GOP caching. When the same video file is requested
                   with a frame_id that falls within a previously cached GOP range,
                   the cached data is returned directly without re-demuxing.
                   Default is False.
    
    Returns:
    - packets: Compressed packet data for the specified frames
    - first_frame_ids: Actual first frame IDs in the extracted GOPs
    - gop_lens: Length information for each GOP (Group of Pictures)
    
    Cache hit condition: first_frame_id <= frame_id < first_frame_id + gop_len
    
    Example with caching:
        # First call - fetches GOP data from video files
        packets, first_ids, gop_lens = decoder.GetGOP(files, [77, 77], useGOPCache=True)
        # Second call with frame_id=80 in same GOP - returns from cache (no I/O)
        packets, first_ids, gop_lens = decoder.GetGOP(files, [80, 80], useGOPCache=True)
    '''
    packets, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
        file_path_list, [77] * len(file_path_list), useGOPCache=True
    )

    # Perform multiple separation access decoding iterations
    num_iterations = 5
    total_packets_extracted = 0
    total_frames_decoded = 0

    print(f"\n" + "=" * 60)
    print("SEPARATION ACCESS DECODING DEMONSTRATION")
    print("=" * 60)
    print(f"Processing {num_iterations} iterations with two-stage architecture")

    # STAGE 2: DIRECT PACKET DECODING TO RGB FRAMES
    # Decode the extracted packets directly to RGB frames without re-demuxing
    print(f"\n🎨 Stage 2: Decoding packets to RGB frames...")
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Generate random frame indices for each video file
        # This simulates real-world scenarios where specific frames are needed for analysis
        frame_id_list = [
            random.randint(first_frame_ids[i], first_frame_ids[i] + gop_lens[i] - 1)
            for i in range(len(file_path_list))
        ]
        print(f"Target frame indices: {frame_id_list}")

        try:
            '''
            With useGOPCache=True, we can re-call GetGOP for each iteration.
            If the frame_id falls within the cached GOP range, it returns cached data.
            If not, it fetches the new GOP and updates the cache automatically.

            This eliminates the need for manual cache management by the user.
            '''
            packets, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
                file_path_list, frame_id_list, useGOPCache=True
            )

            '''
            DecodeFromGOPRGB performs direct GOP-to-frame conversion
            
            Parameters:
            - packets: Compressed GOP data from Stage 1 (GetGOP)
            - filepaths: Original video file paths (for reference/validation)
            - frame_ids: Target frame indices to decode from packets
            - as_bgr: Output format flag (True=BGR, False=RGB)
            
            Returns:
            - List of decoded RGB/BGR frames in host memory
            '''
            decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(
                packets,  # Compressed packet data from Stage 1
                file_path_list,  # Original video file paths
                frame_id_list,  # Target frame indices
                True,  # Output in BGR format (OpenCV compatible)
            )

            total_frames_decoded += len(decoded_frames)

            # Convert decoded frames to PyTorch tensors for ML applications
            print("🔄 Converting frames to PyTorch tensors for ML pipeline integration...")
            tensor_list = [torch.unsqueeze(torch.as_tensor(frame), 0) for frame in decoded_frames]

            # Display detailed analysis of the first decoded frame
            first_tensor = tensor_list[0]
            print(f"\nFrame Analysis (Camera 1):")
            print(f"  Tensor shape: {first_tensor.shape}")  # Expected: [1, height, width, channels]
            print(f"  Data type: {first_tensor.dtype}")  # Typically uint8 for image data
            print(f"  Memory location: {first_tensor.device}")  # CPU (host memory)
            print(f"  Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")
            print(f"  Frame dimensions: {first_tensor.shape[1]}x{first_tensor.shape[2]} (HxW)")
            print(
                f"  Color format: {'BGR' if first_tensor.shape[3] == 3 else 'Unknown'} ({first_tensor.shape[3]} channels)"
            )

        except Exception as e:
            # Comprehensive error handling with stage-specific diagnostics
            print(f"\n❌ Separation access decoding failed in iteration {iteration + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("\nDiagnostic Guide:")
            print("  Stage 1 Issues (Packet Extraction):")
            print("    • Verify video files exist and are accessible")
            print("    • Check frame indices are within valid video bounds")
            print("    • Ensure video codecs are supported by hardware decoder")
            print("    • Validate file formats and container compatibility")
            print("  Stage 2 Issues (Packet Decoding):")
            print("    • Confirm packet data integrity from Stage 1")
            print("    • Verify GPU memory availability for frame decoding")
            print("    • Check decoder configuration consistency between stages")
            print("    • Ensure frame indices match extracted packet ranges")
            print("Continuing with next iteration...\n")
            exit(-1)


if __name__ == "__main__":
    """
    Main entry point for the Separation Access video decoding demonstration.

    This sample showcases advanced two-stage video processing capabilities that
    provide enhanced control and flexibility over traditional integrated decoding.

    Prerequisites:
    1. NVIDIA GPU with hardware video decoding support
    2. CUDA drivers and runtime properly installed
    3. accvlab.on_demand_video_decoder library with Separation Access support
    4. Multi-camera video dataset (nuScenes format recommended)
    5. PyTorch for tensor conversion demonstrations

    Performance Characteristics:
    - Stage 1 (Packet Extraction): Optimized demuxing with selective extraction
    - Stage 2 (Packet Decoding): Direct packet-to-frame conversion
    - Memory Efficiency: Packet data cached for potential reuse
    - Control Granularity: Independent control over demuxing and decoding stages

    Architecture Benefits:
    - Enhanced debugging and profiling capabilities
    - Flexible integration with custom processing pipelines
    - Optimized resource utilization through process separation
    - Foundation for advanced video analysis applications
    """
    print("NVIDIA accvlab.on_demand_video_decoder - Separation Access Video Decoding")
    print("=============================================================")
    print("Advanced sample demonstrating two-stage video processing")
    print("with separated demuxing and decoding operations\n")

    SampleSeparationAccess()
