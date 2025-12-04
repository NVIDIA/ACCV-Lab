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
``accvlab.on_demand_video_decoder`` - Packets List Decoding Sample

This sample demonstrates how to use ``accvlab.on_demand_video_decoder`` library for
efficient decoding from multiple packet lists with on-the-fly merging. This approach
is specifically designed for special scenarios where packet data is extracted
separately but needs to be decoded simultaneously in a unified process.

Key Features Demonstrated:
- Multi-file concurrent decoding (up to configurable limit)
- On-the-fly packet list merging for batch processing
- GPU-accelerated hardware decoding from multiple packet sources
- RGB/BGR/NV12 format output options
- Device memory output for further processing
- Optimized for PyTorch DataLoader integration scenarios
- Special handling for distributed demuxing with centralized decoding
"""

import os
import random
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleDecodeFromPacketsList():
    """
    Demonstrate packets list decoding using NVIDIA's GPU-accelerated decoder.
    This function showcases a specialized approach for scenarios where packet data
    is extracted separately (e.g., by PyTorch DataLoader workers) but needs to be
    decoded simultaneously in the main process.

    This approach is particularly beneficial for distributed video processing scenarios:
    - PyTorch DataLoader workers demux batch_size clips in parallel
    - Main process receives multiple packet lists from different workers
    - Simultaneous decoding of all batch_size clips in the main process
    - Efficient on-the-fly merging without intermediate file storage

    This function showcases the core functionality of accvlab.on_demand_video_decoder for packets list:
    1. Separate packet extraction for each video file (simulating worker processes)
    2. On-the-fly merging of multiple packet lists
    3. Simultaneous decoding of all packets in the main process
    4. Converting decoded frames to PyTorch tensors for ML applications
    5. Handling decoding errors gracefully with comprehensive error reporting

    The example uses a multi-camera setup from nuScenes dataset to demonstrate
    real-world usage patterns in distributed video processing applications.
    """

    # Set random seed for reproducible results
    random.seed(27)

    # Configuration: Maximum number of video files to decode simultaneously
    max_num_files_to_use = 6

    # Frame range for random frame selection
    frame_min = 0
    frame_max = 200

    # Number of iterations to demonstrate the workflow
    num_iterations = 5

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

    print("NVIDIA accvlab.on_demand_video_decoder - Packets List Decoding Sample")
    print("==========================================================")
    print(f"Processing {len(file_list)} video files from multi-camera setup")
    print("Video resolution: 1600x900 pixels")
    print(f"Frame range: {frame_min} to {frame_max}")
    print(f"Number of iterations: {num_iterations}")
    print("\nSpecial Use Case: PyTorch DataLoader Integration")
    print("- Simulates DataLoader workers extracting packets separately")
    print("- Main process merges and decodes all packets simultaneously")
    print("- Optimized for batch_size concurrent clip processing")

    # Initialize NVIDIA GPU video decoders
    print(f"\nInitializing NVIDIA GPU video decoders...")

    # Initialize first decoder for packet extraction (simulating worker processes)
    print("Creating packet extraction decoder (simulating DataLoader workers)...")
    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )

    # Initialize second decoder for packets list decoding (main process)
    print("Creating packets list decoder (main process)...")
    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )

    print(
        f"Decoders initialized successfully on GPU 0 with support for {max_num_files_to_use} concurrent files"
    )

    # Perform multiple packets list decoding iterations
    print(f"\nStarting {num_iterations} packets list decoding iterations...")

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Generate random frame indices for each video file
        frames = [random.randint(frame_min, frame_max) for _ in range(len(file_list))]
        print(f"Target frame indices: {frames}")

        try:
            """
            Phase 1: Separate packet extraction (simulating DataLoader workers)

            In a real PyTorch DataLoader scenario, this would happen in separate
            worker processes, each extracting packets for their assigned video files.
            Here we simulate this by extracting packets for each file individually.
            """
            print("Phase 1: Extracting packets separately (simulating worker processes)...")

            packets_list = []
            for i in range(len(file_list)):
                print(
                    f"  Worker {i+1}: Extracting packets for {os.path.basename(file_list[i])} (frame {frames[i]})"
                )

                # Extract packet data for single file and frame (simulating worker process)
                numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
                    file_list[i : i + 1], frames[i : i + 1]
                )

                packets_list.append(numpy_data)
                print(f"    Extracted packet data: {numpy_data.size} bytes")

            print(f"Successfully extracted packets from {len(packets_list)} files")

            """
            Phase 2: On-the-fly merging and simultaneous decoding (main process)
            
            This phase simulates the main process receiving packet lists from
            multiple DataLoader workers and decoding them simultaneously.
            
            DecodeFromPacketListRGB Parameters:
            - packets_list: List of numpy arrays containing packet data from each file
            - file_path_list: Original video file paths (for metadata)
            - frame_id_list: Target frame indices
            - as_bgr: Output format flag (True=BGR, False=RGB)
            
            Returns:
            - List of decoded frames in host memory as numpy-compatible arrays
            - Each frame maintains original video resolution and color depth
            - Frames are ready for immediate processing or tensor conversion
            """
            print("Phase 2: On-the-fly merging and simultaneous decoding (main process)...")
            print("  Merging packet lists and decoding all frames simultaneously...")

            decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(packets_list, file_list, frames, as_bgr=True)

            print(f"Successfully decoded {len(decoded_frames)} frames from packets list")

            # Convert decoded frames to PyTorch tensors for ML applications
            print("Converting frames to PyTorch tensors...")
            gop_decoded = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]

            # Display tensor information for first frame (representative of all frames)
            if gop_decoded:
                first_tensor = gop_decoded[0]
                print(f"Tensor shape: {first_tensor.shape}")  # Expected: [1, height, width, channels]
                print(f"Tensor dtype: {first_tensor.dtype}")  # Typically uint8 for image data
                print(f"Tensor device: {first_tensor.device}")  # CPU (host memory)
                print(f"Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")
                print(f"Frame dimensions: {first_tensor.shape[1]}x{first_tensor.shape[2]} (HxW)")
                print(
                    f"Color channels: {first_tensor.shape[3]} ({'BGR' if first_tensor.shape[3] == 3 else 'Unknown'})"
                )

            # Display batch information for DataLoader scenario
            print(f"Batch processing summary:")
            print(f"  Total files processed: {len(file_list)}")
            print(f"  Total frames decoded: {len(decoded_frames)}")
            print(f"  Total packet data size: {sum(packet.size for packet in packets_list)} bytes")
            print(
                f"  Average packet size per file: {sum(packet.size for packet in packets_list) // len(packets_list)} bytes"
            )

        except Exception as e:
            # Comprehensive error handling for production robustness
            print(f"Packets list decoding failed in iteration {iteration + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("Possible causes:")
            print("  - Video files not accessible at specified paths")
            print("  - Frame index exceeds video length")
            print("  - Insufficient GPU memory for concurrent decoding")
            print("  - Unsupported video codec or container format")
            print("  - Packet data corruption or format mismatch")
            print("  - Worker process communication issues (in real DataLoader scenario)")
            print("  - Memory allocation failures for packet merging")
            return 1

    print(f"\nPackets list decoding completed successfully!")
    print(f"Processed {num_iterations} iterations with {len(file_list)} files each")
    print("\nKey Benefits for PyTorch DataLoader Integration:")
    print("- Parallel packet extraction across multiple worker processes")
    print("- Efficient on-the-fly merging without intermediate storage")
    print("- Simultaneous decoding of batch_size clips in main process")
    print("- Optimized memory usage and GPU utilization")
    print("- Reduced I/O overhead compared to file-based approaches")
    return 0


if __name__ == "__main__":
    """
    Main entry point for the packets list decoding demonstration.

    This sample demonstrates the specialized approach for distributed video processing
    scenarios, particularly PyTorch DataLoader integration:

    1. DataLoader workers extract packets separately in parallel
    2. Main process receives multiple packet lists from workers
    3. On-the-fly merging and simultaneous decoding of all packets
    4. Efficient batch processing without intermediate file storage

    This approach is ideal for scenarios where:
    - Multiple worker processes are demuxing video clips
    - Main process needs to decode batch_size clips simultaneously
    - Memory efficiency and GPU utilization are critical
    - Avoiding intermediate file I/O is beneficial

    Ensure that:
    1. NVIDIA GPU drivers and CUDA are properly installed
    2. accvlab.on_demand_video_decoder library is available in Python path
    3. Sample video files exist at specified paths (or update paths accordingly)
    4. PyTorch is installed for tensor conversion examples
    5. Sufficient GPU memory is available for concurrent decoding
    6. Sufficient system memory for packet list merging
    7. Multi-process environment is properly configured (for real DataLoader usage)
    """
    SampleDecodeFromPacketsList()
