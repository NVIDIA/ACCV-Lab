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
``accvlab.on_demand_video_decoder`` - Random Access Video Decoding Sample

This sample demonstrates how to use ``accvlab.on_demand_video_decoder`` library for
efficient random access video decoding with GPU acceleration. The decoder
enables high-performance decoding of specific frames from multiple video files
simultaneously, making it ideal for applications requiring non-sequential
frame access.

Key Features Demonstrated:
- Multi-file concurrent decoding (up to configurable limit)
- Random frame access without sequential decoding overhead
- GPU-accelerated hardware decoding
- RGB/BGR/NV12 format output options
- Device memory output for further processing
"""

import os
import random
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleRandomAccess():
    """
    Demonstrate random access video decoding using NVIDIA's GPU-accelerated decoder.
    It is possible to randomly switch frames or select different videos with high performance.

    This function showcases the core functionality of accvlab.on_demand_video_decoder:
    1. Initializing the GOP (Group of Pictures) decoder with multi-file support
    2. Performing random frame access across multiple video streams
    3. Converting decoded frames to PyTorch tensors for ML applications
    4. Handling decoding errors gracefully

    The example uses a multi-camera setup from nuScenes dataset to demonstrate
    real-world usage patterns in autonomous driving applications.
    """

    # Configuration: Maximum number of video files to decode simultaneously
    # This should be set based on available GPU memory and processing requirements
    max_num_files_to_use = 6

    # Initialize the GOP decoder with optimized settings for multi-file decoding
    print("Initializing NVIDIA GPU video decoder...")
    nv_gop_dec = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files (should match or exceed file list length)
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )
    print(
        f"Decoder initialized successfully on GPU 0 with support for {max_num_files_to_use} concurrent files"
    )

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

    print(f"Processing {len(file_path_list)} video files from multi-camera setup")
    print("Video resolution: 1600x900 pixels")

    # Perform multiple random access decoding iterations to demonstrate performance
    num_iterations = 5
    max_frame_index = 100  # Assuming videos have at least 100 frames

    print(f"\nStarting {num_iterations} random access decoding iterations...")

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Generate random frame indices for each video file
        # This simulates real-world scenarios where different frames are needed from each camera
        frame_id_list = [random.randint(0, max_frame_index) for _ in range(len(file_path_list))]
        print(f"Target frame indices: {frame_id_list}")

        try:
            """
            Perform GPU-accelerated random access decoding

            DecodeN12ToRGB Parameters:
            - filepaths: List of video file paths to decode from
            - frame_ids: List of specific frame indices to extract (one per file)
            - as_bgr: Output format flag (True=BGR, False=RGB)

            Returns:
            - List of decoded frames in host memory as numpy-compatible arrays
            - Each frame maintains original video resolution and color depth
            - Frames are ready for immediate processing or tensor conversion
            """
            print("Initiating GPU decoding...")
            decoded_frames = nv_gop_dec.DecodeN12ToRGB(
                file_path_list,  # Input video files
                frame_id_list,  # Target frame indices
                True,  # Output in BGR format (OpenCV compatible)
            )

            print(f"Successfully decoded {len(decoded_frames)} frames")

            # Convert decoded frames to PyTorch tensors for ML applications
            # This demonstrates integration with deep learning workflows
            print("Converting frames to PyTorch tensors...")
            tensor_list = [torch.unsqueeze(torch.as_tensor(frame), 0) for frame in decoded_frames]

            # Display tensor information for first frame (representative of all frames)
            first_tensor = tensor_list[0]
            print(f"Tensor shape: {first_tensor.shape}")  # Expected: [1, height, width, channels]
            print(f"Tensor dtype: {first_tensor.dtype}")  # Typically uint8 for image data
            print(f"Tensor device: {first_tensor.device}")  # CPU (host memory)
            print(f"Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")

            # Optional: Display frame statistics for debugging/validation
            print(f"Frame dimensions: {first_tensor.shape[1]}x{first_tensor.shape[2]} (HxW)")
            print(
                f"Color channels: {first_tensor.shape[3]} ({'BGR' if first_tensor.shape[3] == 3 else 'Unknown'})"
            )

        except Exception as e:
            # Comprehensive error handling for production robustness
            # Common issues: file not found, unsupported codec, insufficient GPU memory
            print(f"Decoding failed in iteration {iteration + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("Possible causes:")
            print("  - Video files not accessible at specified paths")
            print("  - Frame index exceeds video length")
            print("  - Insufficient GPU memory for concurrent decoding")
            print("  - Unsupported video codec or container format")
            print("Continuing with next iteration...\n")
            exit(-1)


if __name__ == "__main__":
    """
    Main entry point for the random access video decoding demonstration.

    This sample can be run directly to see accvlab.on_demand_video_decoder in action.
    Ensure that:
    1. NVIDIA GPU drivers and CUDA are properly installed
    2. accvlab.on_demand_video_decoder library is available in Python path
    3. Sample video files exist at specified paths (or update paths accordingly)
    4. PyTorch is installed for tensor conversion examples
    """
    print("NVIDIA accvlab.on_demand_video_decoder - Random Access Video Decoding Sample")
    print("================================================================")
    print()

    SampleRandomAccess()
