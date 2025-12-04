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
``accvlab.on_demand_video_decoder`` - GOP Files Decoding Sample

This sample demonstrates how to use ``accvlab.on_demand_video_decoder`` library for
efficient decoding from pre-stored GOP (Group of Pictures) files. This approach
enables high-performance video decoding by separating the packet extraction and
decoding phases, allowing for optimized storage and retrieval of video data.

Key Features Demonstrated:
- Multi-file concurrent decoding (up to configurable limit)
- Two-phase approach: GOP data storage and subsequent decoding
- Efficient packet data storage to binary files
- GPU-accelerated hardware decoding from stored GOP files
- RGB/BGR/NV12 format output options
- Device memory output for further processing
- Optimized workflow for repeated access to same video segments
"""

import os
import random
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleDecodeFromGopFiles():
    """
    Demonstrate GOP files decoding using NVIDIA's GPU-accelerated decoder.
    This function showcases a two-phase approach for optimal performance:
    1. Phase 1: Extract and store GOP packet data to binary files
    2. Phase 2: Load stored GOP files and decode to video frames

    This approach is particularly beneficial for applications that need to
    repeatedly access the same video segments, as it eliminates the need
    to re-extract packet data for each decoding operation.

    This function showcases the core functionality of accvlab.on_demand_video_decoder for GOP files:
    1. Packet extraction and storage to binary files
    2. Loading stored GOP data for efficient decoding
    3. Converting decoded frames to PyTorch tensors for ML applications
    4. Handling decoding errors gracefully with comprehensive error reporting

    The example uses a multi-camera setup from nuScenes dataset to demonstrate
    real-world usage patterns in autonomous driving applications.
    """

    # Set random seed for reproducible results
    random.seed(27)

    # Configuration: Maximum number of video files to decode simultaneously
    # This should be set based on available GPU memory and processing requirements
    max_num_files_to_use = 6

    # Frame range for random frame selection
    frame_min = 0
    frame_max = 200

    # Number of iterations to demonstrate the workflow
    num_iterations = 5

    # Sample video files from nuScenes multi-camera dataset
    # These represent synchronized camera views from autonomous vehicle sensors
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    file_list = [
        os.path.join(sample_clip_dir, "moving_shape_circle_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_ellipse_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_hexagon_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_rect_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_triangle_h265.mp4"),
    ]

    print("NVIDIA accvlab.on_demand_video_decoder - GOP Files Decoding Sample")
    print("=======================================================")
    print(f"Processing {len(file_list)} video files from multi-camera setup")
    print("Video resolution: 1600x900 pixels")
    print(f"Frame range: {frame_min} to {frame_max}")
    print(f"Number of iterations: {num_iterations}")

    # Initialize NVIDIA GPU video decoders
    print(f"\nInitializing NVIDIA GPU video decoders...")

    # Initialize first decoder for packet extraction and storage
    print("Creating packet extraction decoder...")
    nv_gop_dec1 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )

    # Initialize second decoder for GOP file decoding
    print("Creating GOP file decoder...")
    nv_gop_dec2 = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )

    print(
        f"Decoders initialized successfully on GPU 0 with support for {max_num_files_to_use} concurrent files"
    )

    # Phase 1: Extract and store GOP packet data
    print(f"\n=== Phase 1: GOP Data Storage ===")
    print("Extracting packet data and storing to binary files...")

    stored_gop_files = []
    target_frames = []

    for iteration in range(num_iterations):
        print(f"\n--- Storage Iteration {iteration + 1}/{num_iterations} ---")

        # Generate random frame indices for each video file
        frames = [random.randint(frame_min, frame_max) for _ in range(len(file_list))]
        target_frames.append(frames)
        print(f"Target frame indices: {frames}")

        try:
            # Extract packet data for each file and store to binary files
            packet_files = []
            for i in range(len(file_list)):
                print(
                    f"  Extracting packets for file {i+1}/{len(file_list)}: {os.path.basename(file_list[i])}"
                )

                # Extract packet data for single file and frame
                numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
                    file_list[i : i + 1], frames[i : i + 1]
                )

                # Create unique filename for this packet data
                packet_file = f"./gop_packets_{iteration:02d}_{i:02d}.bin"
                packet_files.append(packet_file)

                # Save packet data to binary file
                print(f"    Saving packet data to: {packet_file}")
                nvc.SavePacketsToFile(numpy_data, packet_file)

                # Verify file was created successfully
                if not os.path.exists(packet_file):
                    raise FileNotFoundError(f"Packet file not created: {packet_file}")

                file_size = os.path.getsize(packet_file)
                expected_size = numpy_data.size
                if file_size != expected_size:
                    raise ValueError(
                        f"File size mismatch for {packet_file}: expected {expected_size}, got {file_size}"
                    )

                print(f"    Packet data saved successfully: {file_size} bytes")

            stored_gop_files.append(packet_files)
            print(f"Successfully stored GOP data for iteration {iteration + 1}")

        except Exception as e:
            # Clean up any created files on error
            for packet_file in packet_files:
                if os.path.exists(packet_file):
                    os.remove(packet_file)

            print(f"GOP data storage failed in iteration {iteration + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("Possible causes:")
            print("  - Video files not accessible at specified paths")
            print("  - Frame index exceeds video length")
            print("  - Insufficient disk space for packet storage")
            print("  - Unsupported video codec or container format")
            print("  - Permission issues for file creation")
            return 1

    print(f"\nPhase 1 completed successfully. Stored {len(stored_gop_files)} GOP datasets.")

    # Phase 2: Load stored GOP files and decode
    print(f"\n=== Phase 2: GOP File Decoding ===")
    print("Loading stored GOP files and decoding to video frames...")

    for iteration in range(num_iterations):
        print(f"\n--- Decoding Iteration {iteration + 1}/{num_iterations} ---")

        packet_files = stored_gop_files[iteration]
        frames = target_frames[iteration]

        print(f"Loading GOP files for frames: {frames}")

        try:
            """
            Load stored GOP data and decode to video frames

            LoadGops Parameters:
            - packet_files: List of binary file paths containing stored packet data

            Returns:
            - Merged numpy array containing all packet data for decoding
            """
            print("Loading stored GOP data...")
            merged_numpy_data = nv_gop_dec2.LoadGops(packet_files)

            print(f"Successfully loaded GOP data: {merged_numpy_data.size} bytes")

            """
            Decode from loaded GOP data
            
            DecodeFromGOPRGB Parameters:
            - packets: Merged packet data from LoadGops
            - file_path_list: Original video file paths (for metadata)
            - frame_id_list: Target frame indices
            - as_bgr: Output format flag (True=BGR, False=RGB)
            
            Returns:
            - List of decoded frames in host memory as numpy-compatible arrays
            - Each frame maintains original video resolution and color depth
            - Frames are ready for immediate processing or tensor conversion
            """
            print("Decoding frames from GOP data...")
            decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(merged_numpy_data, file_list, frames, as_bgr=True)

            print(f"Successfully decoded {len(decoded_frames)} frames from GOP data")

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

        except Exception as e:
            print(f"GOP file decoding failed in iteration {iteration + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("Possible causes:")
            print("  - GOP files not accessible or corrupted")
            print("  - Insufficient GPU memory for concurrent decoding")
            print("  - Mismatch between stored packet data and decoder expectations")
            print("  - Unsupported video codec or container format")
            print("  - Frame index exceeds video length")
            return 1

    # Clean up stored GOP files
    print(f"\nCleaning up stored GOP files...")
    for packet_files in stored_gop_files:
        for packet_file in packet_files:
            if os.path.exists(packet_file):
                os.remove(packet_file)
                print(f"  Removed: {packet_file}")

    print(f"\nGOP files decoding completed successfully!")
    print(f"Processed {num_iterations} iterations with {len(file_list)} files each")
    return 0


if __name__ == "__main__":
    """
    Main entry point for the GOP files decoding demonstration.

    This sample demonstrates the two-phase approach for efficient video decoding:
    1. Extract and store GOP packet data to binary files
    2. Load stored GOP files and decode to video frames

    This approach is particularly beneficial for applications that need to
    repeatedly access the same video segments, as it eliminates the need
    to re-extract packet data for each decoding operation.

    Ensure that:
    1. NVIDIA GPU drivers and CUDA are properly installed
    2. accvlab.on_demand_video_decoder library is available in Python path
    3. Sample video files exist at specified paths (or update paths accordingly)
    4. PyTorch is installed for tensor conversion examples
    5. Sufficient GPU memory is available for concurrent decoding
    6. Sufficient disk space is available for GOP file storage
    7. Write permissions in the current directory for temporary GOP files
    """
    SampleDecodeFromGopFiles()
