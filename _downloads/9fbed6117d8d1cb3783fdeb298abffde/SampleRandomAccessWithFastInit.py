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
``accvlab.on_demand_video_decoder`` - Random Access Video Decoding Sample with FastInit

This sample demonstrates how to use ``accvlab.on_demand_video_decoder`` library for
efficient random access video decoding with GPU acceleration. The decoder
enables high-performance decoding of specific frames from multiple video files
simultaneously, making it ideal for applications requiring non-sequential
frame access. This sample teaches how to use FastInit to accelerate the demuxing
process.

Key Features Demonstrated:
- Multi-file concurrent decoding (up to configurable limit)
- Random frame access without sequential decoding overhead
- GPU-accelerated hardware decoding
- RGB/BGR/NV12 format output options
- Device memory output for further processing
- FastInit to accelerate the demuxing process
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
    2. Getting fast initialization information for the first clip from one clip
    3. Performing random frame access across multiple video streams with fast initialization information
    4. Converting decoded frames to PyTorch tensors for ML applications
    5. Handling decoding errors gracefully

    The example uses a multi-camera setup from nuScenes dataset to demonstrate
    real-world usage patterns in autonomous driving applications.
    """

    # Configuration: Set to accommodate maximum number of cameras in the setup
    # For automotive applications, this typically matches the number of vehicle-mounted cameras
    max_num_files_to_use = 6  # Optimized for 6-camera automotive configuration

    # Multi-clip dataset configuration for automotive video processing
    # Each path represents a different time segment or driving scenario
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    path_bases = [
        sample_clip_dir,
    ]

    # Initialize the GOP decoder
    # Just do once, no need to do it for each clip
    nv_gop_dec = nvc.CreateGopDecoder(
        maxfiles=max_num_files_to_use,  # Maximum concurrent files (matches camera count)
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )

    # Get fast initialization information for the first clip
    # Just do once, no need to do it for each clip
    sample_files = [os.path.join(path_bases[0], file_name) for file_name in os.listdir(path_bases[0])]
    fast_stream_infos = nvc.GetFastInitInfo(sample_files)

    # Warmup, skip the first time hardware initialization overhead
    decoded_frames = nv_gop_dec.DecodeN12ToRGB(
        sample_files, [0] * len(os.listdir(path_bases[0])), as_bgr=True, fastStreamInfos=fast_stream_infos
    )
    tensor_list = [torch.unsqueeze(torch.as_tensor(frame), 0) for frame in decoded_frames]

    '''
    Main processing loop: Decode random frames from multiple video clips
    Each iteration processes a complete multi-camera clip using cached stream information
    FastInit eliminates decoder setup overhead, enabling high-throughput processing
    '''
    max_frame_index = 100  # Assuming clips have at least 100 frames

    for i, path_base in enumerate(path_bases):
        print(f"\n--- Processing Clip {i+1}/{len(path_bases)}: {os.path.basename(path_base)} ---")

        file_path_list = [os.path.join(path_base, file_name) for file_name in os.listdir(path_base)]
        frame_id_list = [random.randint(0, max_frame_index) for _ in range(len(file_path_list))]

        print(f"Target frame indices: {frame_id_list}")
        print(f"Processing {len(file_path_list)} synchronized camera streams...")

        # Display file paths for transparency (helpful for debugging)
        for j, file_path in enumerate(file_path_list):
            print(f"  Camera {j+1}: {file_path}")

        try:
            decoded_frames = nv_gop_dec.DecodeN12ToRGB(
                file_path_list,  # Current clip's camera files
                frame_id_list,  # Target frame indices
                as_bgr=True,  # BGR format output (OpenCV compatible)
                fastStreamInfos=fast_stream_infos,  # Cached stream optimization data
            )
            tensor_list = [torch.unsqueeze(torch.as_tensor(frame), 0) for frame in decoded_frames]

            # Display detailed tensor information for the first camera frame
            first_tensor = tensor_list[0]
            print(f"Tensor specifications:")
            print(f"  Shape: {first_tensor.shape}")  # Expected: [1, height, width, channels]
            print(f"  Data type: {first_tensor.dtype}")  # Typically uint8 for image data
            print(f"  Memory location: {first_tensor.device}")  # CPU (host memory)
            print(f"  Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")
            print(f"  Frame dimensions: {first_tensor.shape[1]}x{first_tensor.shape[2]} (HxW)")
            print(
                f"  Color format: {'BGR' if first_tensor.shape[3] == 3 else 'Unknown'} ({first_tensor.shape[3]} channels)"
            )

        except Exception as e:
            print(f"Error: {e}")
            exit(-1)


if __name__ == "__main__":
    """
    Main entry point for the FastInit random access video decoding demonstration.

    This sample showcases advanced video processing capabilities with significant
    performance optimizations for batch processing scenarios.

    Prerequisites:
    1. NVIDIA GPU with hardware video decoding support
    2. CUDA drivers and runtime properly installed
    3. accvlab.on_demand_video_decoder library with FastInit support
    4. Video dataset with consistent multi-camera structure
    5. PyTorch for tensor conversion demonstrations

    Performance Expectations:
    - Initial stream analysis: ~100-500ms (one-time overhead)
    - Subsequent clip processing: 40-70% faster than standard decoding
    - Scalability: Linear performance scaling with additional clips
    """
    print("NVIDIA accvlab.on_demand_video_decoder - FastInit Random Access Video Decoding")
    print("===================================================================")
    print("Advanced sample demonstrating optimized batch video processing")
    print("with Fast Initialization for enhanced performance\n")

    SampleRandomAccess()
