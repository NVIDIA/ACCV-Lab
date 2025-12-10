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
``accvlab.on_demand_video_decoder`` - Async Stream Access Video Decoding Sample

This sample demonstrates how to use ``accvlab.on_demand_video_decoder`` library for
efficient async stream access video decoding with GPU acceleration. The decoder
enables high-performance asynchronous decoding with prefetching capability,
allowing frame N+1 to be decoded while processing frame N.

Key Features Demonstrated:
- Asynchronous decoding with prefetching for improved throughput
- Multi-file concurrent decoding (up to configurable limit)
- Sequential frame access optimized for streaming applications
- GPU-accelerated hardware decoding
- RGB/BGR format output options
- Device memory output for further GPU processing
- Efficient caching for stream-based applications like StreamPETR
"""

import os
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleStreamAsyncAccess():
    """
    Demonstrate async stream access video decoding using NVIDIA's GPU-accelerated decoder.
    This function is optimized for asynchronous frame access with prefetching capability,
    enabling higher throughput by overlapping decode and processing operations.

    This function showcases the core async functionality of accvlab.on_demand_video_decoder:
    1. Initializing the SampleReader with optimized caching for stream-based access
    2. Using DecodeN12ToRGBAsync to start asynchronous decoding
    3. Using DecodeN12ToRGBAsyncGetBuffer to retrieve decoded frames
    4. Prefetching next frame while processing current frame
    5. Converting decoded frames to PyTorch tensors for ML applications
    6. Handling decoding errors gracefully with comprehensive error reporting

    The example uses sample video files to demonstrate real-world usage patterns
    in autonomous driving and video streaming applications.
    """

    # Configuration: Maximum number of video files to decode simultaneously
    # This should be set based on available GPU memory and processing requirements
    max_num_files_to_use = 6

    # Initialize the SampleReader with optimized settings for stream-based decoding
    print("Initializing NVIDIA GPU video decoder for async stream access...")
    nv_stream_dec = nvc.CreateSampleReader(
        # Cache number_of_set videos status in decoder, You can set it to batchsize for
        # stream_petr-like access (i.e. iterating over the individual samples of a batch, so that the current
        # video files are accessed once per batch, i.e. every batch_size-th time the decoder is called).
        num_of_set=1,
        num_of_file=max_num_files_to_use,  # Maximum number of files to use
        iGpu=0,  # Target GPU device ID (0 for primary GPU)
    )
    print(
        f"Decoder initialized successfully on GPU 0 with support for {max_num_files_to_use} concurrent files"
    )
    print("Async stream access mode enabled with prefetching capability")

    # Sample video files
    # The list of decoded videos must have a number of videos that is less than or equal to max_num_files_to_use
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

    # Generate frame list to decode: 0, 7, 14, 21, 28 (every 7th frame)
    num_iterations = 5
    frames_to_decode = [i * 7 for i in range(num_iterations)]

    print(f"\nStarting {num_iterations} async stream access decoding iterations...")
    print("Using async decoding with prefetching: while processing frame N, prefetch frame N+1")
    print(f"Frames to decode: {frames_to_decode}")

    for idx, target_frame in enumerate(frames_to_decode):
        print(f"\n--- Iteration {idx + 1}/{num_iterations} (Frame {target_frame}) ---")

        # Create frame id list (same frame index for all video files)
        frame_id_list = [target_frame] * len(file_path_list)
        print(f"Target frame indices: {frame_id_list}")

        try:
            """
            Async Stream Access Pattern:

            DecodeN12ToRGBAsync: Start asynchronous decoding
            - Initiates GPU decoding without blocking
            - Returns immediately while decoding happens in background

            DecodeN12ToRGBAsyncGetBuffer: Get decoded frames from buffer
            - Waits for async decode to complete if not ready
            - Returns decoded frames from internal buffer
            - Frames are in GPU memory for efficient further processing
            """

            if idx == 0:
                # First iteration: start async decode, then get result immediately
                print("[Async] Starting async decode for first frame...")
                nv_stream_dec.DecodeN12ToRGBAsync(
                    file_path_list,
                    frame_id_list,
                    False,  # Output in RGB format (False=RGB, True=BGR)
                )

                # Get the result (will wait for async decode to complete)
                print("[Async] Getting decoded frames from buffer...")
                decoded_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(
                    file_path_list,
                    frame_id_list,
                    False,  # Output in RGB format
                )
                print(f"[Async] Frame {target_frame} decoded (initial, non-prefetched)")
            else:
                # Subsequent iterations: get prefetched result from buffer
                print(f"[Async] Getting prefetched result for frame {target_frame}...")
                decoded_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(
                    file_path_list,
                    frame_id_list,
                    False,  # Output in RGB format
                )
                print(f"[Async] Frame {target_frame} retrieved from prefetch buffer")

            print(f"Successfully decoded {len(decoded_frames)} frames")

            # Convert decoded frames to PyTorch tensors for ML applications
            # Frames are already in GPU memory, so conversion is efficient
            # INMPORTANT: You must deepcopy the frames: using torch.clone(), torch.stack() or other deepcopy operations
            # Then call DecodeN12ToRGBAsync again.
            # Otherwise, the zero-copy decoded frames will be recorvered by the next DecodeN12ToRGBAsync call.
            print("Converting frames to PyTorch tensors (GPU memory)...")
            tensor_list = [torch.as_tensor(frame, device='cuda') for frame in decoded_frames]

            # Stack tensors for batch processing
            rgb_batch = torch.stack(tensor_list, dim=0)  # Shape: [N, H, W, 3]

            # Prefetch next frame (if not the last frame)
            # This is the key optimization: start decoding next frame while processing current one
            if idx < len(frames_to_decode) - 1:
                next_frame = frames_to_decode[idx + 1]
                next_frame_id_list = [next_frame] * len(file_path_list)
                print(f"[Async] Prefetching next frame {next_frame} while processing frame {target_frame}...")
                try:
                    nv_stream_dec.DecodeN12ToRGBAsync(
                        file_path_list,
                        next_frame_id_list,
                        False,  # Output in RGB format
                    )
                    print(f"[Async] Prefetch for frame {next_frame} initiated")
                except Exception as e:
                    print(f"Warning: Failed to prefetch frame {next_frame}: {e}")

            # Display tensor information for first frame (representative of all frames)
            first_tensor = tensor_list[0]
            print(f"Tensor shape: {first_tensor.shape}")  # Expected: [height, width, channels]
            print(f"Tensor dtype: {first_tensor.dtype}")  # Typically uint8 for image data
            print(f"Tensor device: {first_tensor.device}")  # cuda:0 (GPU memory)
            print(f"Batch shape: {rgb_batch.shape}")  # Expected: [N, height, width, channels]
            print(f"Value range: [{first_tensor.min().item()}, {first_tensor.max().item()}]")

            # Optional: Display frame statistics for debugging/validation
            print(f"Frame dimensions: {first_tensor.shape[0]}x{first_tensor.shape[1]} (HxW)")
            print(
                f"Color channels: {first_tensor.shape[2]} ({'RGB' if first_tensor.shape[2] == 3 else 'Unknown'})"
            )

            # Simulate processing time (in real applications, this would be inference, etc.)
            # During this time, the next frame is being decoded in the background
            print("[Processing] Simulating frame processing...")
            # In a real application, you would do ML inference here
            # The prefetched frame is being decoded in parallel

        except Exception as e:
            # Comprehensive error handling for production robustness
            # Common issues: file not found, unsupported codec, insufficient GPU memory
            print(f"Async stream decoding failed in iteration {idx + 1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            print("Possible causes:")
            print("  - Video files not accessible at specified paths")
            print("  - Frame index exceeds video length")
            print("  - Insufficient GPU memory for concurrent decoding")
            print("  - Unsupported video codec or container format")
            print("  - Async buffer not ready or corrupted")
            print("Continuing with next iteration...\n")
            continue

    print("\n" + "=" * 60)
    print("Async stream access decoding completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    """
    Main entry point for the async stream access video decoding demonstration.

    This sample can be run directly to see accvlab.on_demand_video_decoder async mode in action.
    Ensure that:
    1. NVIDIA GPU drivers and CUDA are properly installed
    2. accvlab.on_demand_video_decoder library is available in Python path
    3. Sample video files exist at specified paths (or update paths accordingly)
    4. PyTorch is installed for tensor conversion examples
    5. Sufficient GPU memory is available for concurrent async decoding

    Key advantages of async mode:
    - Higher throughput through overlapped decode/process operations
    - Reduced latency for streaming applications
    - Better GPU utilization with prefetching
    """
    print("NVIDIA accvlab.on_demand_video_decoder - Async Stream Access Video Decoding Sample")
    print("=================================================================================")
    print()

    SampleStreamAsyncAccess()
