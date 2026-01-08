# Sample Code Documentation

This document provides comprehensive guidance on using the sample codes in 
`packages/on_demand_video_decoder/samples/`. The samples demonstrate various decoding modes and advanced 
features of the `accvlab.on_demand_video_decoder` package.

## 1. Overview

The On-Demand Video Decoder package provides multiple decoding modes optimized for different use cases. This 
section helps you quickly locate the sample code that matches your requirements.

### 1.1 Sample Code Quick Reference

> **ℹ️ Note**: The sample files mentioned in the tabled below are all located in the 
> `packages/on_demand_video_decoder/samples/` directory inside the ACCV-Lab repository.

| Sample File | Use Case | Key APIs |
|------------|----------|----------|
| [SampleRandomAccess.py](../samples/SampleRandomAccess.py) | Random frame sampling for training | {py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeN12ToRGB` |
| [SampleRandomAccessWithFastInit.py](../samples/SampleRandomAccessWithFastInit.py) | Multi-clip batch processing with optimization | {py:func}`~accvlab.on_demand_video_decoder.GetFastInitInfo` |
| [SampleStreamAccess.py](../samples/SampleStreamAccess.py) | Sequential frame decoding | {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader` |
| [SampleSeparationAccess.py](../samples/SampleSeparationAccess.py) | Demuxer/decoder separation with GOP caching | {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOP`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPRGB`, {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.isCacheHit` |
| [SampleSeparationAccessGOPListAPI.py](../samples/SampleSeparationAccessGOPListAPI.py) | Per-video GOP management with caching | {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`, {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.isCacheHit` |
| [SampleDecodeFromGopFiles.py](../samples/SampleDecodeFromGopFiles.py) | GOP data persistence to disk | {py:func}`~accvlab.on_demand_video_decoder.SavePacketsToFile`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGops` |
| [SampleDecodeFromGopFilesToListAPI.py](../samples/SampleDecodeFromGopFilesToListAPI.py) | Selective GOP loading | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB` |
| [SampleDecodeFromGopList.py](../samples/SampleDecodeFromGopList.py) | Batch decode from multiple demux results (N demux → 1 decode) | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB` |
| [SampleStreamAsyncAccess.py](../samples/SampleStreamAsyncAccess.py) | Async stream decoding with prefetching | {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsync`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsyncGetBuffer` |

For details on the **Key APIs**, please refer to the API documentation of the corresponding functions and classes.

### 1.2 Choosing the Right Sample

Use this decision tree to select the appropriate sample for your use case:

```
Decoding Mode Selection:

If you need random frame access:
    If the input video resolution, color information, and other parameters remain unchanged:
        → Use SampleRandomAccessWithFastInit
    Otherwise:
        → Use SampleRandomAccess

If you need sequential frame decoding:
    If you need async decoding with prefetching for lower latency:
        → Use SampleStreamAsyncAccess
    Otherwise:
        → Use SampleStreamAccess

If you need to separate demuxing and decoding:
    If per-video GOP management is required (i.e., use of separate per-video GOP data):
        → Use SampleSeparationAccessGOPListAPI
    Otherwise:
        → Use SampleSeparationAccess

If you need to save GOP data to disk:
    → Use SampleDecodeFromGopFiles

If you need to batch decode from multiple separate demux operations:
    (e.g., DataLoader workers demux in parallel, main process batch decode)
    → Use SampleDecodeFromGopList
```

### 1.3 Core Concepts

Before diving into the samples, understanding these concepts will be helpful:

- **GOP (Group of Pictures)**: A sequence of video frames starting with a keyframe (I-frame). GOP structure is essential for video compression and random access.

- **Decoding Modes**: `accvlab.on_demand_video_decoder` supports four primary modes:
  - **Random Access**: Direct access to any frame without sequential decoding
  - **Stream Access**: Optimized for sequential frame processing with caching
  - **Separation Access**: Separate demuxing and decoding stages
  - **Demuxer-Free**: Decode directly from pre-extracted GOP data

- **FastInit**: An optimization technique that caches stream metadata to accelerate decoder initialization for multiple clips with similar properties.

- **GOP Caching**: A Python-side caching mechanism that stores extracted GOP data in memory. When the same video file is requested with a `frame_id` that falls within an already cached GOP range, the cached data is returned directly without re-demuxing from the video file.

## 2. Quick Start

This section walks you through running your first sample in 5 minutes.

### 2.1 Running Your First Sample

The simplest example is [SampleRandomAccess.py](../samples/SampleRandomAccess.py). Here's how to run it:

**Step 1: Prepare video files**

Edit the file paths in the sample code (also see the [Dataset Preparation](dataset_preparation.md) section):

```python
file_path_list = [
    "/path/to/your/video1.mp4",
    "/path/to/your/video2.mp4",
    # Add more video paths as needed
]
```

**Step 2: Run the sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleRandomAccess.py
```

**Step 3: Verify the output**

Expected output:

```text
NVIDIA accvlab.on_demand_video_decoder - Random Access Video Decoding Sample
================================================================

Initializing NVIDIA GPU video decoder...
Decoder initialized successfully on GPU 0 with support for 6 concurrent files
Processing 6 video files from multi-camera setup

--- Iteration 1/5 ---
Target frame indices: [45, 23, 78, 12, 56, 89]
Initiating GPU decoding...
Successfully decoded 6 frames
Converting frames to PyTorch tensors...
Tensor shape: torch.Size([1, 900, 1600, 3])
Tensor dtype: torch.uint8
```

### 2.2 Understanding the Basic Code Structure

All samples follow a similar structure:

```python
import accvlab.on_demand_video_decoder as nvc
import torch

# 1. Initialize decoder
decoder = nvc.CreateGopDecoder(
    maxfiles=6,  # Maximum concurrent files
    iGpu=0       # GPU device ID
)

# 2. Specify video files and frame IDs
file_path_list = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
frame_id_list = [10, 25]  # Frame ID for each video

# 3. Decode frames
decoded_frames = decoder.DecodeN12ToRGB(
    file_path_list, 
    frame_id_list, 
    as_bgr=True  # Output BGR format
)

# 4. Convert to PyTorch tensors (optional)
tensors = [torch.as_tensor(frame) for frame in decoded_frames]
```

## 3. Decoding Modes

This section provides detailed documentation for each decoding mode with corresponding sample codes.

### 3.1 Random Access Decoding

Random Access mode allows direct access to any frame in a video without sequential decoding. The decoder automatically finds the GOP containing the target frame and decodes from the nearest keyframe.

#### 3.1.1 Use Cases

- Training with random frame sampling
- Processing single video clips
- Random switching between different videos
- Non-sequential frame access patterns

#### 3.1.2 Sample: Basic Random Access

**File:** `packages/on_demand_video_decoder/samples/SampleRandomAccess.py`

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`: Initialize the GOP decoder
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeN12ToRGB`: Decode frames to RGB/BGR format

**Code Walkthrough**

Initialize the decoder:

```python
import accvlab.on_demand_video_decoder as nvc

nv_gop_dec = nvc.CreateGopDecoder(
    maxfiles=6,  # Maximum number of concurrent files
    iGpu=0       # Target GPU device ID
)
```

Prepare video files and frame indices:

```python
# Multi-camera setup from nuScenes dataset (example for sequence named `n008-2018-08-30-15-16-55-0400`)
file_path_list = [
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_BACK_LEFT.mp4",
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_BACK.mp4",
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_BACK_RIGHT.mp4",
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_FRONT_LEFT.mp4",
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_FRONT.mp4",
    "/data/nuscenes/video_samples/n008-2018-08-30-15-16-55-0400/CAM_FRONT_RIGHT.mp4",
]

# Random frame indices (one per video)
frame_id_list = [random.randint(0, 100) for _ in range(len(file_path_list))]
```

Decode frames:

```python
decoded_frames = nv_gop_dec.DecodeN12ToRGB(
    file_path_list,  # List of video file paths
    frame_id_list,   # List of target frame indices
    True             # Output in BGR format (OpenCV compatible)
)
```

Convert to PyTorch tensors:

```python
import torch

tensor_list = [torch.unsqueeze(torch.as_tensor(frame), 0) 
               for frame in decoded_frames]
```

**Performance Characteristics**

- Memory usage: Scales with concurrent file count and video resolution
- GPU utilization: 70-90% depending on video codec complexity
- Throughput: Approximately 500-1500 FPS on modern GPUs (e.g., A100)

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleRandomAccess.py
```

Note: Modify the `file_path_list` in the code to point to your video files.

#### 3.1.3 Sample: Random Access with FastInit

**File:** `packages/on_demand_video_decoder/samples/SampleRandomAccessWithFastInit.py`

**When to Use**

FastInit optimization is beneficial when:
- Processing multiple video clips from the same dataset
- All clips have similar properties (resolution, codec, GOP size)
- Initialization latency is a bottleneck
- Batch processing scenarios

**Performance Improvement**

FastInit can reduce decoder initialization time by 40-70% for subsequent clips after the first one.

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.GetFastInitInfo`: Extract stream metadata for fast initialization
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeN12ToRGB` with `fastStreamInfos` parameter

**Code Walkthrough**

Initialize decoder (one-time setup):

```python
nv_gop_dec = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

Get fast initialization info from sample files:

```python
# Extract metadata from first clip
sample_files = [os.path.join(path_bases[0], f) for f in os.listdir(path_bases[0])]
fast_stream_infos = nvc.GetFastInitInfo(sample_files)
```

> **ℹ️ Note**: {py:func}`~accvlab.on_demand_video_decoder.GetFastInitInfo` only needs to be called once for 
> clips with similar properties.

Warmup (skip first-time hardware initialization overhead):

```python
decoded_frames = nv_gop_dec.DecodeN12ToRGB(
    sample_files, 
    [0] * len(sample_files), 
    as_bgr=True,
    fastStreamInfos=fast_stream_infos
)
```

Process multiple clips with FastInit:

```python
for clip_path in clip_paths:
    file_path_list = [os.path.join(clip_path, f) for f in os.listdir(clip_path)]
    frame_id_list = [random.randint(0, 100) for _ in range(len(file_path_list))]
    
    # Use fastStreamInfos for optimized initialization
    decoded_frames = nv_gop_dec.DecodeN12ToRGB(
        file_path_list,
        frame_id_list,
        as_bgr=True,
        fastStreamInfos=fast_stream_infos  # Reuse cached stream info
    )
```

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleRandomAccessWithFastInit.py
```

### 3.2 Stream Access Decoding

Stream Access mode is optimized for sequential frame processing with intelligent caching. It is particularly useful for temporal models and sequential video analysis.

#### 3.2.1 Use Cases

- Sequential frame decoding from videos
- Temporal models (e.g., StreamPETR, BEVFormer)
- Time-series video analysis
- Scenarios where frames are accessed in order

#### 3.2.2 Sample: Stream Access

**File:** `packages/on_demand_video_decoder/samples/SampleStreamAccess.py`

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader`: Initialize the sample reader (different from 
  {py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeN12ToRGB`: Decode frames with caching 
  optimization

**Key Difference from Random Access**

Stream Access uses {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader` instead of 
{py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`. The key advantage is the use of 
caching-based optimizations. There is also the ability to iterate over individual sets of video file sets, 
each set being accessed sequentially (with the number of sets being controlled by the `num_of_set` parameter).

**Code Walkthrough**

Initialize the sample reader:

```python
nv_gop_dec = nvc.CreateSampleReader(
    num_of_set=1,              # Cache for this many video sets
    num_of_file=6,             # Maximum number of files per set
    iGpu=0
)
```

**Understanding num_of_set**

The `num_of_set` parameter controls caching behavior:
- Set to 1 for simple sequential access
- Set to `batch_size` for StreamPETR-like access patterns (iterating over the samples inside a batch, 
  accessing the same video files in every `batch_size`-th call to the decoder)

Example: If `batch_size==4`, set `num_of_set=4` to cache 4 different video clips.

Process frames sequentially:

```python
file_path_list = [
    "/data/videos/scene_CAM_BACK_LEFT.mp4",
    "/data/videos/scene_CAM_BACK.mp4",
    # ... more files
]

# Start from frame 0
frame_id_list = [0] * len(file_path_list)

for iteration in range(num_iterations):
    # Increment frame indices (sequential access)
    frame_id_list = [fid + 7 for fid in frame_id_list]
    
    decoded_frames = nv_gop_dec.DecodeN12ToRGB(
        file_path_list,
        frame_id_list,
        True
    )
```

**Caching Behavior**

Stream Access mode caches:
- Demuxer state
- Decoder state  
- Recently accessed GOPs

This reduces overhead for sequential access patterns compared to Random Access mode.

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleStreamAccess.py
```

#### 3.2.3 Sample: Async Stream Access

**File:** `packages/on_demand_video_decoder/samples/SampleStreamAsyncAccess.py`

**When to Use**

Async Stream Access is beneficial when:
- Lower latency is required for streaming applications
- Prefetching next frame while processing current frame improves latency
- Labeling task model need high-performance inference
- GPU utilization needs to be maximized through overlapped operations

**Key Advantages Over Basic Stream Access**

| Feature | Stream Access | Async Stream Access |
|---------|---------------|---------------------|
| Decode mode | Synchronous | Asynchronous with prefetching |
| Latency | Standard | Lower (prefetched frames ready) |
| GPU utilization | Standard | Better (decode/process overlap) |

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader`: Initialize the sample reader
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsync`: Start asynchronous decoding
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsyncGetBuffer`: Retrieve decoded frames from buffer

**Code Walkthrough**

Initialize the sample reader:

```python
import accvlab.on_demand_video_decoder as nvc

nv_stream_dec = nvc.CreateSampleReader(
    num_of_set=1,              # Cache for this many video sets
    num_of_file=6,             # Maximum number of files per set
    iGpu=0                     # Target GPU device ID
)
```

**Async Decoding Pattern**

The async pattern consists of two main operations:

1. **`DecodeN12ToRGBAsync`**: Start asynchronous decoding (non-blocking)
2. **`DecodeN12ToRGBAsyncGetBuffer`**: Get decoded frames (waits if not ready)

First iteration - start async decode and get result:

```python
# Start async decode
nv_stream_dec.DecodeN12ToRGBAsync(
    file_path_list,
    frame_id_list,
    False,  # Output in RGB format (False=RGB, True=BGR)
)

# Get the result (will wait for async decode to complete)
decoded_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(
    file_path_list,
    frame_id_list,
    False,  # Output in RGB format
)
```

Subsequent iterations - get prefetched result:

```python
# Get prefetched result from buffer (already decoded in background)
decoded_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(
    file_path_list,
    frame_id_list,
    False,  # Output in RGB format
)
```

**Prefetching Pattern**

The key optimization is prefetching the next frame while processing the current one:

```python
# Process current frame
tensor_list = [torch.as_tensor(frame, device='cuda') for frame in decoded_frames]
rgb_batch = torch.stack(tensor_list, dim=0)

# Prefetch next frame (non-blocking, happens in background)
if idx < len(frames_to_decode) - 1:
    next_frame = frames_to_decode[idx + 1]
    next_frame_id_list = [next_frame] * len(file_path_list)
    nv_stream_dec.DecodeN12ToRGBAsync(
        file_path_list,
        next_frame_id_list,
        False,
    )

# Continue processing current frame...
# Next iteration will get prefetched frame immediately
```

**Important: Zero-Copy Frame Management**

> **⚠️ Warning**: The decoded frames returned by `DecodeN12ToRGBAsyncGetBuffer` are zero-copy 
> references to internal buffers. You **must** deep copy the frames before calling 
> `DecodeN12ToRGBAsync` again, otherwise the data will be overwritten.

```python
# CORRECT: Deep copy frames before next async call
tensor_list = [torch.as_tensor(frame, device='cuda').clone() for frame in decoded_frames]
# or
rgb_batch = torch.stack([torch.as_tensor(frame, device='cuda') for frame in decoded_frames], dim=0)

# Now safe to call DecodeN12ToRGBAsync for next frame
nv_stream_dec.DecodeN12ToRGBAsync(...)
```

**Complete Async Workflow**

```
Iteration 1:
  DecodeN12ToRGBAsync(frame_0)     → Start decode
  DecodeN12ToRGBAsyncGetBuffer()   → Wait & get frame_0
  Process frame_0
  DecodeN12ToRGBAsync(frame_1)     → Prefetch frame_1

Iteration 2:
  DecodeN12ToRGBAsyncGetBuffer()   → Get prefetched frame_1 (fast!)
  Process frame_1
  DecodeN12ToRGBAsync(frame_2)     → Prefetch frame_2

Iteration N:
  DecodeN12ToRGBAsyncGetBuffer()   → Get prefetched frame_N
  Process frame_N
  (No prefetch for last frame)
```

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleStreamAsyncAccess.py
```

### 3.3 Separation Access Decoding

Separation Access mode decouples demuxing and decoding into two separate stages. This provides fine-grained control over the video processing pipeline and enables advanced optimization strategies.

#### 3.3.1 Use Cases

- Need separate control over demuxing and decoding
- One-time demuxing, multiple decoding operations
- Inspection or processing of intermediate packet data
- Custom processing pipelines

#### 3.3.2 Two-Stage Architecture

```
Stage 1 (Demuxing):
Video File → GetGOP() → Packet Data (GOP)
                         ├─ packets
                         ├─ first_frame_ids
                         └─ gop_lens

Stage 2 (Decoding):
Packet Data → DecodeFromGOPRGB() → Decoded Frames
```

#### 3.3.3 Sample: Basic Separation Access

**File:** `packages/on_demand_video_decoder/samples/SampleSeparationAccess.py`

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP`: Extract packet data (demuxing only)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPRGB`: Decode from packet data 
  (decoding only)

**Code Walkthrough**

Initialize two separate decoders:

```python
# Stage 1 decoder: for packet extraction
nv_gop_dec1 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# Stage 2 decoder: for packet decoding
nv_gop_dec2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

> **ℹ️ Note**: Using separate decoder instances allows independent configuration and resource management.

Stage 1 - Extract packet data:

```python
file_path_list = [
    "/data/videos/scene_CAM_BACK_LEFT.mp4",
    "/data/videos/scene_CAM_BACK.mp4",
    # ... more files
]

# Extract GOP data containing frame 77 for all videos
packets, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
    file_path_list,
    [77] * len(file_path_list)
)
```

**Understanding the return values:**
- `packets`: Compressed packet data (numpy array)
- `first_frame_ids`: First frame ID in each extracted GOP
- `gop_lens`: Number of frames in each GOP

Stage 2 - Decode from packet data:

```python
# Generate frame IDs within the GOP range
frame_id_list = [
    random.randint(first_frame_ids[i], first_frame_ids[i] + gop_lens[i] - 1)
    for i in range(len(file_path_list))
]

# Decode frames directly from packet data
decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(
    packets,           # Packet data from Stage 1
    file_path_list,    # Original file paths (for reference)
    frame_id_list,     # Target frame indices
    True               # BGR output
)
```

**Validation**

Always validate that frame IDs are within GOP range:

```python
if frame_id < first_frame_ids[i] or frame_id >= first_frame_ids[i] + gop_lens[i]:
    print(f"Frame {frame_id} is out of range for GOP starting at {first_frame_ids[i]}")
```

**Advantages of Separation**

1. Demux once, decode multiple times with different frame selections
2. Ability to inspect or process packet data
3. Separate optimization of demuxing and decoding stages
4. Foundation for more advanced processing pipelines

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleSeparationAccess.py
```

#### 3.3.4 Sample: Separation Access with GetGOPList API

**File:** `packages/on_demand_video_decoder/samples/SampleSeparationAccessGOPListAPI.py`

**When to Use**

{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOPList` is preferred over 
{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP` when:
- Processing large video collections
- Per-video cache management is needed
- Selective video loading is required
- Distributed storage and processing

**Core Difference: {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP` vs** 
**{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOPList`**

| Feature | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP` | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOPList` |
|---------|--------|------------|
| Return type | Single merged bundle | List of per-video bundles |
| Data structure | `(packets, ids, lens)` | `[(packets1, ids1, lens1), (packets2, ids2, lens2), ...]` |
| Memory management | Load all or nothing | Load selectively |
| Decoding API | DecodeFromGOPRGB | DecodeFromGOPListRGB |
| Best for | Batch processing all videos | Per-video management |

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOPList`: Extract packet data per video (not 
  merged)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Decode from list of packet 
  data

**Code Walkthrough**

Stage 1 - Extract per-video GOP data:

```python
file_path_list = [
    "/data/videos/CAM_BACK_LEFT.mp4",
    "/data/videos/CAM_BACK.mp4",
    "/data/videos/CAM_BACK_RIGHT.mp4",
    "/data/videos/CAM_FRONT_LEFT.mp4",
    "/data/videos/CAM_FRONT.mp4",
    "/data/videos/CAM_FRONT_RIGHT.mp4",
]

# Extract GOP data, returns list of tuples
gop_list = nv_gop_dec1.GetGOPList(
    file_path_list,
    [77] * len(file_path_list)
)

# gop_list structure:
# [
#   (packets_video1, first_frame_ids_video1, gop_lens_video1),
#   (packets_video2, first_frame_ids_video2, gop_lens_video2),
#   ...
# ]
```

Per-video GOP data inspection:

```python
for i, (gop_data, first_frame_ids, gop_lens) in enumerate(gop_list):
    print(f"Video {i}:")
    print(f"  GOP data size: {len(gop_data)} bytes")
    print(f"  First frame ID: {first_frame_ids[0]}")
    print(f"  GOP length: {gop_lens[0]}")
```

Simulating per-video caching:

```python
# Cache GOP data per video
gop_cache = {}
for i, (gop_data, first_frame_ids, gop_lens) in enumerate(gop_list):
    cache_key = f"video_{i}_frame_77"
    gop_cache[cache_key] = {
        'gop_data': gop_data,
        'first_frame_ids': first_frame_ids,
        'gop_lens': gop_lens,
        'filepath': file_path_list[i]
    }
```

Stage 2 - Selective decoding:

```python
# Select only specific videos to decode (e.g., front cameras only)
selected_indices = [3, 4, 5]  # Front-left, front, front-right

selected_gop_data_list = []
selected_filepaths = []
selected_frame_ids = []

for idx in selected_indices:
    cache_key = f"video_{idx}_frame_77"
    cached_item = gop_cache[cache_key]
    
    # Generate random frame within GOP range
    first_frame_id = cached_item['first_frame_ids'][0]
    gop_len = cached_item['gop_lens'][0]
    random_frame = random.randint(first_frame_id, first_frame_id + gop_len - 1)
    
    selected_gop_data_list.append(cached_item['gop_data'])
    selected_filepaths.append(cached_item['filepath'])
    selected_frame_ids.append(random_frame)

# Decode only selected videos
decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    selected_gop_data_list,  # List of GOP data for selected videos
    selected_filepaths,      # Corresponding file paths
    selected_frame_ids,      # Frame IDs to decode
    True                     # BGR output
)
```

**Key Advantages**

1. Load only required videos from cache (memory efficient)
2. Per-video cache management (independent expiration, priority)
3. Better suited for distributed systems
4. Reduced inter-video dependencies

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleSeparationAccessGOPListAPI.py
```

#### 3.3.5 GOP Caching Feature

The GOP caching feature automatically stores extracted GOP data in Python memory, eliminating the need for 
manual cache management by the user. When enabled, subsequent calls to {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOP` or {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` with the same 
video file and a `frame_id` within the cached GOP range will return cached data without re-demuxing.

**Why Use GOP Caching?**

In training scenarios, especially with video datasets:
- The same video file may be accessed multiple times with different frame indices
- Multiple frame indices often fall within the same GOP (Group of Pictures)
- Re-demuxing for each access wastes I/O and CPU resources

Without caching, users would need to manually track GOP ranges and manage cache dictionaries. With the 
`useGOPCache` parameter, this is handled automatically.

**Enabling GOP Caching**

Set `useGOPCache=True` when calling {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOP` or {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`:

```python
import accvlab.on_demand_video_decoder as nvc

decoder = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# First call - fetches GOP data from video files
packets, first_ids, gop_lens = decoder.GetGOP(
    file_path_list, 
    [77] * len(file_path_list), 
    useGOPCache=True
)

# Second call with frame_id=80 (within the same GOP range) - returns from cache
packets, first_ids, gop_lens = decoder.GetGOP(
    file_path_list, 
    [80] * len(file_path_list), 
    useGOPCache=True
)
```

**Cache Hit Condition**

A cache hit occurs when:
- The requested `filepath` matches a cached entry
- The requested `frame_id` satisfies: `first_frame_id <= frame_id < first_frame_id + gop_len`

If the `frame_id` is outside the cached GOP range, a new GOP is fetched and the cache is updated.

**Checking Cache Hit Status**

Use the {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.isCacheHit` method to check whether the last {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOP` or {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` call hit the cache:

```python
# Call GetGOP with caching
packets, first_ids, gop_lens = decoder.GetGOP(file_path_list, frame_ids, useGOPCache=True)

# Check cache hit status for each video
cache_hits = decoder.isCacheHit()
print(cache_hits)  # [True, False, True, True, False] - per-video cache hit status
```

The return value is a list of booleans, one for each video in the request, indicating whether the cached 
data was used (`True`) or new data was fetched (`False`).

**Cache Management Methods**

The decoder provides methods to manage the cache:

| Method | Description |
|--------|-------------|
| {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.get_cache_info` | Returns a dictionary with cache statistics |
| {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.clear_cache` | Clears all cached GOP data |

Example:

```python
# Get cache information
cache_info = decoder.get_cache_info()
print(f"Cached files: {cache_info['cached_files_count']}")
print(f"File paths: {cache_info['cached_files']}")

# Clear all cache when done
decoder.clear_cache()
```

**GOP Caching with GetGOPList**

The caching feature works identically with {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`:

```python
# First call - all videos are fetched
gop_list = decoder.GetGOPList(file_path_list, [77, 77, 77], useGOPCache=True)
print(decoder.isCacheHit())  # [False, False, False]

# Second call with some frame_ids in range, some out of range
gop_list = decoder.GetGOPList(file_path_list, [80, 80, 150], useGOPCache=True)
print(decoder.isCacheHit())  # [True, True, False] - partial cache hit
```

**Shared Cache Between GetGOP and GetGOPList**

The cache is shared between {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOP` and {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` calls on the same decoder instance:

```python
# Cache populated via GetGOP
packets, _, _ = decoder.GetGOP(["/path/to/video.mp4"], [50], useGOPCache=True)

# Cache hit via GetGOPList (same file, frame_id in range)
gop_list = decoder.GetGOPList(["/path/to/video.mp4"], [55], useGOPCache=True)
print(decoder.isCacheHit())  # [True]
```

> **⚠️ Note**: The cache is stored in Python memory. Each video file caches only one GOP (the most 
> recently accessed). For long-running processes with many different videos, use {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.clear_cache` to 
> release memory when needed.

**When to Use GOP Caching**

✓ Training loops with random frame sampling from the same video
✓ Multi-camera setups where cameras are often accessed with similar frame indices
✓ Scenarios where the same GOP is likely to be accessed multiple times
✓ Reducing I/O overhead in data loading pipelines

✗ One-time video processing (no repeated access)
✗ Memory-constrained environments with large video collections
✗ Scenarios where each frame access targets a different GOP

### 3.4 Demuxer-Free Decoding

Demuxer-Free mode allows decoding directly from pre-extracted GOP data, either stored on disk or in memory. This approach is ideal for scenarios requiring repeated access to the same video segments.

#### 3.4.1 Use Cases

- Pre-processing video datasets for training
- Repeated access to same video segments
- Disk storage for GOP data caching
- Eliminating demuxing overhead in production
- PyTorch DataLoader integration with worker processes

#### 3.4.2 Sample: GOP File Storage and Decoding

**File:** `packages/on_demand_video_decoder/samples/SampleDecodeFromGopFiles.py`

**Two-Phase Workflow**

```
Phase 1: GOP Data Preparation
Video Files → GetGOP() → SavePacketsToFile() → .bin files on disk

Phase 2: Decoding from Files
.bin files → LoadGops() → DecodeFromGOPRGB() → Decoded Frames
```

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP`: Extract GOP packet data
- {py:func}`~accvlab.on_demand_video_decoder.SavePacketsToFile`: Save packets to binary file
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGops`: Load packets from binary files (merged)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPRGB`: Decode from loaded packets

**Code Walkthrough**

Initialize decoders:

```python
# Decoder for packet extraction
nv_gop_dec1 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# Decoder for GOP file decoding
nv_gop_dec2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

Phase 1 - Extract and save GOP data:

```python
file_list = [
    "/data/videos/CAM_BACK_LEFT.mp4",
    "/data/videos/CAM_BACK.mp4",
    # ... more files
]

frames = [random.randint(0, 200) for _ in range(len(file_list))]
packet_files = []

for i in range(len(file_list)):
    # Extract packet data for single file
    numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
        file_list[i:i+1],
        frames[i:i+1]
    )
    
    # Save to binary file
    packet_file = f"./gop_packets_{i:02d}.bin"
    nvc.SavePacketsToFile(numpy_data, packet_file)
    packet_files.append(packet_file)
    
    print(f"Saved GOP data: {os.path.getsize(packet_file)} bytes")
```

Phase 2 - Load and decode from GOP files:

```python
# Load stored GOP data
merged_numpy_data = nv_gop_dec2.LoadGops(packet_files)

print(f"Loaded GOP data: {merged_numpy_data.size} bytes")

# Decode frames from loaded data
decoded_frames = nv_gop_dec2.DecodeFromGOPRGB(
    merged_numpy_data,  # Merged packet data from LoadGops
    file_list,          # Original video file paths
    frames,             # Target frame indices
    as_bgr=True
)
```

Cleanup temporary files:

```python
for packet_file in packet_files:
    if os.path.exists(packet_file):
        os.remove(packet_file)
```

**File Format**

GOP files are binary files containing raw packet data. The format is:
- Binary format (no header)
- Direct memory dump of packet data
- File extension: `.bin` (recommended)

**Storage Considerations**

- GOP file size: Typically 5-15% of original video size
- Storage savings: ~85-95% compared to extracted frames
- I/O performance: SSD recommended for best performance

**When to Use**

Use GOP file storage when:
- Same video segments accessed repeatedly
- Training multiple epochs on the same dataset
- Storage is cheaper than compute
- Want to eliminate demuxing overhead

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleDecodeFromGopFiles.py
```

#### 3.4.3 Sample: GOP File List API

**File:** `packages/on_demand_video_decoder/samples/SampleDecodeFromGopFilesToListAPI.py`

**When to Use**

{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList` is preferred over 
{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGops` when:
- Large video collections (>10 videos)
- Need selective loading of specific videos
- Per-video cache management
- Distributed caching systems

**Core Difference: {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGops` vs** 
**{py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`**

| Feature | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGops` | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList` |
|---------|----------|----------------|
| Return type | Single merged numpy array | List of numpy arrays (one per video) |
| Loading | All or nothing | Selective loading possible |
| Memory usage | Load all GOP data at once | Load only needed videos |
| Decoding API | DecodeFromGOPRGB | DecodeFromGOPListRGB |
| Best for | Small video sets | Large video collections |

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.SavePacketsToFile`: Save per-video GOP data
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`: Load GOP files as list (not 
  merged)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Decode from list of GOP 
  data

**Code Walkthrough**

Phase 1 - Save per-video GOP files:

```python
file_list = [
    "/data/videos/CAM_BACK_LEFT.mp4",
    "/data/videos/CAM_BACK.mp4",
    "/data/videos/CAM_BACK_RIGHT.mp4",
    "/data/videos/CAM_FRONT_LEFT.mp4",
    "/data/videos/CAM_FRONT.mp4",
    "/data/videos/CAM_FRONT_RIGHT.mp4",
]

camera_names = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
                "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

packet_files = []
frames = [random.randint(0, 200) for _ in range(len(file_list))]

for i in range(len(file_list)):
    # Extract GOP data for single video
    numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
        file_list[i:i+1],
        frames[i:i+1]
    )
    
    # Create unique filename per video
    packet_file = f"./gop_{camera_names[i]}.bin"
    nvc.SavePacketsToFile(numpy_data, packet_file)
    packet_files.append(packet_file)
```

Phase 2 - Load all GOP files as list:

```python
# Load GOP files as separate bundles (not merged)
gop_data_list = nv_gop_dec2.LoadGopsToList(packet_files)

# gop_data_list is a list of numpy arrays, one per video
print(f"Loaded {len(gop_data_list)} GOP bundles")
for i, gop_data in enumerate(gop_data_list):
    print(f"  Bundle {i} ({camera_names[i]}): {len(gop_data)} bytes")
```

Decode from GOP list:

```python
# Decode all videos
decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    gop_data_list,  # List of GOP data
    file_list,      # List of file paths
    frames,         # List of frame IDs
    as_bgr=True
)
```

Phase 3 - Selective loading demonstration:

```python
# Select only front cameras (indices 3, 4, 5)
selected_indices = [3, 4, 5]
selected_files = [packet_files[i] for i in selected_indices]
selected_video_paths = [file_list[i] for i in selected_indices]
selected_frames = [frames[i] for i in selected_indices]

# Load only selected GOP files
selected_gop_list = nv_gop_dec2.LoadGopsToList(selected_files)

# Decode only selected videos
decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    selected_gop_list,
    selected_video_paths,
    selected_frames,
    as_bgr=True
)

print(f"Loaded and decoded only {len(selected_indices)} out of {len(packet_files)} videos")
```

**Key Advantages**

1. Memory efficiency: Load only needed videos
2. Flexible loading: Different subsets for different batches
3. Distributed caching: Store videos on different machines
4. Per-video cache management: Independent expiration policies

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleDecodeFromGopFilesToListAPI.py
```

#### 3.4.4 Sample: Batch Decode from Multiple Demux Results

**File:** `packages/on_demand_video_decoder/samples/SampleDecodeFromGopList.py`

**When to Use**

This sample demonstrates the pattern of multiple demuxing operations followed by a single batch decode:
- Demux executed N times separately (e.g., in DataLoader `__getitem__`, called batch_size times)
- Decode executed once for the entire batch
- Enables parallel demuxing in worker processes, centralized batch decoding in main process
- No disk I/O for GOP data (in-memory packet passing)

**Architecture: N Demux → 1 Batch Decode**

```
Worker/Process 1: Video File 1 → GetGOP() → packets_1 (in memory)
Worker/Process 2: Video File 2 → GetGOP() → packets_2 (in memory)
Worker/Process 3: Video File 3 → GetGOP() → packets_3 (in memory)
                     ⋮                            ⋮
Worker/Process N: Video File N → GetGOP() → packets_N (in memory)
                                                      ↓
                          Collect all packets: [packets_1, packets_2, ..., packets_N]
                                                      ↓
                  Main Process: DecodeFromGOPListRGB() → Batch of N Frames (single decode call)
```

**Core Concept**

Multiple separate demuxing operations → Single batch decoding operation

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.GetGOP`: Extract packets (called N times, 
  possibly in parallel)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Batch decode from list of 
  packets (called once for entire batch)

**Code Walkthrough**

Initialize decoders:

```python
# Worker decoder (simulated): for packet extraction
nv_gop_dec1 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# Main process decoder: for batch decoding
nv_gop_dec2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

Phase 1 - Multiple demux operations (simulating parallel workers):

```python
file_list = [
    "/data/videos/CAM_BACK_LEFT.mp4",
    "/data/videos/CAM_BACK.mp4",
    # ... more files
]

frames = [random.randint(0, 200) for _ in range(len(file_list))]

# Demux executed N times (e.g., in DataLoader __getitem__, called batch_size times)
packets_list = []

for i in range(len(file_list)):
    # Each demux operation extracts packets for one video
    numpy_data, first_frame_ids, gop_lens = nv_gop_dec1.GetGOP(
        file_list[i:i+1],
        frames[i:i+1]
    )
    packets_list.append(numpy_data)
    print(f"Demux {i+1}: Extracted {numpy_data.size} bytes")
```

Phase 2 - Single batch decode (in main process):

```python
# Decode executed once for all N demux results
decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    packets_list,  # List of N packet data from multiple demux operations
    file_list,     # Original file paths
    frames,        # Target frame IDs
    as_bgr=True
)

print(f"Batch decode: {len(decoded_frames)} frames decoded in one call")
```

**DataLoader Integration Pattern**

In a real PyTorch DataLoader:

```python
# In worker process (worker_fn)
def worker_fn(video_path, frame_id):
    packets, first_ids, gop_lens = decoder.GetGOP([video_path], [frame_id])
    return packets

# In main process collate_fn
def collate_fn(batch):
    packets_list = [item['packets'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    
    # Batch decode in main process
    frames = decoder.DecodeFromGOPListRGB(packets_list, file_paths, frame_ids, True)
    return frames
```

**Key Benefits**

1. **Parallel demuxing**: Each worker demuxes independently in parallel
2. **Single batch decode**: GPU decoder called only once for entire batch (efficient GPU utilization)
3. **No disk I/O**: Packets passed in memory, no temporary file storage
4. **Resource separation**: CPU-heavy demuxing in workers, GPU decoding in main process

**Memory Management**

- Keep packet data lifetime short (decode and release)
- Monitor memory usage in worker processes
- Balance worker count with available memory

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleDecodeFromGopList.py
```
