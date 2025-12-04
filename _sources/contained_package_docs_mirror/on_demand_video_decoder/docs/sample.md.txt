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
| [SampleDemuxerDecoderSeparationAccess.py](../samples/SampleDemuxerDecoderSeparationAccess.py) | Demuxer/decoder separation with GOP caching | {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`, {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.isCacheHit` |
| [SampleDecodeFromGopFiles.py](../samples/SampleDecodeFromGopFiles.py) | GOP file persistence with selective loading | {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB` |
| [SampleStreamAsyncAccess.py](../samples/SampleStreamAsyncAccess.py) | Async stream decoding with prefetching | {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsync`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsyncGetBuffer` |
| [SampleBatchAsyncStreamAccess.py](../samples/SampleBatchAsyncStreamAccess.py) | 2D async stream decoding — multiple frames per video per call, with prefetching | {py:func}`~accvlab.on_demand_video_decoder.CreateBatchAsyncStreamReader`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.Decode`, {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.GetBuffer` |
| [SampleSharedGopStore.py](../samples/SampleSharedGopStore.py) | Cross-process shared GOP cache for DataLoader | {py:class}`~accvlab.on_demand_video_decoder.SharedGopStore`, {py:class}`~accvlab.on_demand_video_decoder.GopRef` |

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
    If you need multiple frames per video per call (2D batch):
        → Use SampleBatchAsyncStreamAccess
    Else if you need async decoding with prefetching for lower latency:
        → Use SampleStreamAsyncAccess
    Otherwise:
        → Use SampleStreamAccess

If you need to separate demuxing and decoding:
    (e.g., DataLoader workers demux on CPU, main process batch-decodes on GPU)
    → Use SampleDemuxerDecoderSeparationAccess

If you need to save serialized GOP bundles to disk:
    → Use SampleDecodeFromGopFiles

If you need cross-process shared GOP caching for DataLoader workers:
    (e.g., workers demux GOPs into shared memory, main process reads zero-copy)
    → Use SampleSharedGopStore
```

### 1.3 Core Concepts

Before diving into the samples, understanding these concepts will be helpful:

- **GOP (Group of Pictures)**: A sequence of video frames starting with a keyframe (I-frame). GOP structure is essential for video compression and random access.

- **Decoding Modes**: `accvlab.on_demand_video_decoder` supports four primary modes:
  - **Random Access**: Direct access to any frame without sequential decoding
  - **Stream Access**: Optimized for sequential frame processing with caching
  - **Separation Access**: Separate demuxing and decoding stages
  - **Demuxer-Free**: Decode directly from pre-extracted serialized GOP bundles

- **FastInit**: An optimization technique that caches stream metadata to accelerate decoder initialization for multiple clips with similar properties.

- **Serialized GOP Bundle**: The numpy object returned by {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` — one video's GOP, serialized by `accvlab` into a self-contained binary format. Treat it as an opaque blob: decode it with {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`, or persist/reload it with {py:func}`~accvlab.on_demand_video_decoder.SaveGopToFile` / {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`.

- **GOP Caching**: A Python-side caching mechanism that stores extracted serialized GOP bundles in memory. When the same video file is requested with a `frame_id` that falls within an already cached GOP range, the cached bundle is returned directly without re-demuxing from the video file.

- **SharedGopStore**: A cross-process shared memory cache for serialized GOP bundles, backed by POSIX SharedMemory (`/dev/shm`). Workers store bundles in shared memory and pass lightweight `GopRef` references through the DataLoader IPC queue. The main process reads the data as zero-copy numpy views via `get_batch()`. Uses file-based locking (`flock`) for cross-process safety and LRU eviction when capacity is exceeded.

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

- {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader`: Create the stream-access decoder (different from 
  {py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGB`: Decode frames with caching 
  optimization

**Key Difference from Random Access**

Stream Access uses {py:func}`~accvlab.on_demand_video_decoder.CreateSampleReader` instead of 
{py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`. The key advantage is the use of 
caching-based optimizations. A *sample* is one set of video files that are read together (e.g. the 
synchronized multi-camera clips of one training sample); the reader keeps up to `num_of_set` samples 
cached, each accessed sequentially.

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

#### 3.2.4 Sample: Batch Async Stream Access (2D)

**File:** `packages/on_demand_video_decoder/samples/SampleBatchAsyncStreamAccess.py`

**When to Use**

The 2D batch async API is preferred over basic async stream access when:
- Each iteration consumes **multiple frames per video** (e.g. multi-sweep
  StreamPETR-like training where one batch needs F sweeps × V cameras;
  multimodal LLM and robotics timeline workloads that read multiple frames
  per video)
- You want to retrieve V × F frames via a **single async submission** — a
  capability the 1D async API cannot provide, because its in-flight buffer
  holds only one result per reader, so issuing F sequential 1D async calls
  in Python does not give you F frames decoding in parallel

The 1D async API ({py:meth}`~accvlab.on_demand_video_decoder.PyNvSampleReader.DecodeN12ToRGBAsync`)
remains the right choice when you only need one frame per video per
iteration.

**Key Differences from 1D Async Stream Access**

The ``filepaths`` argument is the **same** for both APIs — a flat
``List[str]`` with one file name per video / camera (length V). Only the
shape of ``frame_ids`` and the returned structure differ: 1D takes a flat
list of frame ids and returns ``List[RGBFrame]``, 2D takes a ``V × F``
list-of-lists of frame ids and returns ``List[List[RGBFrame]]``.

| Feature | 1D Async ({py:class}`~accvlab.on_demand_video_decoder.PyNvSampleReader`) | 2D Batch Async ({py:class}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader`) |
|---------|---------|---------|
| ``filepaths`` shape | ``List[str]`` (len V) — one file per video | ``List[str]`` (len V) — same as 1D |
| Frame ids shape | ``List[int]`` (len V) | ``List[List[int]]`` (V × F, inner lists must be equal length) |
| Returned structure | ``List[RGBFrame]`` (len V) | ``List[List[RGBFrame]]`` (V × F) |
| Frames decoded per call | V | V × F |
| Result buffer | 1 result, V frames | 1 result, V × F frames |
| Pool sized at construction by | (n/a — per-reader) | ``max_frames_per_decode_call`` |

> **ℹ️ Note**: In one ``Decode()`` call, every video must request the **same**
> number of frames F. The ``frame_ids`` argument is shaped ``V × F``; jagged
> inner lists are rejected with ``invalid_argument``.

**Core APIs**

- {py:func}`~accvlab.on_demand_video_decoder.CreateBatchAsyncStreamReader`: Construct a 2D batch async reader
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.Decode`: Submit an async 2D decode (returns immediately)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.GetBuffer`: Block until decode is done and return decoded frames

**Code Walkthrough**

Construct the reader. ``max_frames_per_decode_call`` is the F upper bound
(per ``Decode()`` call, not per video file):

```python
import accvlab.on_demand_video_decoder as nvc

reader = nvc.CreateBatchAsyncStreamReader(
    num_of_set=1,
    num_of_file=6,                  # V upper bound
    max_frames_per_decode_call=4,   # F upper bound (per Decode() call)
    iGpu=0,
)
```

Build a 2D frame_ids and submit:

```python
V = len(file_path_list)
F = 4
# frame_ids[v][f] = f-th frame requested for video v.
# All inner lists must be the same length (jagged inner lengths are rejected).
frame_ids = [[0, 7, 14, 21]] * V

reader.Decode(file_path_list, frame_ids, as_bgr=False)
# Returns immediately; decoding happens on a background worker thread.
```

Retrieve the result:

```python
out = reader.GetBuffer(file_path_list, frame_ids, as_bgr=False)
# out is List[List[RGBFrame]] indexed [v][f].
# out[v][f].shape == (H, W, 3), dtype uint8, GPU memory.
```

**Two Contracts to Remember**

> **ℹ️ Note**: When 
> {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.GetBuffer` 
> returns, all GPU work (decode + internal copies) is already complete. You 
> can read the returned frames on any CUDA stream — including PyTorch's 
> default stream — without additional synchronization.

> **⚠️ Important**: The returned 
> {py:class}`~accvlab.on_demand_video_decoder.RGBFrame` objects are zero-copy 
> views into the reader's internal aggregator pool. Submitting the next 
> {py:meth}`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader.Decode` 
> reuses that memory. You **must** clone every frame you want to keep 
> **before** the next ``Decode()`` call. Skipping the clone leads to silent 
> data corruption.

**Canonical Prefetch Pattern**

```python
# Iteration 0: prime the pipeline
reader.Decode(files, frame_ids_0, as_bgr=False)
out = reader.GetBuffer(files, frame_ids_0, as_bgr=False)

# Clone before submitting the next batch
tensors_0 = [
    [torch.as_tensor(out[v][f], device="cuda").clone() for f in range(F)]
    for v in range(V)
]

# Prefetch iteration 1 in parallel with processing iteration 0
reader.Decode(files, frame_ids_1, as_bgr=False)
# ... process tensors_0 here (model forward, etc.) ...

# Iteration 1: GetBuffer is usually already-ready because of the prefetch
out = reader.GetBuffer(files, frame_ids_1, as_bgr=False)
tensors_1 = [
    [torch.as_tensor(out[v][f], device="cuda").clone() for f in range(F)]
    for v in range(V)
]
reader.Decode(files, frame_ids_2, as_bgr=False)
# ... process tensors_1 ...
```

**Resolution Handling**

Videos in a single ``Decode()`` call may have **different resolutions** — each
video gets its own per-slot aggregator pool, sized lazily to that video's
``F * H_v * W_v * 3`` on the first ``Decode()`` that hits the slot. If a later
``Decode()`` swaps in a video at the same slot with a different resolution,
the pool is reallocated automatically (grows if larger; reuses the existing
allocation if same or smaller).

Per-frame shape consequence: ``out[v][f].shape == (H_v, W_v, 3)`` may vary
across ``v``. The frames are not stack-able into a single
``[V, F, H, W, 3]`` tensor without resize/pad — that is a physical fact of
mixed-resolution input, not an API limitation.

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleBatchAsyncStreamAccess.py
```

### 3.3 Separation Access Decoding

Separation Access mode decouples demuxing and decoding into two separate stages. This provides fine-grained control over the video processing pipeline and enables advanced optimization strategies.

#### 3.3.1 Use Cases

- Need separate control over demuxing and decoding
- One-time demuxing, multiple decoding operations
- Inspection or processing of intermediate GOP bundles
- Custom processing pipelines

#### 3.3.2 Two-Stage Architecture

```
Stage 1 (Demuxing, CPU only):
Video Files → GetGOPList() → serialized GOP bundles, one per video
                             [(bundle, first_frame_ids, gop_lens), ...]

Stage 2 (Decoding, GPU):
serialized GOP bundles → DecodeFromGOPListRGB() → Decoded Frames
```

Stage 1 extracts, for each video, the GOP (Group of Pictures) that contains the
requested frame as a serialized GOP bundle. Stage 2 decodes the requested frames
from the bundles on the GPU. Because the two stages communicate only through the
bundles, they can run at different times, in different processes, or on different
machines.

#### 3.3.3 Sample: Demuxer/Decoder Separation Access

**File:** `packages/on_demand_video_decoder/samples/SampleDemuxerDecoderSeparationAccess.py`

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`: Stage 1 — extract serialized GOP bundles 
  (demuxing only)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Stage 2 — decode frames 
  from serialized GOP bundles (decoding only)

**Code Walkthrough**

Initialize one decoder per stage:

```python
# Stage 1 decoder: for packet extraction
nv_gop_dec1 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# Stage 2 decoder: for packet decoding
nv_gop_dec2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

> **ℹ️ Note**: Using separate decoder instances allows independent configuration and resource management.
> In particular, Stage 1 (demuxing) is CPU-only, so the demuxing decoder can live in DataLoader worker
> processes while the decoding decoder stays in the main process.

Stage 1 - Extract serialized GOP bundles:

```python
file_path_list = [
    "/data/videos/CAM_BACK_LEFT.mp4",
    "/data/videos/CAM_BACK.mp4",
    "/data/videos/CAM_BACK_RIGHT.mp4",
    "/data/videos/CAM_FRONT_LEFT.mp4",
    "/data/videos/CAM_FRONT.mp4",
    "/data/videos/CAM_FRONT_RIGHT.mp4",
]

# Extract the GOP containing frame 77 from every video
gop_list = nv_gop_dec1.GetGOPList(
    file_path_list,
    [77] * len(file_path_list)
)
```

**Understanding the return value:** `gop_list` holds one tuple per video:

```python
for i, (packets, first_frame_ids, gop_lens) in enumerate(gop_list):
    print(f"Video {i}:")
    print(f"  Bundle size: {len(packets)} bytes")        # serialized GOP bundle (numpy array)
    print(f"  First frame ID: {first_frame_ids[0]}")     # first frame of the extracted GOP
    print(f"  GOP length: {gop_lens[0]}")                # number of frames in the GOP
```

Any frame inside `[first_frame_id, first_frame_id + gop_len)` can later be decoded
from this bundle — no further access to the video file is needed.

Stage 2 - Decode frames from the bundles:

```python
# Pick a target frame within each GOP's range
frame_id_list = [
    random.randint(first_ids[0], first_ids[0] + lens[0] - 1)
    for _, first_ids, lens in gop_list
]

decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    [packets for packets, _, _ in gop_list],  # serialized GOP bundles, one per video
    file_path_list,                           # Corresponding file paths
    frame_id_list,                            # Target frame indices
    True                                      # BGR output
)

# Convert to PyTorch tensors on GPU (shape (height, width, 3), uint8)
tensors = [torch.as_tensor(frame).clone() for frame in decoded_frames]
```

**Decoding a subset**

Because each video has its own serialized GOP bundle, Stage 2 does not have to
decode everything that Stage 1 extracted. For example, decode only the front cameras:

```python
selected_indices = [3, 4, 5]  # Front-left, front, front-right

decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    [gop_list[i][0] for i in selected_indices],
    [file_path_list[i] for i in selected_indices],
    [frame_id_list[i] for i in selected_indices],
    True
)
```

This is what makes the separated pipeline flexible: serialized GOP bundles can be
kept in memory, cached, or persisted to disk (see Section 3.4), and decoded later
in any combination.

**Validation**

Always validate that frame IDs are within GOP range:

```python
if frame_id < first_frame_id or frame_id >= first_frame_id + gop_len:
    print(f"Frame {frame_id} is out of range for GOP starting at {first_frame_id}")
```

**Advantages of Separation**

1. Demux once, decode multiple times with different frame selections
2. CPU-bound demuxing and GPU-bound decoding can be scheduled and scaled independently
3. Decode any subset of the extracted videos
4. The serialized GOP bundle is a portable intermediate: cache it in memory (Section 3.3.4),
   share it across processes (Section 3.5), or persist it to disk (Section 3.4)

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleDemuxerDecoderSeparationAccess.py
```

#### 3.3.4 GOP Caching Feature

The GOP caching feature automatically stores extracted serialized GOP bundles in Python memory, eliminating the need for 
manual cache management by the user. When enabled, subsequent calls to {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` with the same 
video file and a `frame_id` within the cached GOP range will return the cached bundle without re-demuxing.

**Why Use GOP Caching?**

In training scenarios, especially with video datasets:
- The same video file may be accessed multiple times with different frame indices
- Multiple frame indices often fall within the same GOP (Group of Pictures)
- Re-demuxing for each access wastes I/O and CPU resources

Without caching, users would need to manually track GOP ranges and manage cache dictionaries. With the 
`useGOPCache` parameter, this is handled automatically.

**Enabling GOP Caching**

Set `useGOPCache=True` when calling {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`:

```python
import accvlab.on_demand_video_decoder as nvc

decoder = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# First call - fetches serialized GOP bundles from video files
gop_list = decoder.GetGOPList(
    file_path_list, 
    [77] * len(file_path_list), 
    useGOPCache=True
)

# Second call with frame_id=80 (within the same GOP range) - returns from cache
gop_list = decoder.GetGOPList(
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

Use the {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.isCacheHit` method to check whether the last {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList` call hit the cache:

```python
# Call GetGOPList with caching
gop_list = decoder.GetGOPList(file_path_list, frame_ids, useGOPCache=True)

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
| {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.clear_cache` | Clears all cached GOP bundles |

Example:

```python
# Get cache information
cache_info = decoder.get_cache_info()
print(f"Cached files: {cache_info['cached_files_count']}")
print(f"File paths: {cache_info['cached_files']}")

# Clear all cache when done
decoder.clear_cache()
```

**Partial Cache Hits**

Cache hits are tracked per video, so a single call can mix hits and misses:

```python
# First call - all videos are fetched
gop_list = decoder.GetGOPList(file_path_list, [77, 77, 77], useGOPCache=True)
print(decoder.isCacheHit())  # [False, False, False]

# Second call with some frame_ids in range, some out of range
gop_list = decoder.GetGOPList(file_path_list, [80, 80, 150], useGOPCache=True)
print(decoder.isCacheHit())  # [True, True, False] - partial cache hit
```

> **⚠️ Note**: The cache is stored in Python memory. Each video file caches only one GOP (the most 
> recently accessed). The cache uses LRU eviction and is bounded by `gopCacheCapacity` in
> {py:func}`~accvlab.on_demand_video_decoder.CreateGopDecoder`; when omitted, it defaults to `maxfiles`.
> Use {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.clear_cache` to release memory manually.

**When to Use GOP Caching**

✓ Training loops with random frame sampling from the same video
✓ Multi-camera setups where cameras are often accessed with similar frame indices
✓ Scenarios where the same GOP is likely to be accessed multiple times
✓ Reducing I/O overhead in data loading pipelines

✗ One-time video processing (no repeated access)
✗ Memory-constrained environments with large video collections
✗ Scenarios where each frame access targets a different GOP

### 3.4 Demuxer-Free Decoding

Demuxer-Free mode allows decoding directly from pre-extracted serialized GOP bundles, either stored on disk or in memory. This approach is ideal for scenarios requiring repeated access to the same video segments.

#### 3.4.1 Use Cases

- Pre-processing video datasets for training
- Repeated access to same video segments
- Disk storage for GOP bundle caching
- Eliminating demuxing overhead in production
- PyTorch DataLoader integration with worker processes

#### 3.4.2 Sample: GOP File Storage and Decoding

**File:** `packages/on_demand_video_decoder/samples/SampleDecodeFromGopFiles.py`

**Two-Phase Workflow**

```
Phase 1: GOP Data Preparation
Video Files → GetGOPList() → SaveGopToFile() → one .bin file per video

Phase 2: Decoding from Files
.bin files → LoadGopsToList() → DecodeFromGOPListRGB() → Decoded Frames
```

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`: Extract serialized GOP bundles (one per video)
- {py:func}`~accvlab.on_demand_video_decoder.SaveGopToFile`: Save one video's serialized GOP bundle to a binary file
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.LoadGopsToList`: Load GOP files back
  (one numpy array per file)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Decode from list of GOP 
  data

**Code Walkthrough**

Initialize decoders:

```python
# Decoder for packet extraction
nv_gop_dec1 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

# Decoder for GOP file decoding
nv_gop_dec2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
```

Phase 1 - Extract serialized GOP bundles and save one file per video:

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

frames = [random.randint(0, 200) for _ in range(len(file_list))]

# Extract serialized GOP bundles for all videos in one call (one per video)
gop_list = nv_gop_dec1.GetGOPList(file_list, frames)

packet_files = []
for i, (numpy_data, first_frame_ids, gop_lens) in enumerate(gop_list):
    packet_file = f"./gop_{camera_names[i]}.bin"
    nvc.SaveGopToFile(numpy_data, packet_file)
    packet_files.append(packet_file)

    print(f"Saved GOP bundle: {os.path.getsize(packet_file)} bytes")
```

Phase 2 - Load the GOP files and decode:

```python
# Load the GOP files (one numpy array per file)
gop_data_list = nv_gop_dec2.LoadGopsToList(packet_files)

# gop_data_list is a list of numpy arrays, one per video
print(f"Loaded {len(gop_data_list)} GOP bundles")
for i, gop_data in enumerate(gop_data_list):
    print(f"  Bundle {i} ({camera_names[i]}): {len(gop_data)} bytes")

# Decode all videos
decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
    gop_data_list,  # List of serialized GOP bundles
    file_list,      # List of file paths
    frames,         # List of frame IDs
    as_bgr=True
)

# Convert to PyTorch tensors on GPU (shape (height, width, 3), uint8)
tensors = [torch.as_tensor(frame).clone() for frame in decoded_frames]
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

Cleanup temporary files:

```python
for packet_file in packet_files:
    if os.path.exists(packet_file):
        os.remove(packet_file)
```

**File Format**

Each GOP file is a binary file containing one video's serialized GOP bundle
(the same self-contained format returned by one element of {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`,
with an embedded frame offset table). Recommended file extension: `.bin`.

**Storage Considerations**

- GOP file size: Typically 5-15% of original video size
- Storage savings: ~85-95% compared to extracted frames
- I/O performance: SSD recommended for best performance

**Key Advantages**

1. Eliminate demuxing overhead: Same video segments accessed repeatedly (e.g.,
   training multiple epochs) are demuxed only once
2. Memory efficiency: Load only needed videos
3. Flexible loading: Different subsets for different batches
4. Distributed caching: Store videos on different machines
5. Per-video cache management: Independent expiration policies

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleDecodeFromGopFiles.py
```

#### 3.4.3 Pattern: Batch Decode from Multiple Demux Results

**When to Use**

A common integration pattern is multiple demuxing operations followed by a single batch decode:
- Demux executed N times separately (e.g., in DataLoader `__getitem__`, called batch_size times)
- Decode executed once for the entire batch
- Enables parallel demuxing in worker processes, centralized batch decoding in main process
- No disk I/O for GOP bundles (passed in memory)

**Architecture: N Demux → 1 Batch Decode**

```
Worker/Process 1: Video File 1 → GetGOPList() → bundle_1 (in memory)
Worker/Process 2: Video File 2 → GetGOPList() → bundle_2 (in memory)
Worker/Process 3: Video File 3 → GetGOPList() → bundle_3 (in memory)
                     ⋮                            ⋮
Worker/Process N: Video File N → GetGOPList() → bundle_N (in memory)
                                                      ↓
                          Collect all bundles: [bundle_1, bundle_2, ..., bundle_N]
                                                      ↓
                  Main Process: DecodeFromGOPListRGB() → Batch of N Frames (single decode call)
```

**Core APIs**

- {py:meth}`~accvlab.on_demand_video_decoder.CachedGopDecoder.GetGOPList`: Extract serialized GOP bundles (called N times, 
  possibly in parallel)
- {py:meth}`~accvlab.on_demand_video_decoder.PyNvGopDecoder.DecodeFromGOPListRGB`: Batch decode from list of 
  bundles (called once for entire batch)

**DataLoader Integration Pattern**

In a real PyTorch DataLoader:

```python
# In worker process (worker_fn)
def worker_fn(video_path, frame_id):
    gop_list = decoder.GetGOPList([video_path], [frame_id])
    packets, first_ids, gop_lens = gop_list[0]
    return packets

# In main process collate_fn
def collate_fn(batch):
    packets_list = [item['packets'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    
    # Batch decode in main process and convert to PyTorch tensors
    frames = decoder.DecodeFromGOPListRGB(packets_list, file_paths, frame_ids, True)
    return [torch.as_tensor(frame).clone() for frame in frames]
```

**Key Benefits**

1. **Parallel demuxing**: Each worker demuxes independently in parallel
2. **Single batch decode**: GPU decoder called only once for entire batch (efficient GPU utilization)
3. **No disk I/O**: Packets passed in memory, no temporary file storage
4. **Resource separation**: CPU-heavy demuxing in workers, GPU decoding in main process

**Memory Management**

- Keep bundle lifetime short (decode and release)
- Monitor memory usage in worker processes
- Balance worker count with available memory

**Where to See It in Action**

- `packages/on_demand_video_decoder/samples/SampleDemuxerDecoderSeparationAccess.py` — the two-stage
  demux/decode flow within one process
- `packages/on_demand_video_decoder/samples/SampleSharedGopStore.py` — passing worker-demuxed GOP bundles
  to the main process via shared memory (see Section 3.5)
- `packages/on_demand_video_decoder/examples/dataloader_separation_decode/` — a complete PyTorch
  DataLoader integration

### 3.5 Shared GOP Store

SharedGopStore provides a cross-process shared memory cache for serialized GOP bundles, designed for PyTorch 
DataLoader integration. Workers store demuxed GOP bundles in POSIX shared memory and pass lightweight `GopRef` 
references through the IPC queue, while the main process reads the data as zero-copy numpy views.

#### 3.5.1 Use Cases

- Multi-worker DataLoader with separation access (workers demux, main process decodes on GPU)
- Multi-camera setups where different workers may request overlapping GOPs
- Reducing redundant demuxing when multiple workers access the same video segment
- Training pipelines that need to pass GOP data from workers to main process efficiently

#### 3.5.2 Architecture

```
Main Process                     Worker Processes
─────────────                    ────────────────
SharedGopStore.create()
    │
    ├──spawn──> Worker 0: SharedGopStore.attach()
    │               lookup(video, frame_id)
    │                 ├─ HIT  → return GopRef
    │                 └─ MISS → demux from disk
    │                          put(video, data) → GopRef
    │               queue.put(GopRef)  ← tens of bytes
    │
    ├──spawn──> Worker 1: (same pattern)
    │               ...
    │
    ◄── queue.get() ── [GopRef, GopRef, ...]
    │
    get_batch(refs)  ← zero-copy numpy views
    │
    DecodeFromGOPListRGB(...)  ← GPU decode
    │
    cleanup()  ← unlink all shm blocks
```

#### 3.5.3 Sample: SharedGopStore

**File:** `packages/on_demand_video_decoder/samples/SampleSharedGopStore.py`

> **ℹ️ Note**: This sample is a pure CPU / shared-memory demo. No GPU or video files are required — 
> GOP data is simulated with random bytes.

**Core APIs**

- {py:class}`~accvlab.on_demand_video_decoder.SharedGopStore`: Cross-process shared memory GOP cache
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.create`: Allocate a new store (main process)
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.attach`: Attach to existing store (worker processes)
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.lookup`: Lock-free cache lookup, returns
    {py:class}`~accvlab.on_demand_video_decoder.GopRef` or ``None``
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.put`: Store GOP data, returns
    {py:class}`~accvlab.on_demand_video_decoder.GopRef`
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.get_batch`: Read a batch of
    {py:class}`~accvlab.on_demand_video_decoder.GopRef` as zero-copy numpy views (main process)
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.read`: Read a single
    {py:class}`~accvlab.on_demand_video_decoder.GopRef` as a zero-copy numpy view
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.cleanup`: Unlink all shared memory blocks
    (main process, on shutdown)
  - {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.close`: Close handles without unlinking
    (worker processes, before exit)
- {py:class}`~accvlab.on_demand_video_decoder.GopRef`: Lightweight, picklable reference to GOP data in
  shared memory (passed through DataLoader IPC queue)

**Code Walkthrough**

**Step 1: Main process creates the store before spawning workers**

```python
from accvlab.on_demand_video_decoder import SharedGopStore

STORE_ID = 0       # typically LOCAL_RANK
CAPACITY = 120     # must exceed in-flight GOPs (see sizing below)

store = SharedGopStore.create(capacity=CAPACITY, store_id=STORE_ID)
```

**Step 2: Worker processes attach and perform lookup/put**

```python
def worker_fn(store_id, capacity, tasks, result_queue):
    store = SharedGopStore.attach(capacity=capacity, store_id=store_id)

    refs = []
    for video_path, frame_id, gop_first_frame, gop_len in tasks:
        # Lock-free lookup
        ref = store.lookup(video_path, frame_id)
        if ref is None:
            # Cache miss: demux from disk (or simulate)
            gop_data = demux_gop_from_video(video_path, frame_id)
            ref = store.put(video_path, gop_first_frame, gop_len, gop_data)
        refs.append(ref)

    # Send lightweight refs (tens of bytes each) through IPC queue
    result_queue.put(refs)
    store.close()
```

**Step 3: Main process reads zero-copy data and decodes**

```python
# `queue_a` and `queue_b` are the IPC queues belonging to two separate
# DataLoader workers spawned in Step 2. Each worker pushes its own list
# of GopRefs onto its own queue; the main process gathers them here.
all_refs = queue_a.get() + queue_b.get()

# Read shared memory blocks as zero-copy numpy views
arrays = store.get_batch(all_refs)

# Decode on GPU
decoded_frames = decoder.DecodeFromGOPListRGB(arrays, file_paths, frame_ids, True)
```

**Step 4: Cleanup on shutdown**

```python
store.cleanup()  # unlinks all /dev/shm blocks for this store
```

**Capacity Sizing**

The `capacity` parameter must exceed the maximum number of GOPs that can be "in flight" (queued in the 
DataLoader + being consumed by the training loop):

```
min_capacity > (prefetch_factor * num_workers + 1) * batch_size * num_cameras
```

A recommended formula is:

```python
capacity = batch_size * num_cameras * 10
```

If capacity is too small, GOPs may be evicted before the main process can read them. In this case,
{py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.read` returns a zeros array and emits a
`RuntimeWarning` with diagnostic information instead of crashing.

**GopRef IPC Efficiency**

{py:class}`~accvlab.on_demand_video_decoder.GopRef` is a `NamedTuple` with 4 fields (shm_name, data_size,
first_frame_id, gop_len). It serializes to ~60 bytes via pickle, compared to ~4-40 KB for the actual serialized
GOP bundle. This makes DataLoader IPC overhead negligible.

```python
import pickle
from accvlab.on_demand_video_decoder import GopRef

ref = GopRef(shm_name="gs_0_12345_0", data_size=4096, first_frame_id=0, gop_len=30)
print(len(pickle.dumps(ref)))  # ~60 bytes
```

**LRU Eviction**

When the store is full, {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.put` evicts the
least-recently-used entry (lowest `access_tick`). Both
{py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.lookup` and
{py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.put` refresh an entry's tick, so frequently
accessed GOPs are retained. Evicted shm blocks are cleaned up during the next
{py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.get_batch` call.

**Cross-Process Safety**

- {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.lookup` is lock-free (worst case: stale miss, one extra disk read)
- {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.put` acquires an `flock` for atomicity
- {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.get_batch` acquires an `flock` to prevent eviction while opening handles
- Works with `spawn`'d DataLoader workers (unlike `multiprocessing.Lock`)

**Running the Sample**

```bash
cd packages/on_demand_video_decoder/samples
python SampleSharedGopStore.py
```

Expected output:

```text
SharedGopStore Demo
============================================================

[Main] Creating SharedGopStore (capacity=12, store_id=0)

[Main] Spawning 2 workers...
  [Worker 12345] Attached to store (id=0, capacity=12)
  [Worker 12345] MISS /data/video/cam0.mp4 frame=15 -> put as gs_0_...
  ...
  [Worker 12346] HIT  /data/video/cam0.mp4 frame=20
  ...

[Main] Received 12 GopRef references from workers
[Main] GopRef size: 60 bytes (vs ~4096 bytes of actual GOP data)

[Main] Got 12 zero-copy numpy views:
  [0] shape=(4096,), dtype=uint8, nbytes=4096
  ...

[Main] Cleanup complete. All shared memory released.
[Main] Verified: no shared memory files leaked.
```
