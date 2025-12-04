===================================
DataLoader Stream Decode Example
===================================

Overview
========

This example demonstrates how to use NVIDIA's accvlab.on_demand_video_decoder library with PyTorch DataLoader for efficient stream-based video decoding in batch form, ideal for training scenarios that require sequential temporal processing (e.g., StreamPETR).

The specific code implementation can be found in ``packages/on_demand_video_decoder/examples/dataloader_stream_decode``.

.. image:: ../images/stream_access.png
   :alt: StreamAccess Optimization Overview

Key Features
============

- **Custom PyTorch Dataset**: Implements lazy initialization of the stream decoder to support multiple processes
- **Custom Sampler**: Organizes sequential video clips into batches for efficient processing with distributed training support
- **Multi-camera Support**: Handles synchronized frames from multiple cameras with sequential access
- **Distributed Training**: Compatible with PyTorch's distributed training framework
- **Performance Profiling**: Includes NVTX markers for performance analysis
- **Error Handling**: Comprehensive error handling and validation
- **Stream Decoding**: Sequential frame extraction optimized for temporal continuity

Index Data Format
=================

.. note::

   The JSON file and the selection of frames to access are used here for demonstration purposes. In a
   real-world scenario, the file names and indices would be obtained from the dataset metadata.

The example expects a JSON file with the following structure:

.. code-block:: text

   {
       "video_dir1": {
           "clip0id": frame_count,
           "clip1id": frame_count,
           ...
       },
       "video_dir2": {
           "clip0id": frame_count,
           "clip1id": frame_count,
           ...
       }
   }

Each video directory should contain subdirectories for each clip, and each clip directory should contain MP4 files for different cameras.

Data Loading Structure
======================

The data loading process follows a hierarchical structure optimized for **sequential access and temporal continuity**:

.. code-block:: text

   Sequential Sampling Dimension
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                         STREAM DECODING PIPELINE                            │
   │                    (Sequential Frame Access)                                │
   └─────────────────────────────────────────────────────────────────────────────┘

   Batch Dimension (group_num clips per batch) - Temporal Batching
   ├── Batch 0 (e.g., 2 clips)
   │   ├─→ Stream Clip A (from video_dir/clipA)
   │   │   └─→ Strean Frame 0 → [cam0_f0,  cam1_f0,  cam2_f0,  cam3_f0,  cam4_f0,  cam5_f0 ]
   │   │
   │   └─→ Stream Clip B (from video_dirB/clipB)
   │       └─→ Stream Frame 0 → [cam0_f0,  cam1_f0,  cam2_f0,  cam3_f0,  cam4_f0,  cam5_f0 ]
   │
   ├── Batch 1 (e.g., 2 clips)
   │   ├─→ Stream Clip A (from video_dir/clipA)
   │   │   └─→ Stream Frame 1 → [cam0_f1,  cam1_f1,  cam2_f1,  cam3_f1,  cam4_f1,  cam5_f1 ]
   │   │
   │   └─→ Stream Clip B (from video_dir/clipB)
   │       └─→ Stream Frame 1  → [cam0_f1,  cam1_f1,  cam2_f1,  cam3_f1,  cam4_f1,  cam5_f1 ]
   │
   ├── Batch 2 (e.g., 2 clips)
   │   ├─→ Stream Clip A (from video_dir/clipA)
   │   │   └─→ Stream Frame 2 → [cam0_f2,  cam1_f2,  cam2_f2,  cam3_f2,  cam4_f2,  cam5_f2 ]
   │   │
   │   └─→ Stream Clip B (from video_dir/clipB)
   │       └─→ Stream Frame 2 → [cam0_f2,  cam1_f2,  cam2_f2,  cam3_f2,  cam4_f2,  cam5_f2 ]
   │
   └── ... (continuous stream flow)

   Data Flow with Time Dimension:
   Time → Batch → Clip → Frame_ID → Multi-camera frames
    ↓      ↓       ↓        ↓              ↓
   t=0ms  group_num clips  frame_count   [cam0_t, cam1_t, cam2_t, ...]

**Sequential Access Dimension Details:**

- **Sequential Order**: Clips are processed in order (maintaining temporal sequence)
- **Batch**: Contains ``group_num`` sequential clips (batch size is **fixed**)
- **Sequential Clip**: Each clip is processed in order from the same or adjacent video directories
- **Sequential Frame_ID**: Frame indices within each clip are accessed sequentially (f0, f1, f2, ...)
- **Multi-camera**: **Fixed number** of synchronized cameras, all accessing the **same frame index** simultaneously

  - Example: If frame_id=15 is requested, all cameras (cam0, cam1, cam2, ..., cam5) decode frame 15
  - Camera count remains constant across all clips and batches

**Stream Access Benefits:**

- **Temporal Continuity**: Maintains chronological order for time-sensitive applications (e.g., StreamPETR)
- **Memory Efficiency**: Only decode frames when needed, reducing memory footprint
- **Cache Friendly**: Sequential access patterns improve cache hit rates
- **Efficient Decoding**: Leverages video codec's sequential decoding optimization
- **Real-time Capability**: Supports live video stream processing with predictable latency

Usage
=====

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   python main.py --index_file /path/to/index_frame.json --group_num 4 --num_workers 2

Distributed Training
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m torch.distributed.run --nproc_per_node=2 main.py --index_file /path/to/index_frame.json --group_num 4 --num_workers 2

Command Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``--index_file``: Path to the JSON file containing frame index information (default: ``example/index_frame.json``)
- ``--group_num``: Number of clips to process in each batch (default: 4)
- ``--num_workers``: Number of worker processes for data loading (default: 2)

Architecture
============

VideoClipDataset
^^^^^^^^^^^^^^^^

The ``VideoClipDataset`` class extends PyTorch's ``Dataset`` and provides:

- Lazy initialization of the NVIDIA stream decoder
- GPU-accelerated sequential frame decoding
- Multi-camera frame synchronization with sequential access
- Error handling and validation
- **Stream Decoder Configuration** (``nvc.CreateSampleReader``):

  - ``num_of_set``: Controls the decoding cycle for sequential access
  - ``num_of_file``: Maximum number of video files to keep open simultaneously
  - ``iGpu``: GPU device ID for decoding

**Implementation:**

.. note-literalinclude:: ../../examples/dataloader_stream_decode/main.py
   :language: python
   :caption: examples/dataloader_stream_decode/main.py
   :linenos:
   :lineno-match:
   :start-at: # .. doc-marker-begin: dataset-stream
   :end-before: # .. doc-marker-end: dataset-stream

VideoClipSampler
^^^^^^^^^^^^^^^^

The ``VideoClipSampler`` class extends PyTorch's ``Sampler`` and provides:

- Sequential clip ordering for temporal continuity
- Efficient batch organization with temporal locality
- Distributed training support with proper sharding
- Optional shuffling support for each epoch

Example Training Loop
^^^^^^^^^^^^^^^^^^^^^

The following code shows the complete setup for training with stream-based sequential access:

.. note-literalinclude:: ../../examples/dataloader_stream_decode/main.py
   :language: python
   :caption: examples/dataloader_stream_decode/main.py
   :linenos:
   :lineno-match:
   :start-at: # .. doc-marker-begin: training-setup-stream
   :end-before: # .. doc-marker-end: training-setup-stream

Performance
===========

Test Environment
^^^^^^^^^^^^^^^^

- **GPU**: NVIDIA A100-SXM4-80GB
- **CPU**: AMD EPYC 7J13 64-Core Processor
- **Storage**: NFS
- **DataSet**: NuScenes-mini, 10 clips, 6 cameras, H.265, 1600x900, GOP_SIZE=30, non-bframe

Performance Metrics (frames/sec)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the Stream Access API, measure the throughput (frames per second):

===============  ============
configuration    throughput      
===============  ============
1 GPU            2481.54 
2 GPU            2885.02 
4 GPU            3236.65 
8 GPU            3497.02 
===============  ============

Performance Optimization Tips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Spawn Mode**: PyTorch must be set to spawn mode when using GPU (NvCodec) within the DataLoader
- **GPU Memory**: Each worker maintains its own CUDA context, which can lead to increased GPU memory usage
- **Batch Size**: Adjust ``group_num`` based on available GPU memory
- **Worker Processes**: Adjust ``num_workers`` based on CPU cores and I/O requirements
- **Sequential Access Advantage**: Stream decoder is optimized for sequential access, significantly faster than random access
- **GOP Size Impact**: Sequential access benefits from any GOP size as frames are decoded in order
- **File Handle Management**: The ``num_of_file`` parameter controls how many video files stay open; tune based on dataset size
- **Cache Locality**: Sequential access patterns have higher cache hit rates compared to random access
- **Num of Set Configuration**: The parameter ``num_of_set`` in ``nvc.CreateSampleReader`` controls the decoding cycle:

  - For a specific decoder instance, if you are decoding clipA, after calling ``DecodeN12ToRGB`` ``num_of_set`` times, the input returns to clipA again
  - If you are continuously decoding the same clip, then ``num_of_set`` can be set to 1
  - For batch processing with multiple clips, set ``num_of_set`` to match the number of clips in rotation

Performance Profiling
^^^^^^^^^^^^^^^^^^^^^^

Use NVIDIA Nsight Systems for detailed performance analysis:

.. code-block:: bash

   nsys profile --trace-fork-before-exec true -w true -f true -t cuda,nvtx,osrt,cudnn,cublas,nvvideo --gpu-video-device all -x true -o dataloader_stream_decode python main.py --index_file /path/to/index_frame.json

