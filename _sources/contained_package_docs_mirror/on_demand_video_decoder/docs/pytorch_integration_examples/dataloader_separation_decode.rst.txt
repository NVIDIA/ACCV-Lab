========================================
DataLoader Separation Decode Example
========================================

Overview
========

This example demonstrates how to use NVIDIA's accvlab.on_demand_video_decoder library with PyTorch DataLoader for efficient separation-based video decoding, ideal for training scenarios that require random access to video frames (e.g., random sampling for data augmentation).

The specific code implementation can be found in ``packages/on_demand_video_decoder/examples/dataloader_separation_decode``.

The "separation" approach refers to decoupling the video demuxing (CPU) and decoding (GPU) processes, which enables:

- **Process Isolation**: Demuxing runs in DataLoader workers (CPU-only), while decoding runs in the main process (GPU)
- **Memory Efficiency**: Avoids GPU memory duplication across multiple workers
- **Flexible Sampling**: Supports random frame access patterns efficiently

.. image:: ../images/separation_access.png
   :alt: SeparationAccess Optimization Overview

Key Features
============

- **Custom PyTorch Dataset**: Implements lazy initialization of the GPU decoder to optimize memory usage
- **Custom Sampler**: Organizes video clips into batches for efficient processing with distributed training support
- **Multi-camera Support**: Handles synchronized frames from multiple cameras with random or stream access
- **Distributed Training**: Compatible with PyTorch's distributed training framework
- **Performance Profiling**: Includes NVTX markers for performance analysis
- **Error Handling**: Comprehensive error handling and validation
- **Separation Decoding**: CPU demuxing in workers, GPU decoding in main process

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

- For Random Access Pattern, ref to ``packages/on_demand_video_decoder/docs/example/dataloader_random_decode.md``.
- For Stream Access Pattern, ref to ``packages/on_demand_video_decoder/docs/example/dataloader_stream_decode.md``.

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
- ``--disable_cache``: If cache gop in main processor.
- ``--frame_read_type``: Choose ``stream`` or ``random`` access pattern.

Architecture
============

VideoClipDatasetWithPyNVVideo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``VideoClipDatasetWithPyNVVideo`` class extends PyTorch's ``Dataset`` and provides:

- Lazy initialization of the NVIDIA separation decoder
- GPU-accelerated random frame decoding
- Multi-camera frame synchronization with random access
- Error handling and validation
- **Separation Decoder Configuration** (``accvlab.on_demand_video_decoder``):

  - ``demux_mode``: Separation mode decouples demuxing (CPU workers) from decoding (GPU main process)
  - ``device``: GPU device ID for decoding
  - ``packet_cache``: Caches GOP packets in main process to avoid IPC overhead

**Implementation:**

.. note-literalinclude:: ../../examples/dataloader_separation_decode/main.py
   :language: python
   :caption: examples/dataloader_separation_decode/main.py
   :linenos:
   :lineno-match:
   :start-at: # .. doc-marker-begin: dataset-separation
   :end-before: # .. doc-marker-end: dataset-separation

VideoClipSampler
^^^^^^^^^^^^^^^^

The ``VideoClipSampler`` class extends PyTorch's ``Sampler`` and provides:

- Random clip sampling for data diversity
- Efficient batch organization with flexible sampling
- Distributed training support with proper sharding
- Optional shuffling support for each epoch

Example Training Loop
^^^^^^^^^^^^^^^^^^^^^

The following code shows the complete setup for training with separation-based decoding:

.. note-literalinclude:: ../../examples/dataloader_separation_decode/main.py
   :language: python
   :caption: examples/dataloader_separation_decode/main.py
   :linenos:
   :lineno-match:
   :start-at: # .. doc-marker-begin: training-setup-separation
   :end-before: # .. doc-marker-end: training-setup-separation

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

Using the Separation Access API, measure the throughput (frames per second) with stream frame access:

===============  =============  ================
configuration    w gop cache    w/o gop cache
===============  =============  ================
1 GPU            3270.13        1069.59
2 GPU            3699.34        1881.1
4 GPU            4337.68        2809.6
8 GPU            4834.13        3824.36
===============  =============  ================

Using the Separation Access API, measure the throughput (frames per second) between stream frame access and random frame access:

===============  ===============  ==================
configuration    stream access    random access
===============  ===============  ==================
1 GPU            3270.13          311.2
2 GPU            3699.34          603.94
4 GPU            4337.68          1220.55
8 GPU            4834.13          2345.44
===============  ===============  ==================

Performance Optimization Tips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Fork Mode**: Fork or spawn mode both supported; fork recommended for CPU-only demuxing in workers
- **GPU Memory**: Only main process requires GPU resources for decoding, workers are CPU-only
- **Batch Size**: Adjust ``group_num`` based on available GPU memory
- **Worker Processes**: Adjust ``num_workers`` based on CPU cores and I/O requirements
- **Stream Access Advantage**: Separation decoder efficiently handles stream frame access through packet caching
- **GOP Size Impact**: Smaller GOP sizes improve random access performance (less decoding overhead per frame)
- **Packet Cache**: GOP packets are cached in the main process, avoiding memory IPC overhead between workers
- **Memory Efficiency**: No GPU context duplication across workers, all decoding happens in main process

Performance Profiling
^^^^^^^^^^^^^^^^^^^^^^

Use NVIDIA Nsight Systems for detailed performance analysis:

.. code-block:: bash

   nsys profile --trace-fork-before-exec true -w true -f true -t cuda,nvtx,osrt,cudnn,cublas,nvvideo --gpu-video-device all -x true -o dataloader_separation_decode python main.py --index_file /path/to/index_frame.json

