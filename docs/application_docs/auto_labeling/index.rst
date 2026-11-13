Auto Labeling
=============

This application demonstrates how to use ACCV-Lab to accelerate an auto-labeling
pipeline, with a concrete example based on
`StreamPETR <https://arxiv.org/abs/2303.11926>`_.

The example processes calibrated six-camera driving videos and exports object
classes, 3D boxes, velocities and confidence scores as JSON in the LiDAR
coordinate frame. These predictions provide initial annotations for review,
correction and perception dataset preparation.

Environment and Weights
-----------------------

Use the :doc:`StreamPETR environment <../../demo_docs/streampetr/docker_setup>`
with its patched MMCV/MMDetection3D dependencies. Install ACCV-Lab's
``on_demand_video_decoder``, ``optim_test_tools``, ``batching_helpers`` and
``draw_heatmap`` packages following the
:doc:`installation guide <../../guides/INSTALLATION_GUIDE>`.

Use the official R50 428-query checkpoint from
`upstream StreamPETR <https://github.com/exiawsh/StreamPETR>`_:

.. code-block:: bash

   mkdir -p applications/auto_labeling/checkpoints
   curl -fL -o applications/auto_labeling/checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth \
     https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth

The included config inherits the demo model with 300 learned queries, 128
propagated queries and 512 memory entries. Checkpoint loading is strict and
preserves PyTorch version metadata. Only load trusted checkpoints and pickles.

Inference
---------

The input pickle contains ``infos`` with sample/scene tokens, timestamps in
microseconds, ego/lidar poses and calibrated ``cams``. Every camera entry needs
``timestamp``, ``data_path``, ``video_path``, ``video_frame_id``,
``sensor2lidar_rotation``, ``sensor2lidar_translation`` and ``cam_intrinsic``.
Frames are ordered by ``frame_idx``, or timestamp when that field is absent.
Camera order must be consistent across samples. Relative video paths resolve
against ``--video-base-path`` (by default ``--data-root``); image filenames are
retained for visualization.

Each batch slot follows one scene in temporal order on one GPU. Memory resets
between scene groups. Short scenes hold their last frame internally; finished
and padding slots never produce duplicate labels. Results stream to a temporary
JSON file, which replaces the output only after the run succeeds.

From the repository root:

.. code-block:: bash

   python -m applications.auto_labeling.inference \
     --checkpoint applications/auto_labeling/checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth \
     --pkl-path /data/nuscenes-mini/stream_petr/nuscenes2d_video_temporal_infos_train.pkl \
     --data-root /data/nuscenes-mini \
     --video-base-path /data/nuscenes-mini/video_samples \
     --batch-size 2 --device cuda:0 \
     --output-dir applications/auto_labeling/outputs/inference

``applications/auto_labeling/tools/run_inference.sh`` accepts the same options
from any working directory. ``--num-samples N`` limits the number of saved
samples; ``--score-threshold`` defaults to 0.3. Normalization and image dimensions
come from the model config. Read and decode failures terminate the run.

``detection_results.json`` is an array of samples containing tokens, image
filenames and detections. A detection contains ``translation``, ``size``
(dx, dy, dz), ``rotation_yaw`` (radians), ``velocity`` (vx, vy), ``score``,
``label`` and ``label_name``. Coordinates use the LiDAR frame; translation is
the box's bottom center. Class names come from the model config.

.. _auto-labeling-performance:

Performance
-----------

These measurements compare parallel CPU JPEG decoding with the complete
ACCV-Lab video input pipeline for auto-labeling.

Measurement Setup
~~~~~~~~~~~~~~~~~

Input videos are **HEVC (H.265) in MP4 containers, 1600 × 900 pixels per camera
(width × height), 10 fps, GOP 30**, with six camera streams per scene.
The CPU baseline decodes corresponding JPEG views at the same resolution.

Measurements use one NVIDIA H100 80 GB HBM3 GPU, 32 allocated CPU cores, and
StreamPETR R50 428q in the NGC PyTorch 25.05 container (PyTorch 2.8.0a0,
CUDA 12.9). A sample is one timestamp with six camera views. Batch size (BS)
is the number of parallel scenes on that single GPU.

All configurations process the same 1,024 samples: the first 16 frames of each of
64 scene streams from nuScenes-mini ×8. Scene/sample IDs are unique, while the
underlying image/video content is replicated. Each value is the median of three
runs after a separate five-batch warmup, with the input file cache primed.
Timing includes loading/decoding, preprocessing, model inference and transfer of
predictions to CPU. Initialization, warmup, checksums and JSON writing are
excluded. Throughput counts real samples and excludes padded or finished slots.

CPU JPEG versus ACCV-Lab by Batch Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPU JPEG baseline uses parallel CPU decoding with up to min(BS × 6, 32)
workers, followed by GPU preprocessing and GPU model inference. The complete
ACCV-Lab path enables batched NVDEC decoding across camera streams, GPU
preprocessing, GPU frame residency and asynchronous prefetch.

.. Data source: applications/auto_labeling/evaluation_results/throughput_by_bs.csv

.. figure:: _generated/throughput_by_bs.svg
   :alt: CPU JPEG and ACCV-Lab throughput by batch size for six-camera 1600 × 900 HEVC input on one H100, with ACCV-Lab reaching 134.52 samples/s at BS=64.
   :width: 100%

   Median throughput with minimum–maximum error bars over three runs.
   Green squares represent ACCV-Lab; blue circles represent parallel CPU JPEG.
   Both pipelines use the same GPU model inference and GPU preprocessing.
   Only fully occupied batches are plotted.

The highest measured throughput is **134.52 samples/s at BS=64**, versus
**89.97 samples/s** for parallel CPU JPEG at the same BS (**1.50×**). BS=64 is the
scene-count limit of this workload, so this is the best value in the tested
range. BS=48 is omitted from the plot because it uses only two-thirds of the
batch slots across its two scene groups; its throughput includes padding costs.

Ablations at the Best Measured Batch Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All controls use **BS=64**. Each row starts from the full ACCV-Lab path and applies
the indicated change. Throughput is shown as median (minimum–maximum) over three
runs; loss is 1 − variant throughput / full-path throughput. The final column
reports full-path throughput divided by variant throughput.

.. list-table:: ACCV-Lab pipeline ablations at BS=64
   :header-rows: 1
   :widths: 30 40 15 15

   * - Variant
     - Throughput (samples/s)
     - Loss
     - Full / variant
   * - ACCV-Lab (full)
     - 135.08 (134.44–135.73)
     - 0.0%
     - 1.00×
   * - No prefetch
     - 115.36 (114.67–115.49)
     - 14.6%
     - 1.17×
   * - Serial decode
     - 41.12 (41.11–41.52)
     - 69.6%
     - 3.28×
   * - Host round trip
     - 35.85 (35.56–36.16)
     - 73.5%
     - 3.77×
   * - CPU preprocessing
     - 8.28 (8.20–8.57)
     - 93.9%
     - 16.32×
   * - CPU video decode
     - 29.92 (29.83–29.99)
     - 77.8%
     - 4.51×
   * - CPU JPEG
     - 89.94 (89.62–90.14)
     - 33.4%
     - 1.50×

The controls are defined as follows:

* **No prefetch:** disable asynchronous next-batch decoding.
* **Serial decode:** submit frames one at a time, retaining cache capacity for
  BS × 6 videos and background prefetch.
* **Host round trip:** copy decoded frames GPU → CPU → GPU before GPU preprocessing.
* **CPU preprocessing:** perform preprocessing on the CPU, including the necessary
  host transfers.
* **CPU video decode:** use parallel CPU video decoding; color conversion and
  frame upload also change.
* **CPU JPEG:** use the parallel JPEG baseline described above.

The full path is measured again in interleaved ablation runs, giving a slightly
different median from the BS sweep. Effects interact and should not be added.

The full path, prefetch-disabled, serial-submission and forced-roundtrip runs
produce identical prediction checksums. JPEG/video encoding and CPU/GPU
preprocessing can change predictions in the other controls. These measurements
assess pipeline throughput; annotation quality needs separate evaluation.
Experimental controls are kept in local patches, outside the production CLI.

Visualization and Tests
-----------------------

Visualization uses the original camera JPEGs referenced in the pickle:

.. code-block:: bash

   python -m applications.auto_labeling.visualize_results \
     --json-path applications/auto_labeling/outputs/inference/detection_results.json \
     --pkl-path /data/nuscenes-mini/stream_petr/nuscenes2d_video_temporal_infos_train.pkl \
     --data-root /data/nuscenes-mini \
     --output-dir applications/auto_labeling/outputs/visualizations --num-samples 4

   bash applications/auto_labeling/tools/run_tests.sh -q

Sampler and JSON tests need pytest and NumPy. Tensor, decoder-boundary and shared
model tests additionally require PyTorch and the demo dependencies. Checkpoints
and generated outputs are ignored by Git.
