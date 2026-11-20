Auto Labeling
=============

This demo demonstrates how to use ACCV-Lab to accelerate an auto-labeling
pipeline, with a concrete example based on
`StreamPETR <https://arxiv.org/abs/2303.11926>`_.

The example processes calibrated six-camera driving videos and exports object
classes, 3D boxes, velocities and confidence scores as JSON in the LiDAR
coordinate frame. With a trained checkpoint, these predictions provide initial
annotations for review, correction and perception dataset preparation.

Environment and Weights
-----------------------

Use the :doc:`StreamPETR environment <../../demo_docs/streampetr/docker_setup>`
with its patched MMCV/MMDetection3D dependencies. Install ACCV-Lab's
``on_demand_video_decoder``, ``optim_test_tools``, ``batching_helpers`` and
``draw_heatmap`` packages following the
:doc:`installation guide <../../guides/INSTALLATION_GUIDE>`.

For annotation, use a checkpoint you have trained with the selected model
configuration. You may also consult the
`official StreamPETR project <https://github.com/exiawsh/StreamPETR>`_
for checkpoints that match your requirements and configuration.

Omitting ``--checkpoint`` uses standard random initialization with a fixed seed
of 0 for performance testing. It does not download or load pretrained weights,
and the untrained predictions are not meaningful annotations. Initialization
runs once before inference; no checkpoint file needs to be generated or saved.

The included config uses an R50 backbone with 300 learned queries, 128 propagated
queries and 512 memory entries. To load your own checkpoint, add
``--checkpoint /path/to/your_checkpoint.pth``. Loading is strict: incompatible
or missing files fail rather than falling back to random initialization.
Only load trusted checkpoints and pickles.

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

From the repository root, run with random initialization for performance testing:

.. code-block:: bash

   python -m demos.auto_labeling.inference \
     --pkl-path /data/nuscenes-mini/stream_petr/nuscenes2d_video_temporal_infos_train.pkl \
     --data-root /data/nuscenes-mini \
     --video-base-path /data/nuscenes-mini/video_samples \
     --batch-size 2 --device cuda:0 \
     --output-dir demos/auto_labeling/outputs/inference

``demos/auto_labeling/tools/run_inference.sh`` accepts the same options
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

Measurements use one NVIDIA H100 GPU, 32 allocated CPUs, and StreamPETR R50
with standard random initialization (seed 0) in a SGLang container
(PyTorch 2.8.0, CUDA 12.9). All variants use the same initialization and model
configuration. A sample is one timestamp with six camera views. Batch size (BS)
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

.. Data source: demos/auto_labeling/evaluation_results/throughput_by_bs.csv

.. figure:: _generated/throughput_by_bs.svg
   :alt: CPU JPEG and ACCV-Lab throughput by batch size for six-camera 1600 × 900 HEVC input on one H100, with ACCV-Lab reaching 111.86 samples/s at BS=64.
   :width: 100%

   Median throughput with minimum–maximum error bars over three runs.
   Green squares represent ACCV-Lab; blue circles represent parallel CPU JPEG.
   Both pipelines use the same GPU model inference and GPU preprocessing.
   Only fully occupied batches are plotted.

The highest measured throughput is **111.86 samples/s at BS=64**, versus
**87.40 samples/s** for parallel CPU JPEG at the same BS
(**1.28×**). This is the best value in the tested
range of 1–64 parallel scene streams. Only fully occupied batches are plotted;
partial-batch refinement results remain in the raw measurements.

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
     - 112.65 (111.64–112.81)
     - 0.0%
     - 1.00×
   * - No prefetch
     - 97.87 (97.41–97.88)
     - 13.1%
     - 1.15×
   * - Serial decode
     - 38.55 (38.13–38.68)
     - 65.8%
     - 2.92×
   * - Host round trip
     - 33.48 (33.42–33.84)
     - 70.3%
     - 3.36×
   * - CPU preprocessing
     - 8.71 (8.00–8.81)
     - 92.3%
     - 12.94×
   * - CPU video decode
     - 30.89 (30.67–31.17)
     - 72.6%
     - 3.65×
   * - CPU JPEG
     - 87.63 (87.56–87.69)
     - 22.2%
     - 1.29×

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

   python -m demos.auto_labeling.visualize_results \
     --json-path demos/auto_labeling/outputs/inference/detection_results.json \
     --pkl-path /data/nuscenes-mini/stream_petr/nuscenes2d_video_temporal_infos_train.pkl \
     --data-root /data/nuscenes-mini \
     --output-dir demos/auto_labeling/outputs/visualizations --num-samples 4

   bash demos/auto_labeling/tools/run_tests.sh -q

Sampler and JSON tests need pytest and NumPy. Tensor, decoder-boundary and shared
model tests additionally require PyTorch and the demo dependencies. Checkpoints
and generated outputs are ignored by Git.
