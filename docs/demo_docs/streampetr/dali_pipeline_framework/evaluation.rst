Evaluation
==========

This page summarizes the performance impact of using the DALI Pipeline Framework
as a replacement for a conventional PyTorch ``Dataset`` + ``DataLoader`` when training
StreamPETR on NuScenes mini. It also describes a separate data-loading-only evaluation,
which measures data-loader iteration time without model training.

.. seealso::

   For the pipeline and integration changes behind these results, see the
   :doc:`DALI Pipeline Framework demo <index>` and its
   :doc:`data-loader integration tutorial <replacing_data_loader>`.
   The :doc:`Optimization Testing Tools demo <../optim_test_tools/index>` explains the runtime
   measurements and profiling support used in the evaluation.

.. _streampetr-dali-training-experiment:

Training Experiment Setup
-------------------------

Configuration
~~~~~~~~~~~~~

The training is performed for the NuScenes mini dataset, using the StreamPETR model. In order to evaluate
a realistic setup with a high pre-processing overhead, we increase both the image resolution and the batch 
size compared to the original configuration. We use the following configuration:

  - Image size: 1024 × 372
  - Batch size: 8 per GPU

The training evaluation was performed with the standard PyTorch data loader and the data loader built
with the DALI Pipeline Framework on 2 and 8 GPUs. The model implementation is fixed within each comparison.

.. _streampetr-dali-dataset-replication:

Data
~~~~

- NuScenes mini
- The mini training split is too small to populate ordinary sharded sampling with 8 GPUs
  and a batch size of 8 per GPU. For these evaluations, each GPU therefore samples the
  whole dataset instead of a separate shard. The DALI Pipeline Framework comparison and profiling
  configurations enable this behavior for both loaders by setting
  ``data_loading_config.replicate_dataset_per_gpu=True``. This is a custom option added
  to the demo for these evaluations, not a standard NVIDIA DALI or StreamPETR configuration option.

.. important::

   Replication allows performance measurements with a small dataset. Each GPU sees
   the same sample order within a run. The underlying NVIDIA DALI pipeline derives its reproducible
   augmentation seed from the sampling rank, so the provided configurations also use identical
   augmentations across GPUs. PyTorch worker seeds use the real rank, so its augmentations
   can differ across GPUs. Sample order is not guaranteed to match between the PyTorch and
   DALI Pipeline Framework variants.

   DDP, GPU assignment, and validation sharding are unchanged. These runs measure a
   replicated workload; they do not represent training on independent batches across
   GPUs. Disable the option for normal sharded training with enough data.
   With one GPU, the option does not change sampling. See the
   :doc:`data-loader integration tutorial <replacing_data_loader>` for the implementation.

Adaptation to the DALI Pipeline Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Replace the PyTorch ``DataLoader`` by a DALI pipeline
- Code changes are limited to the pipeline setup vs. PyTorch ``Dataset`` and ``DataLoader``
- The training loop implementation remains unchanged

Hardware Setup
~~~~~~~~~~~~~~

.. list-table:: Test Device
   :header-rows: 1

   * - GPU
     - CPU
   * - 8 × NVIDIA A100‑SXM4‑80GB
     - 2 × AMD EPYC 7742 64‑Core Processors

Training Results & Discussion
-----------------------------

Results
~~~~~~~~

The numbers report average run time per batch after the warm‑up phase. Each run performs
1,000 iterations; the first 50 are warm-up iterations and the remaining 950 are measured.

.. list-table:: Runtime and CPU Usage
   :header-rows: 1

   * - Method
     - Runtime [ms] (2‑GPU)
     - CPU usage [%] (2‑GPU)
     - Runtime [ms] (8‑GPU)
     - CPU usage [%] (8‑GPU)
   * - **Reference (PyTorch DataLoader)**
     - 917
     - 3.4
     - 1027
     - 12.8
   * - **DALI Pipeline Framework**
     - 813
     - 1.4
     - 844
     - 5.5
   * - **Speedup / Savings**
     - × 1.13
     - 57%
     - × 1.22
     - 57%

Discussion
~~~~~~~~~~

The results show that the DALI Pipeline Framework leads to a speedup of **× 1.13** for the 2-GPU configuration
and **× 1.22** for the 8-GPU configuration, with the speedup being larger for the 8-GPU configuration.
The CPU usage is reduced by around **57%** in both configurations. Note that for both the
DALI Pipeline Framework and the reference implementation, the runtime increases for the
8-GPU configuration compared to the 2-GPU configuration. However, the increase is much smaller
for the framework-backed pipeline, indicating that while the CPU is not fully utilized in the
reference approach, there are already bottlenecks present which can be overcome by the
framework-backed pipeline.

.. _streampetr-dali-load-only:

Data Loading Only
-----------------

This experiment compares the PyTorch data loader and the data loader built with the DALI Pipeline
Framework without running model forward or backward passes or optimizer steps. The
``DataPipelineRuntimeEvalRunner`` repeatedly fetches the next batch, measuring data loading and
preprocessing separately from model training.
Pipeline construction is outside the timed loop.

Both use NuScenes mini, one GPU, a batch size of 8, and an output image size of 1024 × 372,
matching the per-GPU batch size and preprocessing settings of the training experiments above.
Random augmentations are enabled. The PyTorch reference uses eight DataLoader worker
subprocesses (``workers_per_gpu=8``); it does not use a single worker. The DALI Pipeline Framework variant
replaces the PyTorch DataLoader and uses one external-source Python worker
(``py_num_workers=1``) plus four NVIDIA DALI CPU/mixed-operator threads (``num_threads=4``).

Each run fetches 1,000 batches, excluding the first 300 as warm-up and averaging the runtime over
the remaining 700 batches. The runner prints
``Average runtime`` in seconds per batch; multiply by 1,000 to report milliseconds below.

Unlike the 2-GPU and 8-GPU training experiments, this comparison uses a single GPU and performs
no model training. Report its runtimes separately from the training runtimes.

.. important::

   This experiment is measuring the data-loading performance in isolation. It is not a prediction of training
   speedup. During training, this work overlaps with the model computations, so faster data loading does not
   translate directly into an equivalent reduction in training time.

   The benefit during training depends on the available CPU and GPU resources and how they are
   shared. For example, increasing the number of workers per GPU can help keep the GPU supplied
   with data, but too many workers across GPUs can compete for CPU resources and slow data loading.
   Use experiments with the full training pipeline to assess the benefit for a particular use case (such as the
   :ref:`experiment described above <streampetr-dali-training-experiment>`).

.. list-table:: Data-loading runtime (1 GPU)
   :header-rows: 1

   * - Method
     - Average runtime [ms/batch]
     - Average CPU usage [%]
   * - **Reference (PyTorch DataLoader)**
     - 416
     - 3.2
   * - **DALI Pipeline Framework**
     - 181
     - 1.3
   * - **Speedup / Savings**
     - × 2.30
     - 59%

The DALI Pipeline Framework reduces isolated data-loading time by **× 2.30** and CPU usage by
approximately **59%** for this workload. Pipeline construction is excluded from these
steady-state timings. In this run, DALI pipeline creation took 723 seconds and pipeline
building took a further 5.8 seconds.

How to Run
----------

.. note::

  This section assumes that you are running the evaluation in the Docker container set up as described in the
  :doc:`StreamPETR Docker setup <../docker_setup>`, with the StreamPETR implementation
  located at ``/streampetr``. Configuration paths below are relative to
  ``demos/streampetr/projects/configs/StreamPETR/``.

Running the Evaluated Training Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training evaluation used the following standard configuration pairs. Within each
pair, the model implementation is fixed and only the data loader changes.

.. list-table:: Evaluated DALI Pipeline Framework training configurations
   :header-rows: 1
   :widths: 20 40 40

   * - GPU count
     - PyTorch reference
     - DALI Pipeline Framework
   * - 2
     - ``Tests/dali_ref_2_gpu.py``
     - ``Tests/dali_test_2_gpu.py``
   * - 8
     - ``Tests/dali_ref_8_gpu.py``
     - ``Tests/dali_test_8_gpu.py``

Run the configurations individually with the distributed training helper:

.. code-block:: bash

    cd /streampetr/tools

    # 2-GPU comparison
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_2_gpu.py 2
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_2_gpu.py 2

    # 8-GPU comparison
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_8_gpu.py 8
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_8_gpu.py 8

Alternatively, run all four evaluated training configurations and save their logs with:

.. code-block:: bash

    cd /streampetr/tools
    bash run_dali_eval.sh

The ``demos/streampetr/tools/run_dali_eval.sh`` script stores the results in the
``res_dali_eval`` folder in the current working directory.

Running the Evaluated Data-Loading-Only Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The separate data-loading-only results use ``DALILoadOnly/ref_load_only.py`` for the
PyTorch reference and ``DALILoadOnly/dali_load_only.py`` for the DALI Pipeline Framework
variant. Run them individually with:

.. code-block:: bash

    cd /streampetr/tools

    python train.py /streampetr/projects/configs/StreamPETR/DALILoadOnly/ref_load_only.py --no-validate
    python train.py /streampetr/projects/configs/StreamPETR/DALILoadOnly/dali_load_only.py --no-validate

Alternatively, run both configurations and save their stdout and stderr logs with:

.. code-block:: bash

    cd /streampetr/tools
    bash run_dali_load_only.sh

The script writes the logs to ``eval_res_dali_load_only`` in the current working directory.

Additional Training Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following configurations are included for further experiments but were not used in
the evaluation described above:

.. list-table:: Additional DALI Pipeline Framework training configurations
   :header-rows: 1
   :widths: 30 45 25

   * - Purpose
     - Configuration files
     - Notes
   * - Single-GPU standard comparison
     - ``Tests/dali_ref_single_gpu.py`` and ``Tests/dali_test_single_gpu.py``
     - A paired comparison with the PyTorch reference.
   * - Batching Helpers enabled
     - Matching ``Tests/dali_ref_bh_{2,8}_gpu.py`` and
       ``Tests/dali_test_bh_{2,8}_gpu.py`` pairs.
     - Both sides use the batched StreamPETR and Focal heads; only ``test_bh`` enables the
       DALI Pipeline Framework data loader.

The Batching-Helpers-enabled configurations answer a different question from the standard
comparison: they measure the effect of replacing the loader after batching optimizations are
already enabled. Do not compare a ``*_bh_*`` configuration with a standard configuration and
attribute the full difference to the DALI Pipeline Framework.

Use ``train.py`` for a single GPU and the distributed training helper ``dist_train.sh``
for the multi-GPU configurations:

.. code-block:: bash

    cd /streampetr/tools

    # Single-GPU standard comparison
    CUDA_VISIBLE_DEVICES=0 python train.py \
      /streampetr/projects/configs/StreamPETR/Tests/dali_ref_single_gpu.py --no-validate
    CUDA_VISIBLE_DEVICES=0 python train.py \
      /streampetr/projects/configs/StreamPETR/Tests/dali_test_single_gpu.py --no-validate

    # 2-GPU comparison with Batching Helpers enabled on both sides
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_bh_2_gpu.py 2
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_bh_2_gpu.py 2

    # 8-GPU comparison with Batching Helpers enabled on both sides
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_ref_bh_8_gpu.py 8
    bash dist_train.sh /streampetr/projects/configs/StreamPETR/Tests/dali_test_bh_8_gpu.py 8
