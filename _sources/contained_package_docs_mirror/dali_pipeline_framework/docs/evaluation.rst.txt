Evaluation
==========

This page summarizes the performance impact of using the DALI Pipeline Framework
as a replacement for a conventional PyTorch ``Dataset`` + ``DataLoader`` when training
StreamPETR on NuScenes mini.

Experiment Setup
----------------

Configuration
~~~~~~~~~~~~~

The training is performed for the NuScenes mini dataset, using the StreamPETR model. In order to evaluate
a realistic setup with a high pre-processing overhead, we increase both the image resolution and the batch 
size compared to the original configuration. We use the following configuration:

  - Image size: 1024 × 372
  - Batch size: 8 per GPU

Data
~~~~

- NuScenes mini
- For the multi‑GPU training, each GPU re‑uses the whole dataset instead of sharding
  (NuScenes mini is otherwise too small)

Adaptation to DALI
~~~~~~~~~~~~~~~~~~

- Replace the PyTorch ``DataLoader`` by a DALI pipeline
- Code changes are limited to the pipeline setup vs. PyTorch ``Dataset`` and ``DataLoader``
- The training loop implementation remains unchanged

.. note::

  We are planning to add a demo for the DALI Pipeline Framework package in the future, including the 
  implementation of the experiments performed in this evaluation.

Hardware Setup
~~~~~~~~~~~~~~

.. list-table:: Test Device
   :header-rows: 1

   * - GPU
     - CPU
   * - 8 × NVIDIA A100‑SXM4‑80GB
     - 2 × AMD EPYC 7742 64‑Core Processors

Results & Discussion
--------------------

Results
~~~~~~~~

The numbers report average run time per batch after the warm‑up phase.

.. list-table:: Runtime and CPU Usage
   :header-rows: 1

   * - Method
     - Runtime [ms] (2‑GPU)
     - CPU usage [%] (2‑GPU)
     - Runtime [ms] (8‑GPU)
     - CPU usage [%] (8‑GPU)
   * - **Reference (PyTorch DataLoader)**
     - 935
     - 3.3
     - 1110
     - 12.3
   * - **DALI pipeline**
     - 829
     - 1.5
     - 868
     - 5.4
   * - **Speedup / Savings**
     - × 1.13
     - 55%
     - × 1.28
     - 56%

Discussion
~~~~~~~~~~

The results show that the DALI pipeline framework leads to a speedup of **× 1.13** for the 2-GPU configuration
and **× 1.28** for the 8-GPU configuration, with the speedup being larger for the 8-GPU configuration.
The CPU usage is reduced by around **55%** in both configurations. Note that for both DALI And the reference 
implementation, the runtime increases for the 8-GPU configuration compared to the 2-GPU configuration.
However, the increase is much smaller for the DALI pipeline, indicating that while the CPU is not fully
utilized in the reference approach, there are already bottlenecks present which can be overcome by the DALI 
pipeline.

