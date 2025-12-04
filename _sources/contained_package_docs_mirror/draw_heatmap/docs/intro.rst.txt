Introduction
============

Overview
--------

In the realm of object detection model training, there is usually a centerness loss term, which measures how 
close the center of predicted bounding box is to that of the ground truth one. 
`GaussianFocalLoss <https://ieeexplore.ieee.org/abstract/document/9776580>`_ is employed to calculate this 
centerness loss term, as the following formula shows:


.. math::

   loss(pred, target) = - \left( 1 - p_t \right)^\alpha \cdot log(p_t) \cdot g_t,  
   where
   \left\{
   \begin{aligned}
   p_t & = pred, \ g_t = 1, \ when \ target = 1 \\
   p_t & = 1 - pred, \ g_t = (1 - target)^\gamma, \ when \ target \ != 1
   \end{aligned}
   \right.

Both `pred` and `target` tensors are of shape ``(num_heatmaps, height, width)`` and their values are within 
the range ``(0, 1)`` (open interval for `pred`, and closed interval for `target`). The `target` tensor is 
actually a Gaussian heatmap which is drawn based on ground truth bounding boxes using a Gaussian 2d kernel.

.. image:: images/bboxes_to_heatmap.png
   :alt: bboxes to heatmap
   :align: center

The existing implementation of drawing Gaussian heatmap (e.g., 
`mmdet.models.utils.gaussian_target <https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/utils/gaussian_target.html>`_) 
involves CPU operation and handles each bounding box in each heatmap sequentially, which is inefficient and of 
low GPU utilization. In this repo, we implement a GPU kernel for drawing Gaussian heatmaps and port it as a 
PyTorch GPU operator, which calculates the `target` tensor based on centers and radii of bounding boxes. The 
GPU kernel can batchify the operation and achieve significant speedup compared to the existing implementation.

Benchmark
---------

Python Wrapper
~~~~~~~~~~~~~~
This package provides convenient Python wrappers for our GPU kernel. In this section, we benchmark the 
performance of the python wrapper and PyTorch implementation. 
Generally, this package provides **two implementations** for drawing the heatmaps:

- :func:`~accvlab.draw_heatmap.draw_heatmap`: this implementation is designed for the **concatenated** input 
  format.

- :func:`~accvlab.draw_heatmap.draw_heatmap_batched`: this implementation is designed for the **batched** 
  input format.

The benchmark results are shown in the following table.

.. note::
   The shape of heatmap is ``(batch_size, height, width)``, and the performance is measured on a single 
   NVIDIA A100 GPU.

+----------------------+----------------+------------------+------------------+
| Implementation       | Heatmap shape  | Performance      | Speedup          |
+======================+================+==================+==================+
| PyTorch              | 48x20x50       | 201.1 ms         | ---              |
+----------------------+----------------+------------------+------------------+
| draw_heatmap (concat)| 48x20x50       | 0.0482 ms        | 4189.42x         |
+----------------------+----------------+------------------+------------------+
| draw_heatmap_batched | 48x20x50       | 0.0366 ms        | 5494.32x         |
+----------------------+----------------+------------------+------------------+

Other than that, this package also provides a classwise implementation for drawing the heatmaps, which is to 
draw one heatmap image for each class. This is only directly supported by 
:func:`~accvlab.draw_heatmap.draw_heatmap_batched`.

The benchmark results for this implementation are shown in the following table.

.. note::
   The shape of heatmap is ``(batch_size, num_classes, height, width)``, and the performance is measured on a 
   single NVIDIA A100 GPU.

+----------------------+----------------+------------------+------------------+
| Implementation       | Heatmap shape  | Performance      | Speedup          |
+======================+================+==================+==================+
| PyTorch              | 48x20x20x50    | 245.1 ms         | ---              |
+----------------------+----------------+------------------+------------------+
| draw_heatmap_batched | 48x20x20x50    | 0.059 ms         | 4154.24x         |
+----------------------+----------------+------------------+------------------+

.. note::
   While the classwise implementation is only directly supported by 
   :func:`~accvlab.draw_heatmap.draw_heatmap_batched`, it can
   be also achieved in :func:`~accvlab.draw_heatmap.draw_heatmap`, as here, the indices of images to draw into 
   are set manually for each bounding box, so that the indices could be set up to map to both different samples 
   and different classes within a sample.

C++ Benchmark Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a C++ benchmark for directly measuring the performance of the C++ implementation without
the PyTorch wrapper. It can be built by running the following commands:

.. code-block:: bash

   cd packages/draw_heatmap/benchmark_cpp
   mkdir build
   cd build
   cmake ..
   make

Then, the performance measurements can be obtained by running the following commands.

.. code-block:: bash

   ./benchmark_flattened
   ./benchmark_batched
   ./benchmark_batched_classwise


Installation
------------

This package is installed as part of the `accvlab` package. Please refer to the 
:doc:`Installation Guide <../../../guides/INSTALLATION_GUIDE>` for more details.

.. important::
   This package has a runtime dependency on the 
   :doc:`accvlab.batching_helpers <../../batching_helpers/docs/introduction>` package. Please install it 
   before installing this package.