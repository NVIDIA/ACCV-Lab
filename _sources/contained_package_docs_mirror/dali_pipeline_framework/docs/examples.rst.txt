Examples
========

The examples are provided in the ``packages/dali_pipeline_framework/examples/`` directory.

Pipeline Setup Examples
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 15, 55, 30
   :class: no-scroll

   * - Example
     - Description
     - Entry Point Script
   * - :doc:`examples/2d_object_detection/index`
     - Shows how to set up a pipeline for 2D object detection. This example also demonstrates the flexibility
       of the pipeline framework by allowing the switch between single-camera and multi-camera data (adding a 
       hierarchy level to the input data format), where no changes are needed in the pipeline implementation.
     - ``packages/dali_pipeline_framework/examples/example_pipeline_obj_det_2d.py``
   * - :doc:`examples/stream_petr/index`
     - Demonstrates a pipeline for StreamPETR, a 3D object detection model which uses multi-camera data. Note
       that this pipeline can be directly used as a drop-in replacement for a PyTorch DataLoader in 
       StreamPETR training.
     - ``packages/dali_pipeline_framework/examples/example_pipeline_stream_petr.py``
   * - :doc:`examples/flexible_step/index`
     - Demonstrates how to implement a flexible step which can be used to process data in a flexible way.
       The :class:`~accvlab.dali_pipeline_framework.processing_steps.ImageDecoder` class is implemented here,
       starting from a simple step with hard-coded input data format, and then modified to achieve the 
       flexibility of automatically finding and processing all relevant images in the input data.
     - (No script)

.. toctree::
   :maxdepth: 1
   :hidden:

   examples/2d_object_detection/index
   examples/stream_petr/index
   examples/flexible_step/index

Data Loaders for NuScenes
-------------------------

Both of the pipeline examples above rely use the NuScenes dataset. 
To enable the use of the dataset in the corresponding pipelines, related data loaders are provided.
These data loaders share a large part of the functionality (both load the same data from NuScenes, but
process and output it in a different format). 
Therefore, they share a large part of the implementation by
using helper classes to read the data and perform the necessary conversions. See
:doc:`examples/use_case_specific/nuscenes_data_loader`.

.. toctree::
   :maxdepth: 1
   :hidden:

   examples/use_case_specific/nuscenes_data_loader

