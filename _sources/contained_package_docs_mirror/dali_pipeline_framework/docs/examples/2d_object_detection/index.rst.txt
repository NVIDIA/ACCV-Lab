2D Object Detection Pipeline
============================

This example consists of the following parts:

.. toctree::
   :maxdepth: 1

   pipeline_setup
   entry_point

.. important::

   You can run the example using the script 
   ``packages/dali_pipeline_framework/examples/example_pipeline_obj_det_2d.py``.

.. important::

   Before running the example, you must prepare the NuScenes 2D image annotations by executing
   the script ``packages/dali_pipeline_framework/examples/prepare_dataset.py``. Set the dataset root
   and version in ``examples/input_definitions.py`` to match your dataset. The script generates
   ``<dataset_root>/<version>/image_annotations.json``, which is required by this pipeline.
