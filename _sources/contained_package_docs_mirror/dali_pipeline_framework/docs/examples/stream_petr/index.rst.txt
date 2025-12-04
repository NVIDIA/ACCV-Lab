StreamPETR Pipeline
===================

This example consists of the following parts:

.. toctree::
   :maxdepth: 1

   pipeline_setup
   custom_processing_step
   custom_post_processing
   entry_point

.. important::

   You can run the example using the script 
   ``packages/dali_pipeline_framework/examples/example_pipeline_stream_petr.py``.

On the first run, the reader generates the projected bounding-box annotations from the
standard NuScenes metadata and stores them in ``<dataset_root>/<version>_preproc_streampetr.pkl``.
The dataset root must be writable when this cache is created. Later runs reuse the cache.
This pipeline does not require ``image_annotations.json`` or the ``prepare_dataset.py`` export step.
