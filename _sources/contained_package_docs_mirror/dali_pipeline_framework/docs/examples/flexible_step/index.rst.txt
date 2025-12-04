Flexible Step Implementation
============================

In this section, we will implement a simple image decoding step first and discuss the basic functionality 
needed for a pipeline processing step. Then, we will extend the processing step to a more generally reusable 
version.

.. seealso::

    The processing steps need to provide a fixed interface, which is defined in the base 
    :class:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase` class. Please see 
    :doc:`../../design/pipeline_processing_steps` for more details.

.. toctree::
   :maxdepth: 1

   simple_step
   flexible_step