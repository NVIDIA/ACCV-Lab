Main API Reference
==================

.. note::
   This page contains the documentation of the main API, which is the API that is used to create a DALI 
   pipeline from a set of available processing steps, as well as the API for the data loading and data 
   provider classes.

   It also contains the documentation for the base classes needed to implement custom processing steps.

   However, it does not contain the documentation of some helpers which are used internally, and can be useful 
   when implementing custom processing steps. For more details on those, see the :doc:`additional_api` page.

.. autosummary::
   accvlab.dali_pipeline_framework.inputs
   accvlab.dali_pipeline_framework.pipeline
   accvlab.dali_pipeline_framework.processing_steps

.. toctree::
   :maxdepth: 2
   :hidden:

   inputs
   pipeline
   processing_steps

.. seealso::
   While the API documentation describes the API of the package, there are some use-case dependent helpers
   which are not part of the API, but are implemented as part of the examples. While these helpers are 
   designed for a specific use-case (e.g. data-set, training implementation etc.), they can be potentially 
   useful for similar use-cases.

   For the NuScenes data loader, which is used in the examples, see :doc:`../examples/use_case_specific/nuscenes_data_loader`.
