Pipeline & Processing Steps
===========================

Processing Steps
----------------

Data processing occurs through a sequence of instances of classed derived from 
:class:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase`. The instances are defined and configured 
before the pipeline is constructed. Once part of the pipeline, the steps are applied sequentially 
to training samples returned by input callable/iterable instances (see :doc:`input`), or by previous 
processing steps. The results of the last processing step are output from the pipeline.

Processing Step Design
^^^^^^^^^^^^^^^^^^^^^^

Data Access and Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

- By default, processing steps can access and modify any part of the input data
- Available steps are designed for broad compatibility across input data formats by not assuming a fixed 
  data format (except where necessary). Instead, they use the 
  :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` class and its utilities to

  - Represent the data in a unified way (i.e. the input is always a single 
    :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` instance)
  - Automatically discover and process all relevant fields in the data where this makes sense. For example, 
    the :class:`~accvlab.dali_pipeline_framework.processing_steps.ImageDecoder` step automatically 
    discovers and processes all relevant images, regardless of quantity or location in the data 
    structure (finding all images by name, where the name is part of the configuration).
  - Resolve paths to individual data fields when needed (e.g. see 
    :meth:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_item_in_path`), which can be used 
    both for accessing manually specified paths as well as paths of auto-discovered data fields.

- For selective processing, wrapper classes (see 
  :class:`~accvlab.dali_pipeline_framework.processing_steps.GroupToApplyToSelectedStepBase`) can limit which data a step 
  processes, separating data selection logic from processing implementation. This facilitates re-use of both 
  processing and data selection implementations across different use-cases.


Consistent & Independent Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some processing steps, the question arises whether all the processed data should be processed consistently 
(e.g. all contained images should be augmented consistently with the same transformation) or independently 
(e.g. different augmentations for different images), with the answer depending on the use-case.

By default, the processing steps are designed to process all data consistently. In combination with the 
:class:`~accvlab.dali_pipeline_framework.processing_steps.GroupToApplyToSelectedStepBase`-derived wrapper classes, this 
allows for fine-grained control over which data is processed consistently (e.g. consistent randomization of 
camera image, segmentation map & projection matrix), and which data is processed independently (e.g. 
different augmentations for different cameras). Please see the
:ref:`corresponding note <pipeline_step_base_access_modifier_wrapper_discussion>` in the documentation 
of :class:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase` for a more detailed discussion of this 
approach and how it is used to group parts of the input data for consistent processing.


Format Validation and Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each processing step must derive from 
:class:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase` and override the following methods:

- :meth:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase.check_input_data_format_and_set_output_data_format`
  to implement:

  - **Validation of Input Format**: Check that required data exists and no conflicts occur (e.g., no existing 
    fields at the same location/name that the step would overwrite).
  - **Inference of Output Format**: Generate a blueprint of the expected output data structure based on the 
    input blueprint.

- :meth:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase._process` to implement the actual 
  processing.

.. note::

  The consistency between the output format as declared by 
  :meth:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase.check_input_data_format_and_set_output_data_format`
  and the actual processing output 
  :meth:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase._process`
  is verified automatically by the base class implementation.

.. important::

  The validation of the format as well as the check for consistency of the output with the declared output 
  format occurs during pipeline construction (DALI graph construction), ensuring that there is no runtime 
  performance impact when using the pipeline.

Pipeline
--------

The input callable/iterable (see :doc:`input`) and the processing steps can be combined into a pipeline. This 
is done by constructing an instance of the 
:class:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition` class (see documentation of the class), 
followed by calling the 
:meth:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition.get_dali_pipeline` to obtain the 
DALI pipeline object. Please see the documentation of the class for more details.

Note that the creation of the DALI pipeline is a 2-step process:

  1. **Creating the pipeline definition**:

    - This is done by constructing an instance of the 
      :class:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition` class
    - The pipeline structure (i.e. input callable/iterable, processing steps) is defined
    - The underlying DALI pipeline is not created yet
    - However, some functionality is available, e.g. the input data format can be obtained by as the
      :attr:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition.input_data_structure` attribute and
      the output data format can be obtained by calling the
      :meth:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition.check_and_get_output_data_structure` 
      method.

  2. **Creating the actual DALI pipeline**:
  
    - This is done by calling the 
      :meth:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition.get_dali_pipeline` method
    - Configuration of the underlying DALI pipeline is done at this point (see method documentation)

The :doc:`../examples` section contains code examples on how to set up a pipeline.
