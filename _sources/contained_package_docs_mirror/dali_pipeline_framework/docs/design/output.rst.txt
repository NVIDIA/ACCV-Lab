Output -- DALI Structured Output Iterator
=========================================

In general, the DALI pipeline emits a flat sequence of tensors (or DALI tensor lists). In case of our 
framework, these are the results obtained from calling 
:meth:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_data` on the
:class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object used in the pipeline. 

For complex data formats, a flat list quickly becomes hard to manage. Therefore, we introduce the 
:class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` class, which re-assembles the data
to its original structure.

The :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` is designed to be a drop-in 
replacement for a PyTorch DataLoader. Apart from the re-assembly of the data, this is achieved by:

  - Using the same interface as a PyTorch DataLoader (i.e. the iterator interface)
  - Option to auto-convert the output to a nested dictionary (using 
    :meth:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.to_dictionary` internally)
  - Option to apply a user-defined post-processing function whenever obtaining the data (to perform 
    light-weight steps not possible in the pipeline, e.g. convert certain fields to a type not directly 
    supported by DALI)

.. note::

    The user-defined post-processing in :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` 
    runs in the training thread when data is requested; keep it lightweight and prefer doing work inside the 
    DALI pipeline where possible.

.. note::
  While the :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` class is designed to be a 
  drop-in replacement for a PyTorch DataLoader, there may be issues if the training implementation
  contains checks in the form of ``assert isinstance(iterator_object, DataLoader)``. These checks may be 
  inside dependencies used by the training implementation, and so cannot be changed easily in a clean way. For 
  these cases, the :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` provides
  a :meth:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator.CreateAsDataLoaderObject` method,
  which creates an iterator object masked as a PyTorch DataLoader object, so that these checks pass.

.. _dali-pipeline-framework-external-source-pass-through-note:

.. important::

    In some DALI versions, pass-through outputs from a parallel
    :func:`~nvidia.dali.fn.external_source` can be corrupted when they are returned directly from a pipeline
    using the dynamic executor. For example, the `Known Issues section in the DALI 1.53.0 Release Notes
    <https://docs.nvidia.com/deeplearning/dali/archives/dali_1_53_0/release-notes/index.html#known-issues>`_
    describes the conditions under which this may occur and recommends adding :func:`~nvidia.dali.fn.copy` to avoid returning
    :func:`~nvidia.dali.fn.external_source` outputs directly.

    If a pipeline is affected, enable ``copy_external_source_passthrough_outputs=True`` when constructing
    :class:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition`. This inserts
    :func:`~nvidia.dali.fn.copy` internally before the final flattened output is returned. If no copy selectors
    are provided, all final output data fields are copied. To reduce overhead, limit copying to final output
    fields where the corresponding data passes or may pass through the pipeline unchanged (i.e. without being
    modified by processing steps) by using the ``passthrough_copy_field_names``,
    ``passthrough_copy_field_names_scope_paths``, or ``passthrough_copy_branch_paths`` constructor arguments.

    This package configures DALI :func:`~nvidia.dali.fn.external_source` differently depending on the input
    base class. Inputs derived from :class:`~accvlab.dali_pipeline_framework.inputs.CallableBase` are used in
    per-sample mode, while inputs derived from :class:`~accvlab.dali_pipeline_framework.inputs.IterableBase`
    are used in per-batch mode. Therefore, for callable inputs the single-contiguous-buffer case relevant to
    this workaround is expected only when the pipeline batch size is ``1``. Iterable inputs provide whole
    batches, so pass-through outputs from parallel :func:`~nvidia.dali.fn.external_source` can be affected
    independently of the pipeline batch size.
