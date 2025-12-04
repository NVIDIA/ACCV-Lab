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
