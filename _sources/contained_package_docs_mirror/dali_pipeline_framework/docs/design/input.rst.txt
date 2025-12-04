Input -- Passing Data to the Pipeline
=====================================

Input Callables & Iterables
---------------------------

The input data for the pipeline is provided by a callable or iterable class, i.e. an object which implements 
the :class:`~accvlab.dali_pipeline_framework.inputs.IterableBase` or :class:`~accvlab.dali_pipeline_framework.inputs.CallableBase` 
interface. These classes are expected to provide the data & format blueprint as follows:

  - Providing the data format blueprint (also see :doc:`sample_data_group`):
  
      - Described by a :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` blueprint (i.e. object with the 
        data format set up, but without any actual data)
      - Needs to be returned by the overriden 
        :meth:`CallableBase.used_sample_data_structure() <accvlab.dali_pipeline_framework.inputs.CallableBase.used_sample_data_structure>`
        or 
        :meth:`IterableBase.used_sample_data_structure() <accvlab.dali_pipeline_framework.inputs.IterableBase.used_sample_data_structure>`
  
  - Providing the actual data:
  
      - Data is output from the input callable/iterable as a flat sequence of data fields, as can be obtained 
        by calling :meth:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.get_data` on a 
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object where the data format corresponds to 
        the provided blueprint
      - Note that while for both input callable and iterable, the data structure is the same (as described by 
        the blueprint), the actual data fields (returned as a flat sequence) are different in that:
        
        - For input callables, each data field corresponds to one sample
        - For input iterables, each data field corresponds to one batch

      - The data needs to be returned by the overriden 
        :meth:`CallableBase.__call__() <accvlab.dali_pipeline_framework.inputs.CallableBase.__call__>` or 
        :meth:`IterableBase.__next__() <accvlab.dali_pipeline_framework.inputs.IterableBase.__next__>`

.. note::

    The flattened data sequence returned by the input callable/iterable is converted back into the 
    structured format automatically by the pipeline using the provided blueprint, so that the user does not 
    need to worry about the conversion and can assume that 
    :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` objects are used throughout the 
    pipeline.

    
The blueprint provided by the input callable/iterable is used by the pipeline to obtain the data format 
after each processing step and check for compatibility (see :doc:`pipeline_processing_steps`). 

.. note::

  The data format for the output of the pipeline is in turn needed to auto-convert the output 
  from the flat sequence back into the structured format, e.g. using the 
  :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` (see :doc:`output`).

.. seealso::

  Note that there are pre-defined input callables/iterables which cover a wide range of use-cases, so that 
  often there is no need to implement a custom callable/iterable. These are:

    - :class:`~accvlab.dali_pipeline_framework.inputs.ShuffledShardedInputCallable`
    - :class:`~accvlab.dali_pipeline_framework.inputs.SamplerInputIterable`
    - :class:`~accvlab.dali_pipeline_framework.inputs.SamplerInputCallable`

  The pre-defined input callables/iterables are designed to be agnostic of the actual dataset, which is 
  interfaced through a :class:`~accvlab.dali_pipeline_framework.inputs.DataProvider` class (see section 
  below). 
  The input data provider is expected to provide the data in the structured format (i.e. as a 
  :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object) one sample at a time,
  and the conversion to the flattened format (and batching of samples if needed) is performed
  internally by the input callable/iterable. In this way, the conversion is transparent to the user of the 
  package when using the included input callables/iterables.

Input Data Provider
-------------------

The pre-defined input callables/iterables are generic, and do not assume a specific dataset. Dataset- (and 
use-case-) specific functionality is provided by a class implementing the 
:class:`~accvlab.dali_pipeline_framework.inputs.DataProvider` interface. Such data providers can be used by 
the input callable/iterable to read the actual data from the dataset. The task of data reading is split as 
follows:

  - Input callable/iterable:

    - Define which sample(s) to load
    - Use the data provider to load the actual data
    - Convert the data to the flat format which needs to be returned by the input callable/iterable
    - When outputting the data format (
      :meth:`~accvlab.dali_pipeline_framework.inputs.CallableBase.used_sample_data_structure`
      or :meth:`~accvlab.dali_pipeline_framework.inputs.IterableBase.used_sample_data_structure`), this information
      is internally obtained from the data provider
    - Similarly, the length of the dataset is internally derived from the data provider (but may be modified, 
      e.g. by sharding, dropping of samples, converting to number of batches, etc.)

  - Data provider:

    - Given a sample index, return the corresponding sample data (see 
      :meth:`~accvlab.dali_pipeline_framework.inputs.DataProvider.get_data`)
    - Define the data format for the sample data (see 
      :meth:`~accvlab.dali_pipeline_framework.inputs.DataProvider.sample_data_structure`)
    - Provide the number of samples in the dataset (see 
      :meth:`~accvlab.dali_pipeline_framework.inputs.DataProvider.get_number_of_samples`)

This means that while the pre-defined input callables/iterables are a fixed and re-usable part of the package, 
the data provider is specific and needs to be implemented by the user of the package.

.. seealso::

  Note that there are data providers for the NuScenes dataset in the examples folder of the package (
  ``packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/data_loading``). They can be used 
  as a reference for implementing data providers for other datasets. They do not read all the data available
  in NuScenes (e.g. lidar point clouds), but rather focus on the data which is needed for the use cases
  at hand. Depending on your use-case, it may be possible to use them as is, or at least as a starting point
  for a customized implementation.

  Please also see the :doc:`../examples/use_case_specific/nuscenes_data_loader` page for more details on 
  the design of the data loaders. 

  Also, note that some re-usability between use-cases is possible by implementing common functionality which
  can be used by different data provides. This is also discussed in the 
  :doc:`../examples/use_case_specific/nuscenes_data_loader` page.
