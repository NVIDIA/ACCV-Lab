Simple Pipeline (Introductory Example)
======================================

This example demonstrates the basic building blocks of the DALI pipeline framework using
**synthetic data** -- no external datasets are required to run it. It is intended as a starting
point for understanding the core design before diving into the more complex examples
(:doc:`../2d_object_detection/index`, :doc:`../stream_petr/index`).

The use of synthetic data keeps the example self-contained while still demonstrating the important concepts. The included code 
snippets/files are best read in order: data provider, pipeline setup, and then entry point.

Concepts Covered
----------------

- **Sample data structure**: Defining a hierarchical
  :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` blueprint representing the training data format. In this 
  example the blueprint contains a ``cameras`` array with per-camera ``image`` and ``annotation/label`` fields, as well as 
  sample-level ``scene_label`` and ``scene_label_as_str`` fields. For details on the blueprint concept and the 
  :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` class in general, please also see 
  :doc:`../../design/sample_data_group`.
- **Data provider**: Subclassing :class:`~accvlab.dali_pipeline_framework.inputs.DataProvider` to provide the defined data 
  structure blueprint as well as individual training samples matching this structure. In this example, those samples are generated
  synthetically.
- **Input callable**: Wrapping the data provider with
  :class:`~accvlab.dali_pipeline_framework.inputs.ShuffledShardedInputCallable` to provide shuffled samples to
  the pipeline.
- **Processing steps**: Chaining built-in processing steps
  (:class:`~accvlab.dali_pipeline_framework.processing_steps.PhotoMetricDistorter`,
  :class:`~accvlab.dali_pipeline_framework.processing_steps.ImageRange01Normalizer`,
  :class:`~accvlab.dali_pipeline_framework.processing_steps.AnnotationElementConditionEval`) to define the processing pipeline.
- **Access modifier wrappers**: Applying photometric augmentation **independently per camera** via
  :class:`~accvlab.dali_pipeline_framework.processing_steps.DataGroupArrayInPathElementsAppliedStep`.
- **Pipeline assembly**: Creating a :class:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition`, building
  the DALI pipeline, and iterating batches through
  :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` (which returns nested dicts).
- **Output handling**: Obtaining the output (which is dumped using :class:`~accvlab.optim_test_tools.TensorDumper` in this 
  simple example and would be used for training in a real-world use-case).

.. important::

   You can run the example using the script
   ``packages/dali_pipeline_framework/examples/simple_full_pipeline/run_example.py``.

   No dataset preparation is needed -- all data is generated synthetically.

.. seealso::

   - :doc:`../../design` for the overall pipeline framework design and a more detailed discussion on the individual concepts.
   - :doc:`../2d_object_detection/index` and :doc:`../stream_petr/index` for full-scale pipeline
     examples using NuScenes data.
   - :doc:`../flexible_step/index` for a tutorial on implementing custom processing steps.

Data Provider
-------------

The data provider is the boundary between use-case-specific data loading and the reusable pipeline framework.
It defines the sample contract once as a ``SampleDataGroup`` blueprint, then returns samples that follow this
contract. When designing a pipeline for a specific use-case, this is the place to decide the input data format
and to implement the data loading logic.

.. note::

  The output data format for the overall pipeline is a consequence of the input data format and the 
  processing steps applied to it, so that it does not need to be defined explicitly.

.. note-literalinclude:: ../../../examples/simple_full_pipeline/simple_data_provider.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: packages/dali_pipeline_framework/examples/simple_full_pipeline/simple_data_provider.py
   :name: simple_data_provider

Pipeline Setup
--------------

The pipeline setup connects the data provider to the pipeline, composes the processing steps for the pipeline,
and wraps the resulting DALI pipeline as a structured iterator. In this example, it also demonstrates how an
access modifier wrapper is used to apply photometric augmentation independently to each camera while allowing
the other processing steps to operate on the full sample.

.. note::

   The use of per-camera photometric augmentation illustrates the distinction between consistent and independent processing:
   related fields can be grouped and transformed consistently within a selected sub-tree of the input data structure, while
   different selected sub-trees can receive independent randomization. See :doc:`../../design/pipeline_processing_steps` and the
   API documentation of :class:`~accvlab.dali_pipeline_framework.processing_steps.PipelineStepBase` and
   :class:`~accvlab.dali_pipeline_framework.processing_steps.GroupToApplyToSelectedStepBase` for a more
   detailed discussion of this approach.

.. note-literalinclude:: ../../../examples/simple_full_pipeline/simple_pipeline_setup.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: packages/dali_pipeline_framework/examples/simple_full_pipeline/simple_pipeline_setup.py
   :name: simple_pipeline_setup

Entry Point
-----------

The entry point shows how the configured pipeline is used from an application script. It iterates over a few
batches, prints the nested output structure, and dumps the processed results for inspection.

.. note-literalinclude:: ../../../examples/simple_full_pipeline/run_example.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: packages/dali_pipeline_framework/examples/simple_full_pipeline/run_example.py
   :name: run_example

Expected Output
---------------

Running the example prints a short summary for each processed batch, including per-camera image shapes, mapped
annotation labels, derived ``is_active`` flags, and the sample-level scene labels. For each iteration, it also
writes TensorDumper output at
``packages/dali_pipeline_framework/examples/simple_full_pipeline/dump_simple_pipeline``. Image fields are dumped
as RGB image files, while the remaining JSON-compatible values and metadata are stored in the dump for
inspection.
