2D Object Detection Pipeline Setup
==================================

This page documents the DALI-based 2D object detection input pipeline as implemented in
``packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py``.

Overview and Goals
------------------

.. module-docstring:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py


Details
-------

Here, we discuss the details of the 2D object detection pipeline setup.

Input Data Handling
^^^^^^^^^^^^^^^^^^^

The first step is to ensure that input data is provided to the pipeline.
Please also see :doc:`../use_case_specific/nuscenes_data_loader` for more details on the data loading
in this case and :doc:`../../design/input` for a general overview of the input handling.

.. note-literalinclude:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Input data handling =====
   :end-before: # ===== Define processing steps =====

Define Processing Steps
^^^^^^^^^^^^^^^^^^^^^^^

Next, we define the processing steps that will be applied to the input data.
Please also see :doc:`../../design/pipeline_processing_steps` for a general overview of the processing steps
and :doc:`../../design/sample_data_group` for more details on the data format.

.. note-literalinclude:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Define processing steps =====
   :end-before: # ===== Pipeline definition & Output data format =====

Pipeline Definition & Output Data Structure Blueprint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline definition wires the input callable with the processing steps. It also provides utilities
such as obtaining the output data structure blueprint (see 
:meth:`~accvlab.dali_pipeline_framework.pipeline.PipelineDefinition.check_and_get_output_data_structure`), which 
we use here.

The output blueprint captures the hierarchical structure and types of the pipeline output.
It is later used to reconstruct structured samples from flat DALI outputs.

.. note-literalinclude:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pipeline definition & Output data format =====
   :end-before: # ===== Create DALI pipeline =====


Create and Build the DALI Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The concrete DALI pipeline can be created using the pipeline definition and configured similar to how
it is done for a standalone DALI pipeline.


.. note-literalinclude:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Create DALI pipeline =====
   :end-before: # ===== Wrap as iterator =====


Wrap as Structured Iterator & Return
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DALI pipeline is wrapped as a ``DALIStructuredOutputIterator`` that can be used as a
drop-in replacement for a PyTorch ``DataLoader``.

.. note-literalinclude:: ../../../examples/pipeline_setup/object_detection_2d_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/object_detection_2d_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Wrap as iterator =====
   :end-at: return res_iterator

.. seealso::
   For an example using sequence-based sampling and multi-task learning (3D object detection with auxiliary 
   2D object detection) see :doc:`../stream_petr/index`.


