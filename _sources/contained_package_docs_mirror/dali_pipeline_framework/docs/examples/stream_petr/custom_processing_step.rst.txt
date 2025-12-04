Custom Processing Step: ``StreamPETRDataCombiner``
==================================================

This page documents the custom processing step ``StreamPETRDataCombiner`` used by the StreamPETR pipeline.
It prepares images, camera geometry and (optionally) ground truth into tensors close to the format expected
by StreamPETR training, minimizing the work required in the post‑processing function (see 
:doc:`custom_post_processing`).

Overview and Goals
------------------

.. module-docstring:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py

.. seealso::

   Please also see the :doc:`../flexible_step/index` page for a general introduction on how to implement 
   custom processing steps.

Details
-------

Here, we discuss the details of the ``StreamPETRDataCombiner`` custom processing step.

Add and Initialize Output Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New fields are added to hold combined tensors while original fields remain available during conversion.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: def _process(self, data: SampleDataGroup) -> SampleDataGroup:
   :end-before: # ===== Combine camera images =====

Combine Camera Images
^^^^^^^^^^^^^^^^^^^^^

Per‑camera images are transposed to channel‑first (C,H,W) and stacked across cameras.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Combine camera images =====
   :end-before: # ===== Adjust and combine projection geometry =====

Adjust and Combine Projection Geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrices are padded to homogeneous 4×4 where necessary and stacked across cameras.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Adjust and combine projection geometry =====
   :end-before: # ===== Store combined tensors =====

Store Combined Tensors
^^^^^^^^^^^^^^^^^^^^^^

Combined images, geometry, ego poses, and timestamps are stored in the new fields.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Store combined tensors =====
   :end-before: # ===== Process ground truth =====

Process Ground Truth
^^^^^^^^^^^^^^^^^^^^

The 3D bounding boxes are converted to the training layout, angles wrapped, NaNs removed; per‑camera 2D GT is 
padded and stacked with per‑camera object counts.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Process ground truth =====
   :end-before: # ===== Convert `prev_exists` to float =====

Convert Existence Flag to Float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``prev_exists`` flag is converted from boolean to ``float`` inside the pipeline. This is done as the flag 
is expected to be a ``float`` in the training implementation.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Convert `prev_exists` to float =====
   :end-before: # ===== Cleanup =====

Cleanup
^^^^^^^

Unneeded fields are removed to leave only combined outputs.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Cleanup =====
   :end-before: return data

Data Structure Checks and & Output Adjustments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper methods validate the input data structure and add/remove fields to match the output structure. 
For more details on how to implement custom processing steps, see 
:doc:`../flexible_step/index`.

.. note-literalinclude:: ../../../examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/additional_impl/processing_steps/stream_petr_data_combiner.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Data structure checks and & output adjustments =====


