Custom Post-Processing
======================

This page documents the post-processing function used by the StreamPETR pipeline to align DALI outputs
with the training format expected by StreamPETR (including wrapping into ``DataContainer`` and 
``LiDARInstance3DBoxes`` objects where needed).

Overview and Goals
------------------

Here, we demonstrate how to align the DALI outputs with the training format expected by StreamPETR by 
means of a custom post-processing function. This function can be used to implement any custom post-processing 
that cannot be implemented inside the DALI pipeline (e.g. due to data types not supported by DALI or 
batching conventions which differ from the DALI batching conventions).

Details
-------

Here, we discuss the details of the post-processing function used by the StreamPETR pipeline.

Helper and Input Shape
^^^^^^^^^^^^^^^^^^^^^^

Derive batch/image shape information and define a helper for converting 3D boxes to
``LiDARInstance3DBoxes``.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Helper and input shape =====
   :end-before: # ===== Wrap core tensors as DataContainer objects =====

Wrap Core Tensors as ``DataContainer`` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a sequence dimension and wrap the actual tensors as ``DataContainer`` objects for training code 
compatibility.
Note that we also add a sequence dimension to the tensors (length 1, as we use the streaming video training 
mode), as this is the format that the training implementation expects.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Wrap core tensors as DataContainer objects =====
   :end-before: # ===== Process ground truth (training only) =====

Process Ground Truth (Training Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trim padded regions, structure nesting for sequences/batches, convert 3D boxes to
``LiDARInstance3DBoxes``, and wrap results as ``DataContainer`` objects.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Process ground truth (training only) =====
   :end-before: # ===== Set image metas =====

Set Image Metadata
^^^^^^^^^^^^^^^^^^

Populate ``img_metas`` to mirror the format of the data used in the original implementation.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Set image metas =====
   :end-before: # ===== Cleanup =====

Cleanup
^^^^^^^

Remove internal fields that are no longer needed once padding has been reversed.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Cleanup =====
   :end-before: return batch

