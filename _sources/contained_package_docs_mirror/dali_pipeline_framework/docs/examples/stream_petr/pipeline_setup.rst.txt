StreamPETR Pipeline Setup
=========================

This page documents the DALI-based StreamPETR training pipeline as implemented in
``examples/pipeline_setup/stream_petr_pipeline.py``.

Overview and Goals
------------------

.. module-docstring:: ../../../examples/pipeline_setup/stream_petr_pipeline.py

.. important::

   The pipeline described here is designed as a drop-in replacement for the PyTorch DataLoader
   which can be used for the training of the 
   `original StreamPETR implementation <https://github.com/exiawsh/StreamPETR>`_ in the 
   **streaming video training mode** (``seq_mode=True``).
   Therefore, we implement the pipeline in a way that the final output data structure (of the 
   :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` object wrapping the 
   pipeline) is compatible with the output data structure of the PyTorch DataLoader in the original 
   implementation and contains all the data needed for training in the expected format.

Details
-------

Here, we discuss the details of the ``StreamPETR`` pipeline setup.

Input Data Handling
^^^^^^^^^^^^^^^^^^^

Here, we use a :class:`~accvlab.dali_pipeline_framework.inputs.SequenceSampler` to load sequence-based data.
When loading the data, we also perform some simple high-level pre-processing of the metadata, such as scene 
filtering and sequence splitting.

See the :doc:`../2d_object_detection/pipeline_setup` for general input callable mechanics as well as 
sharding/shuffling.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Input data handling =====
   :end-before: # ===== Define processing steps =====

Define Processing Steps
^^^^^^^^^^^^^^^^^^^^^^^

Here, we define the pre-processing steps that will be applied to the input data.

The processing is more involved than in the :doc:`../2d_object_detection/pipeline_setup`. Please see the 
code below, and refer to :doc:`../2d_object_detection/pipeline_setup` for more details on some of the steps. 

Also note that this pipeline uses a custom step to combine and format the data to a format close to StreamPETR 
training expectations; see :doc:`custom_processing_step` for details.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Define processing steps =====
   :end-before: # ===== Pipeline definition & Output data format =====

Pipeline Definition & Output Data Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wiring the input implementation with the steps and capturing the output data structure blueprint.
For details on the output blueprint, refer back to the :doc:`../2d_object_detection/pipeline_setup`.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pipeline definition & Output data format =====
   :end-before: # ===== Create DALI pipeline =====

Create and Build the DALI Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Seed setup for augmentation reproducibility, pipeline creation, and build. See the 
:doc:`../2d_object_detection/pipeline_setup` for more background on these steps.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Create DALI pipeline =====
   :end-before: # ===== Wrap as iterator =====

Wrap as Structured Iterator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline is wrapped as a ``DALIStructuredOutputIterator``. A post-processing function aligns the
output format to the structures expected by StreamPETR training.

For detail on the post-processing function, see :doc:`custom_post_processing`.

.. note-literalinclude:: ../../../examples/pipeline_setup/stream_petr_pipeline.py
   :language: python
   :caption: packages/dali_pipeline_framework/examples/pipeline_setup/stream_petr_pipeline.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Wrap as iterator =====
   :end-before: return res_iterator


