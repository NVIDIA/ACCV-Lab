Replacing the Data Loader
=========================

The DALI pipeline for StreamPETR is created using the pipeline described in the 
:doc:`../../../contained_package_docs_mirror/dali_pipeline_framework/docs/examples/stream_petr/index` example (including the
pipeline itself, the custom processing step, and the custom post-processing function). It also uses the
NuScenes loading implementation described in 
:doc:`../../../contained_package_docs_mirror/dali_pipeline_framework/docs/examples/use_case_specific/nuscenes_data_loader`.

The goal is to replace the original PyTorch DataLoader (used for the training data loading) with the DALI
pipeline. Note that we focus on the training data loading, and do not replace the validation data loading.

This page describes the changes needed to the training script to enable the use of the DALI pipeline.

See the :doc:`evaluation` for the runtime and CPU usage gains from this integration and instructions
for comparing it with the PyTorch data loader.

.. note::

   This demo re-uses the StreamPETR pipeline setup already described in the
   :doc:`../../../contained_package_docs_mirror/dali_pipeline_framework/docs/examples/stream_petr/pipeline_setup`
   example, and assumes that the ``pipeline_setup`` module is available in the Python path. That example
   primarily describes the design and construction of the DALI pipeline. Please make sure that you are
   familiar with it before continuing here. This page focuses on the remaining integration work needed to
   use the completed pipeline inside the StreamPETR training implementation.
   
Main Training Script
--------------------

.. seealso::

   The main training script can be found at ``demos/streampetr/tools/train.py`` in the repository.

Environment and Paths
^^^^^^^^^^^^^^^^^^^^^

Before creating the pipeline, the script configures the Python path to include the DALI example module and
project roots needed by the demo.

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Setting up the Python Path (including DALI pipeline) =====
   :end-at: # ===== End of Setting up the Python Path (including DALI pipeline) =====

DALI Pipeline Setup
^^^^^^^^^^^^^^^^^^^

The major change is the creation of the DALI pipeline instead of a PyTorch Dataset. Both options are supported
in our modified implementation, and can be enabled by setting the ``use_dali_pipeline`` flag to ``True`` in
the used configuration file.

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Get Dataset or DALI Output Iterator =====
   :end-at: # ===== End of Getting Dataset or DALI Output Iterator =====


Checkpoint Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The setting of the checkpoint meta information is slightly adjusted to use the ``classes`` variable instead of 
``dataset[0].CLASSES`` (as the latter is not available when DALI is used).

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Checkpoint Configuration =====
   :end-at: # ===== End of Checkpoint Configuration =====

Training Entry Point
^^^^^^^^^^^^^^^^^^^^

The training call is unchanged except that the dataset list may contain a DALI iterator. This case needs to
be handled correctly in the training function (see the next section for details).

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Train the model =====
   :end-at: # ===== End of Train the model =====


Adjustments to the Training Function
------------------------------------

In the training script, we call ``custom_train_model()`` to train the model, which in turn calls 
``custom_train_detector()`` (with the latter defined in ``projects/mmdet3d_plugin/core/apis/mmdet_train.py``). 
The implementation of ``custom_train_detector()`` needs to be adjusted to handle the case that the passed 
dataset is actually a :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator` object. Please see
the code snippet below for details on how this is done.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/apis/mmdet_train.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/apis/mmdet_train.py
   :start-at: # ===== Using the DALI Iterator Directly (if used) =====
   :end-at: # ===== End of Using the DALI Iterator Directly (if used) =====

.. important::

    Note that from this point on, the :class:`~accvlab.dali_pipeline_framework.pipeline.DALIStructuredOutputIterator`
    acts as a drop-in replacement for the original PyTorch DataLoader (in case DALI is used), and no changes 
    deeper in the training code are needed to switch from the original PyTorch DataLoader to the DALI 
    pipeline.

.. note::

   **Mini-dataset replication.** The performance configurations set
   ``data_loading_config.replicate_dataset_per_gpu=True`` for both loaders because
   NuScenes mini has too few sequences for the global batch size. This lets each GPU
   sample the full dataset. It is an evaluation setting, not a required step when
   integrating DALI into another project; the option defaults to ``False``.

   In ``tools/train.py``, ``get_dali_pipeline_distributed_args()`` passes ``rank=0`` and
   ``world_size=1`` to the pipeline setup, while keeping ``device_id=local_rank``.
   DDP continues to use the real distributed ranks.

   For PyTorch, ``core/apis/mmdet_train.py`` forwards the flag to ``datasets/builder.py``,
   which passes ``rank=0`` and ``num_replicas=1`` to the training sequence sampler.
   Worker initialization still uses the real rank, and validation retains its normal sampler.
   See :ref:`the evaluation's replication policy <streampetr-dali-dataset-replication>`
   for the workload and augmentation-seed limitations.
