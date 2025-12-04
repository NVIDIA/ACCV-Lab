Batching Loss — StreamPETR Head
===============================

This page documents the batched loss computation used by ``StreamPETRHeadBatched``. It builds on the same
approach as the focal head but extends it to multiple decoder layers, keeping the computation batched over
both samples and layers while returning per-layer losses.

The reference head is already partially batched over samples: within each decoder layer, it concatenates
the per-sample targets and invokes each classification and bounding-box loss function once. The optimized
head does not newly batch those loss-function calls over samples. Instead, it batches the surrounding
per-sample target preparation, keeps variable-length ground truth in ``RaggedBatch`` objects, computes
assignment cost matrices in batches, and computes bounding-box loss only for matched prediction/target
pairs. The SciPy Hungarian solver still handles each layer/sample entry separately. The optimized head
additionally invokes the classification and bounding-box loss functions across all decoder layers at once,
including the DN losses; the reference processes one layer at a time.

You can find the implementation in the repository here:
  - StreamPETRHeadBatched: 
    ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py``
  - StreamPETRHead (reference): 
    ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py``
  - Helper classes:
  
    - HungarianAssigner3DBatched:
      ``demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d_batched.py``
    - HungarianAssigner3D (non-batched reference):
      ``demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py``

.. note::
   The used ``HungarianAssigner3DBatched`` follows the same approach as the ``HungarianAssigner2DBatched``,
   with some changes in the cost matrix computation (but still using the same approach to compute the 
   matrices). For more details, please also see the source code of the ``HungarianAssigner3DBatched``
   and the page on the :doc:`matching implementation <batching_matching>`.

Overview and Goals
------------------

.. module-docstring:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py

Below, we discuss the implementation in detail, focusing on the parts of the implementation that differ from 
or extend the focal head case.

Top-level Loss Computation
--------------------------

Pre-processing
^^^^^^^^^^^^^^

Initial validation and GT preparation (combining samples as ragged) mirrors the focal head.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-processing (`loss()`) =====
   :end-before: # ===== Prepare per-layer GT =====

Prepare Per-layer GT
^^^^^^^^^^^^^^^^^^^^

A key difference comes next where GT is duplicated across decoder layers to match prediction shapes.
Introduce a new batch dimension for decoder layers and repeat GT along it to match prediction layout:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Prepare per-layer GT =====
   :end-before: # ===== Compute batched loss over decoder layers =====

Compute Batched Loss over Decoder Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delegate to ``loss_batched`` to compute per-layer losses in a batched manner:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Compute batched loss over decoder layers =====
   :end-before: # ===== Assemble results =====

Assemble Results
^^^^^^^^^^^^^^^^

Collect losses into the output dictionary and finalize timers:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Assemble results =====
   :end-before: # ===== DN Loss (batched over layers) =====

DN Loss (Batched over Layers) & Build Loss Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When DN is enabled, DN losses are also computed batched over decoder layers and merged into the dictionary.
Note that in the original implementation, the ``dn_loss_single()`` function is called for each decoder layer
separately. Here, we implement ``dn_loss_batched()`` to compute the DN loss batched over all decoder layers.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== DN Loss (batched over layers) =====
   :end-before: # ===== Finalize (`loss()`) =====

Finalize
^^^^^^^^

The final step is to return the loss dictionary.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Finalize (`loss()`) =====
   :end-at: return loss_dict

Inside ``loss_batched()``
-------------------------

Pre-processing
^^^^^^^^^^^^^^

Shape extraction and entry-point for the per-layer batched computation:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-processing (`loss_batched()`) =====
   :end-before: # ===== Perform the matching =====

Perform the Matching
^^^^^^^^^^^^^^^^^^^^

Call the 3D batched assigner to obtain matched indices. Note that this is analogous to the focal head case.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Perform the matching =====
   :end-before: # ===== Per-layer counts and avg factors =====

Per-decoder-layer Counts and Average Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute per-layer number of positives/negatives and average factors. In contrast to the focal head case,
the number of positives/negatives is computed for each decoder layer separately.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Per-layer counts and avg factors =====
   :end-before: # ===== Align labels and align matched boxes =====

Align Labels and Align Matched Boxes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Map GT labels to predictions (for classification) and align matched GT/prediction boxes (for bounding
box regression loss):

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Align labels and align matched boxes =====
   :end-before: # ===== Losses: Classification (per-layer wrapper) =====

Losses: Classification
^^^^^^^^^^^^^^^^^^^^^^

Compute classification loss per layer using the wrapper to reduce over samples while keeping layers separate.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: Classification (per-layer wrapper) =====
   :end-before: # ===== Synchronize positives =====

Synchronize Positives
^^^^^^^^^^^^^^^^^^^^^

Synchronize positive counts across devices:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Synchronize positives =====
   :end-before: # ===== Prepare bbox targets & masks =====

Prepare Bounding Box Targets & Masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we prepare the bounding box targets and masks:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Prepare bbox targets & masks =====
   :end-before: # ===== Losses: BBox (per-layer wrapper) =====

Losses: Bounding Box Overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute bbox regression loss per layer using the wrapper, operating only on matched predictions:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: BBox (per-layer wrapper) =====
   :end-before: # ===== Finalize (`loss_batched()`) =====

Finalize
^^^^^^^^

End of per-layer loss computation:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Finalize (`loss_batched()`) =====
   :end-at: return loss_cls, loss_bbox


Other Notable Implementations Changes
-------------------------------------

In this section, we discussed the changes needed to implement the batched loss computation for the StreamPETR
head.

In addition to the changes discussed above, there are some other notable changes to the implementation.
Please refer to the source code (``streampetr_head_batched.py``, see beginning of this page) for more details.
Changes not discussed above in detail are:

  - The ``dn_loss_batched()`` method is used to compute the DN loss batched over all decoder layers. As the 
    GT data is of uniform size, there is not need to use the ``RaggedBatch`` data format. However, the 
    implementation is similar to ``loss_batched()`` in that the GT data is repeated for the individual decoder
    layers, and the loss is computed in a batched manner.
  - The ``_compute_per_layer_loss()`` method is used to compute the loss for the individual decoder layers.
    This is achieved by disabling the reduction in the called loss function, and performing the reduction 
    after the call in ``_compute_per_layer_loss()`` instead. In this way, we can decide over which dimensions 
    to reduce, allowing us to keep the batch dimension corresponding to the decoder layers intact, while 
    reducing over the individual samples. Note that the type of reduction can still be configured for the 
    loss function, and the configuration will be read and applied accordingly in the 
    ``_compute_per_layer_loss()`` method.
