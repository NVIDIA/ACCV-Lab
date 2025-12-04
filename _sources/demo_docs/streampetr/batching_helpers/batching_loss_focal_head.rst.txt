Batching Loss — Focal Head
==========================

This page describes the batched loss computation used by ``FocalHeadBatched``. 

You can find the implementation in the repository here:
  - FocalHeadBatched: ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py``
  - FocalHead (reference): ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head.py``


Overview and Goals
------------------

.. module-docstring:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py

Below, we walk through the important parts of the implementation which were modified to support batching
using the ``batching-helpers`` package.

Top-level Loss Computation Implementation
-----------------------------------------

The top-level loss computation which is called from outside the class is called ``loss()`` and is 
described here.

Pre-processing
^^^^^^^^^^^^^^

The pre-processing step performs some validation and extracts the necessary predictions from a dictionary.
Compared to the original implementation, there is one important difference: the ground truth (GT) data of the 
individual samples and cameras is combined into ``RaggedBatch`` objects (see comments in the code).

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-processing (`loss()`) =====
   :end-before: # ===== Compute batched loss =====

Compute Batched Loss (Delegation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Loss computation is delegated to ``loss_batched`` with batched inputs.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Compute batched loss =====
   :end-before: # ===== Assemble results =====

Assemble Results
^^^^^^^^^^^^^^^^

The results are collected into a dictionary and returned.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Assemble results =====
   :end-at: return loss_dict

Detailed Loss Computation Implementation
----------------------------------------

The actual loss computation is done in the ``loss_batched()`` function. This function is called once per batch
and is provided with the data of all samples and cameras (combined into 
:class:`~accvlab.batching_helpers.RaggedBatch` objects for the GT data inside ``loss()``). The function is 
called internally from ``loss()``.

Pre-processing
^^^^^^^^^^^^^^

Some simple steps are performed first. Note that splitting the predictions into individual samples and cameras 
(as done in the original implementation) is not needed here (see comment in the code).

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-processing (`loss_batched()`) =====
   :end-before: # ===== Perform the matching =====

Matching
^^^^^^^^

Invoke the batched assigner to get matched indices as ``RaggedBatch`` pairs:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Perform the matching =====
   :end-before: # ===== Normalization factors =====

Normalization Factors
^^^^^^^^^^^^^^^^^^^^^

Compute image-size-based factors used for scaling and conversions:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Normalization factors =====
   :end-before: # ===== Align data by matches =====

Align Data by Matches
^^^^^^^^^^^^^^^^^^^^^

In this step, the alignment of the predictions and GT is performed using the matched indices, so that loss
computation can be performed on tensors where the elements are aligned according to the matching results.

Note that in contrast to all other losses, the classification loss is computed for all predictions, not only 
for matched pairs. This leads to a different implementation of the alignment step (see comments in the code).

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Align data by matches =====
   :end-before: # ===== Pre-process tensors =====

Pre-process Tensors
^^^^^^^^^^^^^^^^^^^

Convert bounding boxes between formats, normalize/denormalize as required, and normalize centers. This step makes
heavy use of the :meth:`~accvlab.batching_helpers.RaggedBatch.apply` method to modify the data in the 
:class:`~accvlab.batching_helpers.RaggedBatch`. Apart from this, it differs from the original implementation,
as the format of the data is different at this point (some conversion is already performed there on the data 
before it is returned from the ``get_targets()`` function).

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-process tensors =====
   :end-before: # ===== Existence weights & count valid and invalid elements =====

Count Valid and Invalid Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Count the number of the valid and invalid elements, and derive existence weights. This is done using
functionality of the :class:`~accvlab.batching_helpers.RaggedBatch` class. Note that all of these are also
used in the original implementation, but there, they are provided by the ``get_targets()`` function.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Existence weights & count valid and invalid elements =====
   :end-before: # ===== Losses: IoU & IoU score =====

Losses: IoU Loss and IoU Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The IoU loss is a typical example of a loss computation, where only the matched pairs are considered.
Apart from the loss, the IoU score is also computed and used as a weight for the classification loss later on.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: IoU & IoU score =====
   :end-before: # ===== Losses: Classification (setup & weights) =====

Losses: Classification (Setup & Weights)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare class scores, average factor, and weights; compute IoU scores and map back to prediction shape for weighting:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: Classification (setup & weights) =====
   :end-before: # ===== Losses: Classification =====

Losses: Classification Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the classification loss over all predictions, weighted by IoU for existent predictions:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: Classification =====
   :end-before: # ===== Synchronize positives =====

Synchronize Positives
^^^^^^^^^^^^^^^^^^^^^

Synchronize positive counts across devices (same as reference):

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Synchronize positives =====
   :end-before: # ===== Heatmap generation =====

Heatmap Generation
^^^^^^^^^^^^^^^^^^

Generate target heatmaps in a batched way using ``draw_heatmap_batched``. Note that by using the custom
heatmap generation OP in combination with the :class:`~accvlab.batching_helpers.RaggedBatch` class, the 
implementation:

  - Is optimized by batching compared to the original implementation.
  - Is simplified compared to our batched implementation of the heatmap generation that was introduces in the 
    reference implementation, as the :class:`~accvlab.batching_helpers.RaggedBatch` class makes it clear which 
    bounding box is from which entry in the batch.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Heatmap generation =====
   :end-before: # ===== Losses: Centerness =====

Losses: Centerness Loss
^^^^^^^^^^^^^^^^^^^^^^^

Centerness is computed for all predictions. Note that this loss is special in the sense that the fixed-sized
heatmap is instead of the original, non-uniformly-sized GT.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: Centerness =====
   :end-before: # ===== Losses: BBox and Centers2D =====

Losses: BBox and Centers2D Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These losses are computed only for the matched predictions (similar to the IoU loss discussed before).

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Losses: BBox and Centers2D =====
   :end-before: # ===== Finalize =====

Finalize Loss Computation
^^^^^^^^^^^^^^^^^^^^^^^^^

In the final step, the individual losses are returned.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Finalize =====
   :end-at: return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness
