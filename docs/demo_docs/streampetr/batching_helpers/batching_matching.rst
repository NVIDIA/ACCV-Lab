Batching Matching
=================

This page documents the batched matching (``HungarianAssigner2DBatched``) used by 
``FocalHeadBatched``.

You can find the implementation in the repository here:
  - HungarianAssigner2DBatched: 
    ``demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py``
  - HungarianAssigner2D (non-batched reference): 
    ``demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d.py``
  - FocalHeadBatched: ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head_batched.py``
  - FocalHead (reference): ``demos/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head.py``

Overview and Goals
------------------

.. module-docstring:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py

Pre-processing
--------------

In contrast to the original implementation, the matching is performed for the whole batch at once. 
:class:`~accvlab.batching_helpers.RaggedBatch` objects are used to handle the input data.

The following code snippet shows the pre-processing step, which is the first step performed inside the 
``assign`` method (which performs the matching). Please refer to the highlighted comments (marked with 
``#@NOTE``) for more details.

(As a side note, we use NVTX ranges to annotate major phases for profiling, using the 
:class:`~accvlab.optim_test_tools.NVTXRangeWrapper` class.)


.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Pre-processing =====
   :end-before: # ===== Compute the cost matrices =====

Cost Matrix Computation
-----------------------

The cost matrices are obtained by applying multiple cost functions to the input data. For each cost function, 
a matrix is computed which contains the cost for each possible prediction-GT pair. In the original 
implementation, the cost matrices are computed for a single sample at a time. Here, we show the code 
for the computation of the cost matrices in a batched manner. Note that the size of the cost matrices depends
on the number of GTs, and so varies between samples.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Compute the cost matrices =====
   :end-before: # ===== Perform the matching =====

Perform the Matching
--------------------

While the cost matrices are computed in a batched manner, the matching is performed one sample at a time.
This is done because the matching is implemented on the CPU and does not support batched operation.

As the rest of the function is batched, we need to iterate over the samples and perform the matching for each 
sample individually. After that, the results are combined into a single `RaggedBatch` object. 

Note that this is equivalent to the matching used in the original implementation. However, as there, the 
``assign`` method processes one sample at a time, no loop is used there.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Perform the matching =====
   :end-before: # ===== Combine the matching results & post-processing =====


Combine Results and Post-processing
-----------------------------------

After performing the matching, the results are combined into ``RaggedBatch`` objects. In addition, note that

  - The original batch shape is restored using the ``reshape_batch_dims`` method.
  - The result format is different from the original implementation. Here, we return the matching results
    as index pairs (in two separate ``RaggedBatch`` objects, one containing indices in the predictions, and
    one containing corresponding indices in the GTs). In the original implementation, an ``AssignResult`` 
    object is returned.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Combine the matching results & post-processing =====

Note on the 3D Assigner Implementation
--------------------------------------

Note that apart from the HungarianAssigner2DBatched implementation, there is also a 
``HungarianAssigner3DBatched`` implementation, which is used by the ``StreamPETRHeadBatched`` class.
The implementation is very similar to the ``HungarianAssigner2DBatched`` implementation, and so we will not
discuss it here. Please refer to the ``HungarianAssigner3DBatched`` implementation for more details.
