Introduction
============

The goal of the `batching-helpers` package is to enable an efficient and easy-to-use batching in case 
of non-uniform batch sizes in order to speed up loss computation on the GPU. To achieve this, the 
package defines a custom data type (:class:`~accvlab.batching_helpers.RaggedBatch`) and provides helper functions which 
enable/simplify batched operations on the data. 
In object detection and autonomous driving training tasks, a common challenge arises from non-uniform batch sizes. 
This occurs because:

1. The number of ground truth (GT) objects varies per sample
2. While the number of predictions is typically fixed, not all predictions correspond to actual objects (e.g. some predictions are background)

This non-uniformity affects:

- Ground truth data
- Predictions used in loss computation (if e.g. only matched predictions are processed)
- Correspondences between predictions and GT objects (e.g. index pairs linking predictions to GT objects)

The `batching-helpers` package addresses these challenges through:

1. :class:`~accvlab.batching_helpers.RaggedBatch`: An efficient data format for storing and processing non-uniform batches
2. Specialized operations for common loss computation tasks that are difficult to implement efficiently with standard PyTorch operators in a batched manner

A key example of a specialized operation is selecting matched objects from both ground truth data and predictions based on index pairs established by the matching 
algorithm. While PyTorch's built-in indexing works well for single-sample processing, batch processing with varying indices and numbers of indices per sample 
requires specialized handling. The package provides efficient helper functions for this and other common operations.

Note that some specialized operations are GPU-only (denoted as such in the API documentation), while others support both CPU and GPU execution. 
As the goal of the package is to improve efficiency on the GPU through batching, many operations are implemented for the GPU only.
However, for some of the operations, a CPU implementation is provided. This includes operations which are potentially more efficient on the CPU (e.g. due to processing
of many small tensors, such as in the case of :func:`~accvlab.batching_helpers.combine_data`) or allow for more efficient CPU-GPU memory 
transfers (e.g. by first performing :func:`~accvlab.batching_helpers.combine_data` on CPU tensors, followed by a transfer of the batch as a whole onto the GPU).

.. seealso::

   Please refer to the :doc:`api` for details on the provided functionality and the :doc:`example` for a 
   practical example and a short discussion about why using the approach implemented in this package leads to 
   performance gains.
