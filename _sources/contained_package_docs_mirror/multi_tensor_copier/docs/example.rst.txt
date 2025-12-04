Example
=======

Here, we provide an example of how to use the ``multi-tensor-copier`` package to efficiently copy data
containing many small tensors in a nested structure (here: training meta-data) from CPU to GPU.

The example consists of the following steps:

1. Construction of per-sample meta-data with variable-size tensors (for illustration purposes; in a real
   use-case, the meta-data originates e.g. from a PyTorch DataLoader)
2. Asynchronous copy of the entire batch of meta-data to the GPU
3. Overlapping useful work with the transfer
4. Retrieval and consumption of the GPU-resident meta-data

.. important::

   You can run the example using the script ``packages/multi_tensor_copier/example/example.py``.


Example Data Structure
----------------------

The meta-data is organized as a list of per-sample dictionaries (one per batch element). Each sample 
dictionary contains:

- ``"cams_gt"``: a list of 6 camera dicts, each holding:

  - ``"bounding_boxes"``: an ``(N, 4)`` tensor of 2D bounding boxes, where ``N`` varies per image
    (number of visible objects)
  - ``"class_ids"``: an ``(N,)`` tensor of integer class IDs
  - ``"active"``: an ``(N,)`` boolean tensor indicating active objects
  - ``"depths"``: an ``(N,)`` tensor of depth values
  - ``"proj_mat"``: a ``(3, 4)`` projection matrix from camera coordinates to image coordinates

- ``"gt_data"``: a dict with:

  - ``"bounding_boxes_3d"``: an ``(N, 7)`` tensor of 3D ground truth bounding boxes, where ``N`` varies per
    sample (number of ground truth objects)
  - ``"class_ids"``: an ``(N,)`` tensor of integer class IDs
  - ``"active"``: an ``(N,)`` boolean tensor indicating active objects

This nested structure of lists, dicts, and variable-size tensors is representative of real-world
training tasks (e.g. a multi-camera 3D object detection like
`StreamPETR <https://arxiv.org/abs/2303.11926>`_). It is also a 
scenario where standard PyTorch ``.to()`` calls are particularly inefficient: the batch contains many small 
tensors in non-pinned memory, and each individual ``.to()`` call incurs overhead that can 
dominate the actual transfer time for a small tensor. See the :doc:`introduction <intro>` for a detailed discussion of the 
motivation and the optimizations that ``multi-tensor-copier`` applies to ensure efficient copying in this 
scenario.


Workflow
--------

The optimizations described in the :doc:`intro` are applied automatically (all enabled by default).

The following snippet shows the core workflow. After the batch meta-data has been assembled (see the
full script at ``packages/multi_tensor_copier/example/example.py`` for the data creation helpers used in this
example), we pass it to :func:`~accvlab.multi_tensor_copier.start_copy` together with the target device.
The function traverses the nested structure and returns an
:class:`~accvlab.multi_tensor_copier.AsyncCopyHandle` while the transfer proceeds in the background.
Because the copy runs asynchronously, the main thread is free to perform other operations while the
transfer is in flight (e.g. computations not involving the copied data, logging, etc.).
Finally, :meth:`~accvlab.multi_tensor_copier.AsyncCopyHandle.get` blocks until the copy is complete
and returns the a nested structure corresponding to the input, but with all tensors now residing on the GPU.

.. note-literalinclude:: ../example/example.py
   :language: python
   :caption: packages/multi_tensor_copier/example/example.py
   :linenos:
   :lineno-match:
   :start-at: # ----------------------- Create the batch meta-data -----------------------
   :end-before: if __name__
