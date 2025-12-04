Entry Point
===========

The entry point script is a runnable driver around the pipeline setup documented in this tutorial. You can run
it using:

``packages/dali_pipeline_framework/examples/example_pipeline_obj_det_2d.py``

The script parses the example configuration, builds the DALI pipeline as a structured iterator that can be used
as a drop-in replacement for a PyTorch ``DataLoader``, and iterates a few batches in a training-loop-like
structure.

The script contains additional inline comments explaining how the pipeline configuration is assembled from the
command-line options, how the structured iterator is used like a PyTorch ``DataLoader``, how the end of an
epoch is handled in the example loop, and how the returned data structure differs between single-image and
multi-camera modes.

.. note::

   At the end of the run, the script uses :class:`~accvlab.optim_test_tools.TensorDumper` to dump the last
   batch. It also creates debugging visualizations for the 2D detection output, including images with bounding
   boxes and heatmap overlays.
