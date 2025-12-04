Entry Point
===========

The entry point script is a runnable driver around the StreamPETR pipeline setup documented in this tutorial.
You can run it using:

``packages/dali_pipeline_framework/examples/example_pipeline_stream_petr.py``

The script parses the example configuration, builds the DALI pipeline as a structured iterator that can be used
as a drop-in replacement for a PyTorch ``DataLoader``, iterates a few batches in a training-loop-like
structure, and exercises the output format expected by the StreamPETR training integration.

The script contains additional inline comments explaining how the runnable example overrides the pipeline
configuration, how the structured iterator is used like a PyTorch ``DataLoader``, how the end of an epoch is
handled in the example loop, and how the pipeline output relates to the StreamPETR training format.

.. note::

   At the end of the run, the script uses :class:`~accvlab.optim_test_tools.TensorDumper` to dump the last
   batch. Because the StreamPETR example can produce objects from the original training interface, the script
   also registers custom dumper converters for ``LiDARInstance3DBoxes`` and ``DataContainer``.
