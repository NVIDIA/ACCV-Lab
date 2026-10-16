Summary
=======

The demo uses :class:`~accvlab.batching_helpers.RaggedBatch` to represent samples with different numbers
of objects. This allows cost matrices and losses to be computed in batches while tracking which entries
belong to each sample.

The :doc:`matching tutorial <batching_matching>` explains batched cost-matrix construction; the
Hungarian assignment itself still runs per sample on the CPU. The loss tutorials show how to handle
the sample and camera dimensions in the :doc:`Focal head <batching_loss_focal_head>` and the additional
decoder-layer dimension in the :doc:`StreamPETR head <batching_loss_stream_petr_head>`.

See the :doc:`evaluation` for the measured gains, the changes shared by the baseline and optimized
implementations, and commands for running the comparison.
