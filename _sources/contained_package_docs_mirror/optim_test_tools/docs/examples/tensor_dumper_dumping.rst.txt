Tensor Dumper – Extended Dumping Example
========================================

This example demonstrates advanced features: custom converters, per‑tensor dump type/permute overrides, 
:class:`~accvlab.batching_helpers.RaggedBatch` handling, custom processing executed only when the dumper is 
enabled, and early exit after a fixed number of dumps.

.. seealso::

    It is advisable to start with the comparison example first: :doc:`tensor_dumper_comparison`, which
    also introduces the dumping functionality, but keeps it simple.

.. seealso::
    
    The code of this example can be found in the repository under 
    ``packages/optim_test_tools/examples/tensor_dumper_dumping_example.py``.

Overview
--------

- Register custom converters for non‑tensor containers.
- Use per‑tensor overrides (format, axis permutation, exclusion) within nested structures.
- Dump gradients by registering tensors first and providing scalar losses to ``set_gradients([...])``.
- RaggedBatch support: dump as per‑sample or as a structured :class:`~accvlab.batching_helpers.RaggedBatch`.
- Run custom pre‑dump logic only when enabled via ``run_if_enabled``.

Details
-------

.. important::

   In this example, we do not divide the code into different parts which correspond to e.g. different source
   files in the actual use case, to make the example more concise. However, as 
   :class:`~accvlab.optim_test_tools.TensorDumper` is a singleton, this can be easily done in practice. 
   Please see the :doc:`stopwatch` or the :doc:`nvtx_range_wrapper` for examples of how to do this. The same 
   approach can be used with the :class:`~accvlab.optim_test_tools.TensorDumper`.

Below, we walk through the example section by section. Notes in the code are highlighted.

Synthetic Inputs Generation Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, helpers are defined to create dummy data which will be dumped in the example. Note that apart from 
functions for image and bounding box data generation, a wrapper class is defined. That class is later used to
showcase the custom converter functionality of the dumper.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------- Helper: Create synthetic inputs -------------------------
   :end-before: # ------------------- Initialize and configure the dumper -------------------

Initialize and Configure the Dumper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we initialize and configure the dumper. Note the use of 
:meth:`~accvlab.optim_test_tools.TensorDumper.perform_after_dump_count` to exit the
program after a fixed number of dumps. This can e.g. be useful during debugging to only dump a few iterations.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------- Initialize and configure the dumper -------------------
   :end-before: # ------------------------- Register custom converters -------------------------

Register Custom Converters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we register a custom converter for the ``TensorWrapper`` class.

The task of this converter is to convert the ``TensorWrapper`` object to a nested structure containing only
values supported by the dumper (tensors, NumPy arrays, types for which other custom converters are 
registered, or simple types which can written out as-is (e.g. strings)).

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------- Register custom converters -------------------------
   :end-before: # ------------------------------- Main loop -------------------------------

Main Loop
^^^^^^^^^

Here, we loop over some iterations (e.g. training iterations) and dump the data (see following sections).
Note that all the following sections are are inside the main loop.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Main loop -------------------------------
   :end-before: # --------------------------- Create the test data ---------------------------

Create the Test Data
^^^^^^^^^^^^^^^^^^^^

Generate some synthetic test data to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # --------------------------- Create the test data ---------------------------
   :end-before: # ------------------------------- Add tensors -------------------------------

Add Tensors
^^^^^^^^^^^

Add tensors to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Add tensors -------------------------------
   :end-before: # ------------------------------- Add gradients ------------------------------

Add Gradients
^^^^^^^^^^^^^

Gradients to be dumped. Note that here, tensors are added and the corresponding gradients are computed
automatically based on output (loss) values later.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Add gradients ------------------------------
   :end-before: # --------------------- Custom processing prior to dumping --------------------

Custom Processing Prior to Dumping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run custom processing prior to dumping to enable dumping of in a more accessible format.
Note that the processing is only executed if the dumper is enabled.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # --------------------- Custom processing prior to dumping --------------------
   :end-before: # ---------------------------- RaggedBatch dumping ----------------------------

RaggedBatch Dumping
^^^^^^^^^^^^^^^^^^^

Dump :class:`~accvlab.batching_helpers.RaggedBatch` data.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ---------------------------- RaggedBatch dumping ----------------------------
   :end-before: # --------------------------------- Inner loop --------------------------------


Use of Ranges for Disambiguation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, there is an inner loop where the same data is added to the dump in multiple iterations. To disambiguate
the names of the data entries, ranges can be used to add context to the data entries path. In this example,
a range containing the iteration index is used.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # --------------------------------- Inner loop --------------------------------
   :end-before: # ------------------- Placeholder for e.g. loss computation -------------------


Obtaining Gradients
^^^^^^^^^^^^^^^^^^^

Previously, we added tensors for which gradients are to be dumped (using 
:meth:`~accvlab.optim_test_tools.TensorDumper.add_grad_data`).
Here, we demonstrate how to obtain these gradients automatically based on output (e.g. loss) values.

We use a placeholder for the loss computation (``summed_3`` and ``summed_5``).
To obtain the gradients, the function :meth:`~accvlab.optim_test_tools.TensorDumper.set_gradients` is used. 
This function takes a list of scalar (loss) tensors as input, and the gradients are computed from each of 
these values and accumulated to obtain the final gradients. 
Note that this computation is performed independently of the gradients computed elsewhere (e.g. in the 
training loop). After calling this function, the gradients are ready to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------- Placeholder for e.g. loss computation -------------------
   :end-before: # ---------------------------------- Dump ----------------------------------

Dump Data
^^^^^^^^^

Finally, we dump the data.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ---------------------------------- Dump ----------------------------------

We invite the reader to run the example and inspect the dumped data.

Related Examples
----------------

- See :doc:`tensor_dumper_comparison` for a minimal setup and comparison flow.
- See :doc:`stopwatch` and :doc:`nvtx_range_wrapper` for examples of singleton tools used across multiple
  code parts.


