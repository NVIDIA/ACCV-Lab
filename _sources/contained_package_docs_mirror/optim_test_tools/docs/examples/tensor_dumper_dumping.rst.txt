Tensor Dumper – Extended Dumping Example
========================================

This example demonstrates advanced features: custom converters, per‑tensor dump type/permute overrides, 
:class:`RaggedBatch` handling, custom processing executed only when the dumper is enabled, and 
early exit after a fixed number of dumps.

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
- RaggedBatch support: dump as per‑sample or as a structured :class:`RaggedBatch`.
- Run custom pre‑dump logic only when enabled via ``run_if_enabled``.

Details
-------

Below, we walk through the example section by section. Notes in the code are highlighted.

Create Synthetic Inputs Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we define a helper function to create some synthetic inputs to be dumped as well as a wrapper class for 
demonstrating the custom converter functionality.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------- Helper: Create synthetic inputs -------------------------
   :end-before: # ------------------- Initialize and configure the dumper -------------------

Initialize and Configure the Dumper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we initialize and configure the dumper.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------- Initialize and configure the dumper -------------------
   :end-before: # ------------------------- Register custom converters -------------------------

Register Custom Converters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we register a custom converter for the ``TensorWrapper`` class (which is a simple wrapper used for 
demonstrating the custom converter functionality).

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

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Main loop -------------------------------
   :end-before: # --------------------------- Create the test data ---------------------------

Create the Test Data
^^^^^^^^^^^^^^^^^^^^

Here, we create some synthetic test data to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # --------------------------- Create the test data ---------------------------
   :end-before: # ------------------------------- Add tensors -------------------------------

Add Tensors
^^^^^^^^^^^

Here, we add the tensors to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Add tensors -------------------------------
   :end-before: # ------------------------------- Add gradients ------------------------------

Add Gradients
^^^^^^^^^^^^^

Here, we add the gradients to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------------------- Add gradients ------------------------------
   :end-before: # --------------------- Custom processing prior to dumping --------------------

Custom Processing Prior to Dumping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we run some custom processing prior to dumping to enable dumping of in a more accessible format.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # --------------------- Custom processing prior to dumping --------------------
   :end-before: # ---------------------------- RaggedBatch dumping ----------------------------

RaggedBatch Dumping
^^^^^^^^^^^^^^^^^^^

Here, we dump the RaggedBatch data.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ---------------------------- RaggedBatch dumping ----------------------------
   :end-before: # ------------------- Placeholder for e.g. loss computation -------------------

Placeholder for e.g. Loss Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we place a placeholder for e.g. the loss computation to demonstrate how the gradients are computed
automatically.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ------------------- Placeholder for e.g. loss computation -------------------
   :end-before: # ----------------------------- Set the gradients -----------------------------

Set the Gradients
^^^^^^^^^^^^^^^^^

Here, we set the gradients to be dumped.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ----------------------------- Set the gradients -----------------------------
   :end-before: # ---------------------------------- Dump ----------------------------------

Dump Data
^^^^^^^^^

Finally, we dump the data. We invite the reader to run the example and inspect the dumped data.

.. note-literalinclude:: ../../examples/tensor_dumper_dumping_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_dumping_example.py
   :linenos:
   :lineno-match:
   :start-at: # ---------------------------------- Dump ----------------------------------

Related Examples
----------------

- See :doc:`tensor_dumper_comparison` for a minimal setup and comparison flow.
- See :doc:`stopwatch` and :doc:`nvtx_range_wrapper` for examples of singleton tools used across multiple
  code parts.


