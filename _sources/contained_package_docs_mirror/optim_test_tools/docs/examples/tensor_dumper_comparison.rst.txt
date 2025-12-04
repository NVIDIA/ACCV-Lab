Tensor Dumper – Comparison Example
==================================

Start here for the tensor dumper. This example shows how to enable the dumper, dump tensors,
and compare a subsequent run against a stored reference. The :doc:`tensor_dumper_dumping` example shows 
more features (custom converters, per‑tensor overrides, RaggedBatch support, custom pre‑dump processing, 
early exit).

Note that the dumper is a singleton similar to the other tools (see :doc:`stopwatch` and 
:doc:`nvtx_range_wrapper`). Therefore, it can be enabled once and used across multiple code parts.
However, this is not the focus of this example.

.. seealso::
    
    The code of this example can be found in the repository under 
    ``packages/optim_test_tools/examples/tensor_dumper_comparison_example.py``.

Overview
--------

- Enable the dumper and choose dump location.
- Dump data in one run; compare in a later run (or next loop iteration here for demonstration).
- Override dump formats to JSON for comparison (comparison supports only JSON).
- Gradients are auto-computed when you call ``set_gradients([...])`` with scalar losses.

Example
-------

Example Code
^^^^^^^^^^^^

Please see the notes in the code for more details.

.. note-literalinclude:: ../../examples/tensor_dumper_comparison_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/tensor_dumper_comparison_example.py
   :linenos:
   :lineno-match:

Output
^^^^^^

Here, we show example output (for the test configuration as set in the example code).

There are two warnings, one for the tensors and one for the gradients.

Tensor Comparison
~~~~~~~~~~~~~~~~~

Note that 
  - The number of numerical errors is limited per tensor. This is controlled by the 
    ``num_errors_per_tensor_to_show`` parameter and used to prevent the output from being too verbose (e.g.
    when comparing large tensors such as images)
  - Apart from the numerical errors, there are also mismatch errors for missing/extra keys. Missing keys may
    be allowed (configurable). This option may be useful when the current data is based on an implementation 
    in progress, so that some of the data is not yet available.
  - Instead of displaying warnings, an error can be raised instead by setting the ``as_warning`` parameter to 
    ``False``.

.. code-block:: text

  Comparison of data with previously dumped data failed for [tensors] data for dump 0.
  NOTE: The following errors were found for the dumped [tensors] data for dump 0.
        Up to 3 most significant errors are shown per tensor.
    Numerical mismatch at path: tensor1[0,9]
      Dumped data: 0.7697955965995789
      Struct to compare: -2.429549217224121
      Difference (current - dumped): -3.1993448138237
    Numerical mismatch at path: tensor1[9,2]
      Dumped data: 2.10503888130188
      Struct to compare: -0.7213656306266785
      Difference (current - dumped): -2.8264045119285583
    Numerical mismatch at path: tensor1[5,8]
      Dumped data: -1.2030445337295532
      Struct to compare: 1.4832252264022827
      Difference (current - dumped): 2.686269760131836
    Numerical mismatch at path: other_tensors.tensor2[1,7]
      Dumped data: 1.8169379234313965
      Struct to compare: -2.4539949893951416
      Difference (current - dumped): -4.270932912826538
    Numerical mismatch at path: other_tensors.tensor2[1,2]
      Dumped data: -2.2293128967285156
      Struct to compare: 0.7702249884605408
      Difference (current - dumped): 2.9995378851890564
    Numerical mismatch at path: other_tensors.tensor2[4,7]
      Dumped data: -1.3049689531326294
      Struct to compare: 1.670259952545166
      Difference (current - dumped): 2.9752289056777954
    Missing key 'tensor3' at path: 'other_tensors' in dumped reference
    Extra key 'tensor4' at path: 'other_tensors' in dumped reference
    warnings.warn(error_message)


Gradients
~~~~~~~~~

The comparison of gradients is analogous to the comparison of tensors. In this example, only numerical errors 
are present.

.. code-block:: text

  UserWarning: Comparison of data with previously dumped data failed for [grads] data for dump 0.
  NOTE: The following errors were found for the dumped [grads] data for dump 0.
        Up to 3 most significant errors are shown per tensor.
    Numerical mismatch at path: tensor1[0,0]
      Dumped data: -0.5149053931236267
      Struct to compare: 0.9704018831253052
      Difference (current - dumped): 1.4853072762489319
    Numerical mismatch at path: tensor1[0,9]
      Dumped data: 0.7180529236793518
      Struct to compare: -0.7570282816886902
      Difference (current - dumped): -1.475081205368042
    Numerical mismatch at path: tensor1[1,8]
      Dumped data: -0.33085548877716064
      Struct to compare: 0.9765480756759644
      Difference (current - dumped): 1.307403564453125
    warnings.warn(error_message)


Related examples
----------------

- The extended dumping example (:doc:`tensor_dumper_dumping`) demonstrates more features (converters, per‑tensor
  overrides, RaggedBatch support, custom pre‑dump processing, early exit).
- See :doc:`stopwatch` and :doc:`nvtx_range_wrapper` for the multi‑file singleton usage pattern.


