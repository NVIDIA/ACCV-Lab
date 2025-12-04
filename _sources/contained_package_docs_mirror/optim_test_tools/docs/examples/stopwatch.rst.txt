Stopwatch Example
=================


Overview
--------

.. module-docstring:: ../../examples/stopwatch_example.py

.. seealso::
    
    The code of this example can be found in the repository under 
    ``packages/optim_test_tools/examples/stopwatch_example.py``.

Example
-------

Example Implementation
^^^^^^^^^^^^^^^^^^^^^^^

Please see the notes in the code for more details.

.. note-literalinclude:: ../../examples/stopwatch_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/stopwatch_example.py
   :linenos:
   :lineno-match:

Resulting Output
^^^^^^^^^^^^^^^^

Here, we show an example output for the last time that the stopwatch was used to print the results. Note
that the actual result will vary depending on the actual runtime of the code parts. Note that for 
``meas2`` and ``meas3``, the average runtime per iteration is different from the average runtime per measured 
interval. This is because these measurements were performed multiple times per iteration / only in some 
of the iterations, respectively.

.. code-block:: text

  ######################### Stopwatch #########################
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Num. measured iterations: 11
  . . . . . . . . . . . . . . . . . . . . . . . . . 
  Average runtime per iteration:
    `meas1`: 0.16802375966852362
    `meas2`: 0.060098127885298294
    `meas3`: 0.0036563439802689986
  . . . . . . . . . . . . . . . . . . . . . . . . . 
  Average runtime per measured interval:
    `meas1`: 0.16802375966852362
    `meas2`: 0.030049063942649147
    `meas3`: 0.010054945945739746
  . . . . . . . . . . . . . . . . . . . . . . . . . 
  Total runtime:
    `meas1`: 1.8482613563537598
    `meas2`: 0.6610794067382812
    `meas3`: 0.040219783782958984
  . . . . . . . . . . . . . . . . . . . . . . . . . 
  Mean CPU usage during `meas1`: 2.655618891226586
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  One-time measurements:
    `complete_run`: 3.0058188438415527
    `preparation`: 0.10008645057678223
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #############################################################

Related examples
----------------

- See the :doc:`nvtx_range_wrapper` example for a similar singleton pattern applied to NVTX ranges.
- The tensor dumper examples (:doc:`tensor_dumper_comparison` and :doc:`tensor_dumper_dumping`) also follow
  the singleton pattern and can be enabled once and used across multiple code parts.


