Evaluation
==========

This page summarizes the performance impact of batching the loss computation using
``accvlab.batching_helpers`` in a StreamPETR training setup.

.. seealso::

   For the implementation changes behind these results, see the :doc:`Batching Helpers demo <index>`
   and its tutorials on batching the matching and loss computations. The
   :doc:`Optimization Testing Tools demo <../optim_test_tools/index>` explains the runtime measurements
   and profiling support used in the evaluation.

Setup
-----

Experiment Setup
~~~~~~~~~~~~~~~~

For this evaluation, the training is performed for the NuScenes mini dataset, using the StreamPETR model.
The adaptations to the `original StreamPETR implementation <https://github.com/exiawsh/StreamPETR>`_
are as follows:

.. rubric:: StreamPETR – Optimization Overview

- **HungarianAssigner3D & HungarianAssigner2D**: Original matchers operate on a per‑sample basis.

  - Cost matrix computation (pre‑requisite for actual matching) → **Optimized** (following the approach 
    outlined in the 
    :doc:`batching-helpers example <../../../contained_package_docs_mirror/batching_helpers/docs/example>`)
  - Matching itself (SciPy implementation on the CPU).
  - `HungarianAssigner3D`: `nan_to_num()` was a bottleneck; moved to GPU → Changed in both reference & 
    optimized.

- **StreamPETR Head**

  - The reference already concatenates the samples and calls each classification and bounding-box loss
    function once per decoder layer. The loss functions themselves were therefore already batched over
    samples.
  - Target preparation around those loss calls is further batched → **Optimized**. The optimized head keeps
    the sample dimension in ``RaggedBatch`` objects, computes the assignment cost matrices and performs
    target indexing in batches instead of calling ``_get_target_single()`` for each sample, and computes
    bounding-box loss only for matched pairs. The SciPy Hungarian solver itself still runs once for each
    layer/sample entry.
  - Loss computation is batched across decoder layers → **Optimized**. The reference calls
    ``loss_single()`` and ``dn_loss_single()`` separately for each layer. The optimized head processes the
    layer and sample dimensions together in one loss-function call, then reduces over samples and objects
    while retaining separate loss values for each layer. This also applies to the DN losses.

- **Focal Head**

  - Loss computation is already batched over samples & camera images
  - Can use a batched assigner → **Optimized**.
  - Added use of the custom Gaussian heatmap generator → Changed in both reference & optimized.

.. note::
  Some of the changes/optimizations are done in both the reference and optimized implementation (indicated in
  the overview above). These changes are not specific to the batching optimization, and are therefore 
  applicable to both implementations. Applying them in both ensures a fair comparison, with the obtained 
  speedup reflecting the effect of the batching optimizations.

.. seealso::
  For a general overview on how to use the batching helpers to optimize the loss computation, please refer to 
  the :doc:`example <../../../contained_package_docs_mirror/batching_helpers/docs/example>`.

The evaluation is performed for a **batch size of 8** (in contrast to the original configuration of 2) to
obtain a realistic setup and highlight the performance impact of the batching in this case.

Hardware Setup
~~~~~~~~~~~~~~

The hardware setup used for the evaluation is as follows:

.. list-table:: System Configuration
   :header-rows: 1

   * - GPU 
     - CPU
   * - NVIDIA A100-SXM4-80GB
     - 2x AMD EPYC 7742 64-core Processors


Results
-------

The numbers below are averages over 950 measured iterations after 50 warm-up
iterations.

.. only:: html

  In the following table, the runtime breakdown of the training iteration is shown.
  Note that the grayed out cells do not contain any optimized code. While theoretically, the
  ``Optimization (including Backward)``
  step contains changes (due to the different implementation of the loss, leading to different steps
  in the backward propagation), the resulting runtime differences in this step are negligible.
  The ``Remaining`` columns in the table show the runtime of the parts of the implementation for 
  which the runtime is not measured directly. For the ``Forward`` pass, this mostly corresponds to the forward 
  pass through the network itself (as opposed to the loss computation). For the ``Training Iteration``, this 
  may correspond to additional overhead such as e.g. obtaining/waiting for the next batch of data.

  .. raw:: html
  
     <style>
       /* Local styles for the results table; kept scoped to this page */
       table.bh-table {
         border-collapse: collapse;
         margin: 8px auto 16px auto;
         font-size: 0.95rem;
         min-width: 640px;
       }
       table.bh-table th,
       table.bh-table td {
         border: 1px solid #b7c4bb; /* light grey-green grid */
         padding: 6px 10px;
         text-align: center;
         vertical-align: middle;
         white-space: nowrap;
       }
       /* Dark green header */
       table.bh-table .bh-top {
         background: #11795b;
         color: #ffffff;
         font-weight: 700;
       }
       /* Visually remove a cell while keeping column structure */
       table.bh-table .bh-ghost {
         visibility: hidden;
         border: none !important;
         padding: 0 !important;
       }
       /* Layer color pairs - more distinct */
       table.bh-table .bh-pairA-title { background: #cfeadf; font-weight: 700; }
       table.bh-table .bh-pairA-value { background: #e7f6f1; }
       table.bh-table .bh-pairB-title { background: #e6f2d9; font-weight: 700; }
       table.bh-table .bh-pairB-value { background: #f4faea; }
       /* Neutral (no optimization) pair */
       table.bh-table .bh-gray-title { background: #e9ecef; color: #2f3b3a; font-weight: 700; }
       table.bh-table .bh-gray-value { background: #f6f7f9; color: #2f3b3a; }
     </style>
   
     <table class="bh-table">
       <tbody>
         <tr>
           <th colspan="5" class="bh-top">Runtime Baseline → Runtime Optimized [ms] (Speedup ×-fold)</th>
         </tr>
         <tr>
           <th colspan="5" class="bh-pairA-title">Training Iteration</th>
         </tr>
         <tr>
           <td colspan="5" class="bh-pairA-value">737 → 602 (× 1.22)</td>
         </tr>
         <tr>
           <th colspan="3" class="bh-pairB-title">Forward (including Loss)</th>
           <th class="bh-gray-title">Optimization (including Backward)</th>
           <th class="bh-gray-title">Remaining</th>
         </tr>
         <tr>
           <td colspan="3" class="bh-pairB-value">344 → 210 (× 1.64)</td>
           <td class="bh-gray-value">311 → 307</td>
           <td class="bh-gray-value">81 → 84</td>
         </tr>
         <tr>
           <th class="bh-gray-title">Remaining</th>
           <th colspan="2" class="bh-pairA-title">Loss</th>
           <td class="bh-ghost"></td>
           <td class="bh-ghost"></td>
         </tr>
         <tr>
           <td class="bh-gray-value">169 → 170</td>
           <td colspan="2" class="bh-pairA-value">175 → 40 (× 4.35)</td>
           <td class="bh-ghost"></td>
           <td class="bh-ghost"></td>
         </tr>
         <tr>
           <td class="bh-ghost"></td>
           <th class="bh-pairB-title">StreamPETR Head</th>
           <th class="bh-pairB-title">Focal Head</th>
           <td class="bh-ghost"></td>
           <td class="bh-ghost"></td>
         </tr>
         <tr>
           <td class="bh-ghost"></td>
           <td class="bh-pairB-value">79 → 24 (× 3.22)</td>
           <td class="bh-pairB-value">96 → 15 (× 6.24)</td>
           <td class="bh-ghost"></td>
           <td class="bh-ghost"></td>
         </tr>
       </tbody>
     </table>

.. only:: not html

  In the following table, the runtime breakdown of the training iteration is shown.
  Note that the individual lines in the table correspond to different parts of the implementation, with
  increasing level of detail for the lower entries, which correspond to parts of the implementation
  of upper entries (containing implementation indicated as `[within <...>]`).

  .. note::
 
     A structured table showing the runtime breakdown of the training iteration visually is available in the 
     HTML version of this document.
 
  .. list-table:: Runtime Summary (Baseline → Optimized)
     :header-rows: 1
 
     * - Component
       - Baseline [ms]
       - Optimized [ms]
       - Speedup
     * - Training Iteration
       - 737
       - 602
       - × 1.22
     * - Forward (including Loss) [within Training Iteration]
       - 344
       - 210
       - × 1.64
     * - Optimization (including Backward) [within Training Iteration]
       - 311
       - 307
       - —
     * - Remaining [within Training Iteration]
       - 81
       - 84
       - —
     * - Remaining [within Forward]
       - 169
       - 170
       - —
     * - Loss [within Forward]
       - 175
       - 40
       - × 4.35
     * - StreamPETR Head [within Loss]
       - 79
       - 24
       - × 3.22
     * - Focal Head [within Loss]
       - 96
       - 15
       - × 6.24


The batching of the loss computation leads to a speedup of **× 4.35** for the loss computation itself, with
different speedups achieved for the different types of loss. The loss optimization translates
to an overall speedup of **× 1.64** for the forward pass and **× 1.22** for the training iteration. Note that
the expected speedup strongly depends on the used batch size.

How to Run
----------

Running the Evaluated Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
  
  This section assumes that you are running the evaluation in the Docker container set up as described in the
  :doc:`StreamPETR Docker setup <../docker_setup>`, with the StreamPETR implementation
  located at ``/streampetr``. Configuration paths below are relative to
  ``demos/streampetr/projects/configs/StreamPETR/``.

The evaluation was performed with these two configurations:

.. list-table:: Evaluated Batching Helpers configurations
   :header-rows: 1
   :widths: 35 65

   * - Configuration
     - Purpose
   * - ``Tests/batching_helpers_ref.py``
     - Non-batched reference with the same GPU ``nan_to_num`` and custom heatmap
       improvements as the optimized configuration.
   * - ``Tests/batching_helpers_test.py``
     - Batched assigners and loss computation. Comparing it with the reference
       isolates the batching gains reported above.

Run the configurations individually with:

.. code-block:: bash

    # Non-batched reference with the same unrelated optimizations as the test configuration.
    # Comparing test against ref therefore isolates the performance gains from batching.
    python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_ref.py

    # Batched implementation, including the same unrelated optimizations.
    python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_test.py

Additional Variant
~~~~~~~~~~~~~~~~~~

``Tests/batching_helpers_orig_impl.py`` is a supplemental near-original variant that
uses CPU ``nan_to_num`` and the original heatmap generation. Comparing the optimized
configuration against this variant also includes gains unrelated to batching, so it is
not the baseline for the batching speedups reported above.

Run it with:

.. code-block:: bash

    python /streampetr/tools/train.py --no-validate /streampetr/projects/configs/StreamPETR/Tests/batching_helpers_orig_impl.py

Running All Variants and Storing the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``demos/streampetr/tools/run_batching_eval.sh`` script runs the two evaluated
configurations and the supplemental near-original variant. Run it with:

.. code-block:: bash

    cd /streampetr/tools
    bash run_batching_eval.sh

The script stores the results in the ``res_batching_eval`` folder in the current working
directory. Refer to the script for more details.
