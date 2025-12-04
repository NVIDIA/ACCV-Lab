Evaluation
==========

This page summarizes the performance impact of batching the loss computation using
``accvlab.batching_helpers`` in a StreamPETR training setup.

Setup
-----

Experiment Setup
~~~~~~~~~~~~~~~~

For this evaluation, the training is performed for the NuScenes mini dataset, using the StreamPETR model.
The adaptations to the `original StreamPETR implementation <https://github.com/exiawsh/StreamPETR>`_
are as follows:

.. rubric:: StreamPETR – Optimization Overview

- **HungarianAssigner3D & HungarianAssigner2D**: Original matchers operate on a per‑sample basis.

  - Cost matrix computation (pre‑requisite for actual matching) → **Optimized** (following the approach outlined in the :doc:`example`)
  - Matching itself (SciPy implementation on the CPU).
  - `HungarianAssigner3D`: `nan_to_num()` was a bottleneck; moved to GPU → Changed in both reference & optimized.

- **StreamPETR Head**

  - Loss computation is batched over samples → **Optimized**.
  - Loss computation is also batched over the decoder layers → **Optimized** (using multiple batch dimensions).
  - Can use a batched assigner → **Optimized**.

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
  the :doc:`example`.

The evaluation is performed for a **batch size of 8** (in contrast to the original configuration of 2) to
obtain a realistic setup and highlight the performance impact of the batching in this case.

.. note::

  We are planning to add a demo for the Batching Helpers package in the future, including the implementation 
  of the experiments performed in this evaluation.

Hardware Setup
~~~~~~~~~~~~~~

.. list-table:: System Configuration
   :header-rows: 1

   * - GPU 
     - CPU
   * - NVIDIA A100-SXM4-80GB
     - 2x AMD EPYC 7742 64-core Processors


Results
-------

.. only:: html

  In the following table, the runtime breakdown of the training iteration is shown.
  Note that the grayed out cells do not contain any optimized code. While theoretically, the ``Optimization``
  step contains changes (due to the deifferent implementation of the loss, leading to different steps
  in the packward propagation), the resulting runtime differences in this step are negligible.
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
           <td colspan="5" class="bh-pairA-value">760 → 615 (× 1.24)</td>
         </tr>
         <tr>
           <th colspan="3" class="bh-pairB-title">Forward (including Loss)</th>
           <th class="bh-gray-title">Optimization</th>
           <th class="bh-gray-title">Remaining</th>
         </tr>
         <tr>
           <td colspan="3" class="bh-pairB-value">363 → 221 (× 1.64)</td>
           <td class="bh-gray-value">318 → 314</td>
           <td class="bh-gray-value">79 → 80</td>
         </tr>
         <tr>
           <th class="bh-gray-title">Remaining</th>
           <th colspan="2" class="bh-pairA-title">Loss</th>
           <td class="bh-ghost"></td>
           <td class="bh-ghost"></td>
         </tr>
         <tr>
           <td class="bh-gray-value">180 → 180</td>
           <td colspan="2" class="bh-pairA-value">183 → 41 (× 4.46)</td>
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
           <td class="bh-pairB-value">82 → 25 (× 3.28)</td>
           <td class="bh-pairB-value">100 → 16 (× 6.25)</td>
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
       - 760
       - 615
       - × 1.24
     * - Forward (including Loss) [within Training Iteration]
       - 363
       - 221
       - × 1.64
     * - Optimization [within Training Iteration]
       - 318
       - 314
       - —
     * - Remaining [within Training Iteration]
       - 79
       - 80
       - —
     * - Remaining [within Forward]
       - 180
       - 180
       - —
     * - Loss [within Forward]
       - 183
       - 41
       - × 4.46
     * - StreamPETR Head [within Loss]
       - 82
       - 25
       - × 3.28
     * - Focal Head [within Loss]
       - 100
       - 16
       - × 6.25


The batching of the loss computation leads to a speedup of **× 4.46** for the loss computation itself, with
different speedups achieved for the different types of loss. The loss optimization translates
to an overall speedup of **× 1.64** for the forward pass and **× 1.24** for the training iteration. Note that 
the expected speedup strongly depends on the used batch size.
