Demos
=====

These demos show how to integrate ACCV-Lab packages into training code and measure their performance.

StreamPETR
----------
This is a collection of demos based on the StreamPETR model. The implementation is based on the open-source
release of StreamPETR (https://github.com/exiawsh/StreamPETR).

This documentation explains the optimizations, environment setup, configuration files and evaluation
commands.

.. note::

   For the differences from the original StreamPETR implementation, including removed functionality,
   additional ACCV-Lab implementations, and the GridMask replacement, read
   ``demos/streampetr/README.md`` in the ACCV-Lab repository. That README describes the code in its
   checkout; use it together with documentation from the same revision.

For license and attribution information about third-party code used in ACCV-Lab and the StreamPETR demo, see 
:doc:`Third-Party Licenses <../third_party_licenses>`.

.. important:: 
    The StreamPETR demos share one common implementation under ``demos/streampetr/`` in the ACCV-Lab repository.
    The documentation is organized by package. Configuration files select the baseline or
    optimized implementation for each comparison. See the corresponding evaluation page for further details.

The tutorials explain the integration into StreamPETR as a representative real-world example. Stand-alone examples for the use
of the individual packages are available in the corresponding package documentation.

Demos for Contained Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple modifications have been made to the original implementation to demonstrate different contained
packages. This includes demos of:

.. list-table::
   :header-rows: 1
   :widths: 75 25
   :class: no-scroll

   * - Description
     - Link
   * - Use of ``batching-helpers``:
        - Batches the loss computation (which is already partially batched in the
          original implementation).
        - Demo contains a tutorial on how to implement the batching for both matching and loss computation.
     - :doc:`streampetr/batching_helpers/index`
   * - Use of ``dali-pipeline-framework``:
        - An efficient alternative to the PyTorch DataLoader used in the original implementation is
          implemented.
        - Demo contains a tutorial on how to use the DALI Pipeline Framework to build the data-loading
          pipeline.
     - :doc:`streampetr/dali_pipeline_framework/index`
   * - Use of ``optim-test-tools``:
        - Enables runtime evaluation and profiling of the optimizations. This functionality is used in the
          other demos to perform performance evaluation.
        - Enables dumping of the training input data for debugging/demonstration purposes. This functionality
          is used in the DALI Pipeline Framework demo for debugging/demonstration purposes.
     - :doc:`streampetr/optim_test_tools/index`

.. toctree::
   :hidden:
   :maxdepth: 2

   streampetr/batching_helpers/index
   streampetr/dali_pipeline_framework/index
   streampetr/optim_test_tools/index


Running the Demos
^^^^^^^^^^^^^^^^^

To run the demos, you need to perform the steps described in the table below.

.. list-table::
   :header-rows: 1
   :widths: 25 75
   :class: no-scroll

   * - Step
     - Description
   * - 1. Set up the environment
     - Set up the Docker image and container as described in the
       :doc:`StreamPETR Docker setup <streampetr/docker_setup>`.
   * - 2. Prepare the data
     - The StreamPETR demo uses the NuScenes dataset (the mini version in our
       evaluations), and assumes that the data was converted as needed for StreamPETR (see the 
       ``Data Preparation`` documentation of the original StreamPETR repository, linked from its README: 
       https://github.com/exiawsh/StreamPETR).

       For the provided mini-dataset configurations, generate the reference loader's metadata from
       inside the container with:

       .. code-block:: bash

          cd /streampetr
          python tools/create_data_nusc.py \
            --root-path /data/nuscenes \
            --version v1.0-mini \
            --extra-tag nuscenes2d

       This creates ``nuscenes2d_temporal_infos_train.pkl`` and
       ``nuscenes2d_temporal_infos_val.pkl`` under ``/data/nuscenes``. With this command, the camera
       paths stored in these files are absolute and are used directly by the reference loader. Keep
       the same in-container dataset location during conversion and training, and regenerate the files
       if the location is changed.

       The DALI Pipeline Framework data loader generates its projected annotations and
       ``v1.0-mini_preproc_streampetr.pkl`` cache automatically on the first run; the dataset
       directory must be writable to store it.
   * - 3. Run the package-specific demo
     - Run the demo for the contained package you are interested in by following
       the steps outlined in the demo description (typically in the evaluation section).
       For example, for the ``batching-helpers`` demo, see the
       :doc:`Batching Helpers evaluation <streampetr/batching_helpers/evaluation>` page for commands.

.. toctree::
   :hidden:
   :maxdepth: 2

   streampetr/docker_setup


Choosing a Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The configuration files are under ``demos/streampetr/projects/configs/StreamPETR/``.
They include configurations for the reference demo, Batching Helpers, DALI Pipeline Framework data loading,
and profiling.
For the appropriate configuration, comparison baseline, supported GPU counts, and commands, see the
evaluation section of the corresponding demo, for example the
:doc:`Batching Helpers evaluation <streampetr/batching_helpers/evaluation>` or
:doc:`DALI Pipeline Framework evaluation <streampetr/dali_pipeline_framework/evaluation>`.


Measurement and Profiling
^^^^^^^^^^^^^^^^^^^^^^^^^

Use the Optimization Testing Tools to measure and inspect the training process:

  - Runtime evaluation
  - NVTX range-based profiling (including optional CUDA synchronizations for ranges)
  - Dumping of the training input data for debugging/demonstration purposes

See :doc:`streampetr/optim_test_tools/index` for configuration options and usage instructions, and the
:doc:`optim-test-tools package documentation <../contained_package_docs_mirror/optim_test_tools/docs/index>`
for the API.
