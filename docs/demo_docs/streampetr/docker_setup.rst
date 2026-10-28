Docker Setup & Running StreamPETR
=================================

Preparing the Docker Image
--------------------------

The Dockerfile at ``demos/streampetr/docker/Dockerfile`` installs the dependencies for ACCV-Lab and
StreamPETR. ACCV-Lab itself must then be installed from the repository, as described below.

Build the image with the demo's build script. Setting ``TAG`` explicitly gives the image the name used
in the run command below; the script otherwise defaults to ``accvlab``.

.. code-block:: bash

   # From the root of the repository
   cd demos/streampetr/docker
   TAG=streampetr:latest bash run_docker_build.sh

Running the Docker Container
-----------------------------

Replace the host paths below with your ACCV-Lab checkout and prepared NuScenes dataset directory.
All Docker options, including mounts, must come before the image name.

.. code-block:: bash

   docker run -it \
     --gpus=all --ipc=host \
     --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
     -v /path/to/ACCV-Lab:/ACCV-Lab \
     -v /path/to/ACCV-Lab/demos/streampetr:/streampetr \
     -v /path/to/nuscenes:/data/nuscenes \
     -w /streampetr \
     streampetr:latest bash

.. important::
    Mount the demo at ``/streampetr`` and the full ACCV-Lab repository at ``/ACCV-Lab``. The training
    script uses both paths: the latter provides the DALI Pipeline Framework example imported by the demo. Both
    mounts must refer to the same checkout.

    The provided configurations expect the prepared dataset under ``/data/nuscenes/``. The reference
    loader reads camera paths stored in the converted ``.pkl`` files verbatim. The documented
    conversion command stores absolute paths, so the dataset must be available at the same
    in-container path during conversion and training. Regenerate the files if that path changes.

    To use another location, update ``data_root`` in the configuration files. DALI uses this same root;
    its NuScenes version is set separately as ``data_loading_config.nuscenes_version``. See
    :doc:`Running the Demos <../demos>` for data-preparation guidance.


.. seealso::
    See the :doc:`ACCV-Lab Docker Guide <../../guides/DOCKER_GUIDE>` for NVIDIA Container Toolkit setup
    and additional runtime options.

Installing ACCV-Lab in the Container
------------------------------------

If ACCV-Lab is not already installed from this checkout, follow the
:doc:`Installation Guide <../../guides/INSTALLATION_GUIDE>`. With the repository's Git submodules
initialized, the standard installation command inside the container is:

.. code-block:: bash

   cd /ACCV-Lab
   ./scripts/install_local.sh
   cd /streampetr

Then follow the configuration guidance and commands in the evaluation section of the demo you want
to run. For example, see the :doc:`Batching Helpers evaluation <batching_helpers/evaluation>` or
the :doc:`DALI Pipeline Framework evaluation <dali_pipeline_framework/evaluation>`.
