# Docker Guide

This guide covers how to use the Docker image for ACCV-Lab.

## Setting up the Docker Image

There is a [Dockerfile](../../docker/Dockerfile) that can be used to set up a Docker image with all the 
dependencies needed to build and install ACCV-Lab, including all the contained namespace packages.
The Dockerfile is located in the `docker` directory of the ACCV-Lab repository.

The Dockerfile uses the [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) image 
(version: `nvidia/cuda:12.4.1-devel-ubuntu22.04`) as the base image.

The Dockerfile has the following optional build arguments:
- `USER_ID`: The ID of the user to use for the container.
- `USER_NAME`: The name of the user to use for the container.
- `GROUP_ID`: The ID of the group to use for the container.
- `GROUP_NAME`: The name of the group to use for the container.

If these are not set, the root user will be used.

The Dockerfile can be built as follows:
```bash
docker build -t accvlab .
```

Additional to the dockerfile, there is the `run_docker_build.sh` script (located in the `docker` directory) that can 
be used to build the container for the current user.

## Running the Docker Image

The container can be run as follows:
```bash
docker run -it accvlab <...additional arguments...>
```

The following arguments are recommended to be used when running the container to ensure optimal performance:
``--gpus=all --cap-add SYS_NICE --privileged --shm-size=16g --ipc=host``.
Note that the optimal shared memory size (`--shm-size`) may depend on the specific use case.

To ensure that the the on-demand video decoder package can be used, the Nvidia container runtime needs to be 
used. The runtime can be installed by following the instructions 
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

After installation, the runtime can be enabled by specifying the arguments:
``--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all``
when running the container.

An example for running the container with this repository, some additional data, and a project making use
of the ACCV-Lab packages could be:
```bash
docker run \
  --gpus=all --cap-add SYS_NICE --privileged --shm-size=16g --ipc=host \
  --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /path/to/accvlab_repo:/accvlab \
  -v /path/to/your/data:/data \
  -v /path/to/your/project:/project \
  --rm -it \
  -u $(id -u):$(id -g) -w /project \
  accvlab:latest
```
If you want to run the container without the Nvidia container runtime, you can remove the 
`--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all` arguments. However, note that this may prevent the 
on-demand video decoder package from working.

Note that in this example, the container is run with the current user `$(id -u):$(id -g)` and the working 
directory is set to `/project`.