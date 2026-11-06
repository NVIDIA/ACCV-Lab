# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID=0
ARG USER_NAME=root
ARG GROUP_ID=0
ARG GROUP_NAME=root

RUN (getent group ${GROUP_ID} || groupadd --gid ${GROUP_ID} ${GROUP_NAME}) && \
    (getent passwd ${USER_ID} || useradd --gid ${GROUP_ID} --uid ${USER_ID} --create-home --no-log-init --shell /bin/bash ${USER_NAME})

USER root

# From PyNvOnDemandDecoder
#apt install freeglut3-dev libglew-dev mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev mesa-utils libvulkan-dev pkg-config

RUN apt-get update &&\
    apt-get install -y sudo && \
    apt-get install -y cmake && \
    apt-get install -y gdb && \
    apt-get install -y valgrind && \
    apt-get install -y git && \
    apt-get install -y python3 python3-dev python3-pip && \
    apt-get install -y python-is-python3 && \
    apt-get install -y htop && \
    apt-get install -y nano && \
    apt-get install -y freeglut3-dev && \
    apt-get install -y libglew-dev && \
    apt-get install -y mesa-common-dev && \
    apt-get install -y libegl1-mesa-dev && \
    apt-get install -y libgles2-mesa-dev && \
    apt-get install -y mesa-utils && \
    apt-get install -y libvulkan-dev && \
    apt-get install -y pkg-config && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y clang-format && \
    apt-get install -y enchant-2 && \
    apt-get install -y git-lfs && \
    git lfs install

# Set up FFmpeg (with reduced available functionality) to be used for the On-demand Video Decoder Package.
RUN apt install yasm nasm -y && \
    apt install wget -y && \
    cd /tmp && \
    mkdir ffmpeg && \
    cd ffmpeg && \
    wget https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz && \
    tar -xf ffmpeg-4.4.6.tar.xz && \
    cd ffmpeg-4.4.6 && \
    LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9 --arch=x86_64 --prefix=/ffmpeg && \
    make -j && \
    make install
ENV FFMPEG_DIR=/ffmpeg

# LaTeX toolchain and fonts for Sphinx PDF (XeLaTeX + xindy + Graphviz)
RUN apt-get update &&\
    apt-get install -y latexmk texlive-xetex texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra xindy graphviz fonts-dejavu

# Locales (for sphinx)
#   Install locales package
RUN apt-get update && apt-get install -y locales
#   Generate en_US.UTF-8 locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
#   Set environment variables
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

RUN pip install numpy==1.23.5 \
    matplotlib==3.10.1 \
    scipy==1.15.3 \
    opencv-python-headless==4.5.5.64

RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

RUN pip install black==26.5.1

RUN pip install sphinx==8.1.3 \
    sphinx-rtd-theme==3.1.0 \
    sphinx-autodoc-typehints==3.0.1 \
    sphinx-autobuild==2024.10.3 \
    linkify-it-py==2.1.0 \
    myst-parser==4.0.1 \
    sphinxcontrib-spelling==8.0.2

RUN pip install ninja \
    scikit-build \
    PyYAML==6.0.3

RUN pip install pycuda==2025.1.2 \
    pybind11==3.0.0 \
    cvcuda-cu12==0.15.0

RUN pip install pandas==1.5.3 \
    IPython \
    nvidia-dali-cuda120==2.3.0 \
    packaging==26.2 \
    nvtx \
    psutil \
    numba==0.59 \
    pyquaternion==0.9.9

RUN pip install nuscenes-devkit==1.2.0 && \
    pip install shapely==2.0.7 tqdm==4.67.3 pillow==12.2.0 networkx==3.4.2 fire==0.7.1

RUN pip install pytest pytest-timeout
RUN pip install pynvml

RUN apt-get install -y libjpeg-dev zlib1g-dev

RUN pip install --upgrade pip setuptools==80.9.0 wheel "setuptools-scm>=8" packaging==26.2

RUN pip install cupy-cuda12x==13.6.0

# ========== Corresponds to accvlab Dockerfile above; StreamPETR-specific below ==========

# Some of the installations below require an older version of setuptools. Temporarily install it.
RUN pip install setuptools==59.6

# Note that here, we make a change to `_functions.py` to make it work with the current version of PyTorch.
RUN mkdir /additional_packages
RUN cd /additional_packages &&\
    git clone --recursive --branch=v1.7.2 https://github.com/open-mmlab/mmcv.git &&\
    cd /additional_packages/mmcv/mmcv/parallel &&\
    sed -i _functions.py -e 's/_get_stream(device)/_get_stream(torch.device("cuda", device))/g'
RUN cd /additional_packages/mmcv &&\
    MAX_JOBS=4 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0+PTX" FORCE_CUDA=1 pip install . --no-build-isolation

RUN pip install mmdet==2.28.2 mmsegmentation==0.29.1

# Note that here, we delete some lines in requirement files, __init__.py and setup.py.
# Some of these lines are version checks which would fail (as we use a different version of mmcv),
# while others are dependencies which are already installed, but with a different version.
# This is a workaround to make the installation work.
RUN cd /additional_packages &&\
    git clone --recursive --branch=v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git &&\
    cd /additional_packages/mmdetection3d/requirements &&\
    sed -i '2,4d' runtime.txt &&\
    sed -i '1d' mminstall.txt &&\
    cd /additional_packages/mmdetection3d/mmdet3d &&\
    sed -i '26,29d' __init__.py &&\
    cd /additional_packages/mmdetection3d &&\
    sed -i '228d' setup.py &&\
    sed -i '35,64d' setup.py &&\
    sed -i '9,11d' setup.py

# Ensure that we keep the current version of `numpy`. Also, use the same version of `opencv-python` as version
# of `opencv-python-headless` (used above). As here, we encounter a dependency to `opencv-python` instead of
# `opencv-python-headless`, it will be installed although we already have the headless version.
# Using the same version ensures consistency.
RUN cd /additional_packages/mmdetection3d && pip install "numpy==1.23.5" "opencv-python==4.5.5.64" . --no-build-isolation

RUN pip install einops==0.8.1
RUN MAX_JOBS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --no-build-isolation flash-attn==2.8.3.post1
RUN pip install fvcore==0.1.5.post20221221

# Reverse the temporary installation of the older version of setuptools. Reassert setuptools-scm because
# ACCV-Lab package builds run without build isolation and require it in the active environment.
RUN pip install --upgrade pip setuptools==80.9.0 "setuptools-scm>=8" packaging==26.2

# Ubuntu installs PyGObject as part of the documentation toolchain without its
# split Python Cairo binding, which makes `pip check` report a missing dependency.
RUN apt-get update && apt-get install -y python3-cairo

WORKDIR /workspace

USER ${USER_NAME}
