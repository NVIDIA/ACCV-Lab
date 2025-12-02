# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install black

RUN pip install sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints \
    sphinx-autobuild \
    linkify-it-py \
    myst-parser \
    sphinxcontrib-spelling

RUN pip install ninja \
    scikit-build

RUN pip install pycuda==2025.1.1 \
    pybind11==3.0.0 \
    cvcuda-cu12==0.15.0

RUN pip install pandas==1.5.3 \
    IPython \
    nvidia-dali-cuda120==1.51.2 \
    nvtx \
    psutil \
    numba==0.59 \
    pyquaternion==0.9.9

RUN pip install nuscenes-devkit && \
    pip install shapely tqdm pillow networkx fire

RUN pip install pytest pytest-timeout
RUN pip install pynvml

RUN apt-get install -y libjpeg-dev zlib1g-dev

RUN pip install --upgrade pip setuptools==80.9.0 wheel

RUN pip install cupy

WORKDIR /workspace

USER ${USER_NAME}