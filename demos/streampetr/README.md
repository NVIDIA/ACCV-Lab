# StreamPETR Demo

This directory contains a **modified copy of [StreamPETR](https://github.com/exiawsh/StreamPETR)**, used as
a demo for [ACCV-Lab](../../README.md). It shows how to integrate ACCV-Lab packages into StreamPETR's
training code and measure their effect on training performance.

This demo is **not** intended as a full StreamPETR distribution. Parts of the original codebase that the
demo does not use have been removed, and code for measuring training performance has been added. For the
StreamPETR model and its full functionality, use the upstream repository.

This README describes the differences from the original implementation. For explanations of the
optimizations, environment setup, configuration selection and evaluation commands, see the
[StreamPETR demo documentation](https://nvidia.github.io/ACCV-Lab/demo_docs/demos.html#streampetr).
The online documentation follows ACCV-Lab's `main` branch. To read documentation matching a different
checkout, build it locally using the [ACCV-Lab Quick Start](../../README.md#quick-start).

## Upstream StreamPETR

For documentation about the StreamPETR model itself, including its supported features and instructions
for training and inference, see the [upstream repository](https://github.com/exiawsh/StreamPETR). It also
provides links to the paper, published results and pretrained checkpoints:

| Topic | Where |
| --- | --- |
| Method and paper | [arXiv:2303.11926](https://arxiv.org/abs/2303.11926) |
| Published results and pretrained checkpoints | [StreamPETR](https://github.com/exiawsh/StreamPETR): `README.md` |
| Environment setup | [StreamPETR](https://github.com/exiawsh/StreamPETR): `docs/setup.md` |
| Data preparation | [StreamPETR](https://github.com/exiawsh/StreamPETR): `docs/data_preparation.md` |
| Training and inference | [StreamPETR](https://github.com/exiawsh/StreamPETR): `docs/training_inference.md` |

The demo requires a different software environment from the original StreamPETR implementation, but
uses the same dataset format. Setup instructions for the demo are in the ACCV-Lab documentation linked
above.

If you use StreamPETR in your research, please cite the upstream work and follow the citation and
acknowledgement guidance in the upstream repository.

## What this demo changes

The main changes to the original StreamPETR code are described below.

**1. Additional ACCV-Lab implementations.** The demo adds implementations using Batching Helpers for
loss computation, the DALI Pipeline Framework for data loading, and Draw Heatmap for heatmap generation.
The original implementations are retained, with configuration options to select the implementation used
for a comparison. The optimization details and the configurations used to evaluate them are described
in the demo documentation.

**2. Performance measurement and profiling.** The training code includes NVTX markers for identifying
operations in a profiler and timers for measuring runtime, including in the `mmcv` runner and optimizer
hooks. Configuration files and helper scripts are provided for profiling and evaluation. An option to
disable shuffling of sample groups allows baseline and optimized runs to process samples in the same
order. Shuffling remains enabled by default, as in the original implementation.

For mini-dataset performance comparisons, the demo can also give each GPU the full
dataset instead of a shard. The behavior and limitations are described in the DALI Pipeline Framework
evaluation documentation.

**3. Removal of functionality the demo does not use.** Parts of the original codebase have been removed,
including the standalone inference/test launchers, tracking utilities, additional backbones, the RepDETR3D
and PETRv1 model variants, several unused heads and transformers, and the configuration files belonging to
them.

Some backbones, model variants and configuration files needed to reproduce StreamPETR's published
results are therefore missing from this demo. Use the upstream repository to reproduce those results.

**4. GridMask replacement.**
We replaced `GridMask` with `RandomGridOcclusion` because we could not identify an explicit license for
the original code. This replacement applies to both baseline and optimized runs. It supports the
augmentation settings used by this demo and raises an error for unsupported settings.

The demo's baseline implementations still include shared changes such as the performance-measurement
code and the GridMask replacement. Evaluation configurations may also apply other optimizations to both
variants, such as the custom heatmap operator in the batching comparison. Disabling the optimization
being evaluated therefore does not restore an unmodified StreamPETR implementation; each evaluation
page explains what its baseline includes.

## Licensing and attribution

This demo contains code from StreamPETR and other projects used by StreamPETR. The headers in the source
files identify where the code came from and record its copyright notices and applicable licenses.
Third-party license texts are collected in
[`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).

ACCV-Lab as a whole is distributed under the Apache License 2.0; see [`LICENSE`](../../LICENSE). Individual
files also contain code covered by third-party licenses, as recorded in their headers.
