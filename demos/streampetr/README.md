# StreamPETR cleaned baseline

This directory contains a **modified copy of
[StreamPETR](https://github.com/exiawsh/StreamPETR)** prepared for inclusion in
[ACCV-Lab](../../README.md). It is an intermediate source snapshot for reviewing
the cleanup separately from the ACCV-Lab integration added by the next commit.

This snapshot is **not** intended as a full StreamPETR distribution. For the
StreamPETR model and its full functionality, use the upstream repository. If
you use StreamPETR in your research, please cite the upstream work and follow
the citation and acknowledgement guidance in that repository.

## What this snapshot changes

**1. Removal of functionality the demo does not use.** Parts of the original
codebase have been removed, including the standalone inference/test launchers,
tracking utilities, additional backbones, the RepDETR3D and PETRv1 model
variants, several unused heads and transformers, and the configuration files
belonging to them.

Some backbones, model variants and configuration files needed to reproduce
StreamPETR's published results are therefore missing from this snapshot. Use
the upstream repository to reproduce those results.

**2. GridMask replacement.** We replaced `GridMask` with
`RandomGridOcclusion` because we could not identify an explicit license for
the original code. The replacement supports the augmentation settings retained
by this snapshot and raises an error for unsupported settings.

**3. Source cleanup and compatibility.** Attribution and license notices have
been added where needed. Mixed-source helper modules have been separated to
make their origins and applicable licenses clearer. This snapshot also includes
small compatibility fixes for the FlashAttention import and multiprocessing
setup.

## Intermediate review scope

This snapshot intentionally does not include runnable demo configurations,
tests, a container environment, or online demo documentation. The next commit
adds the ACCV-Lab integrations and the complete demo setup, tests, and
documentation.

## Licensing and attribution

This snapshot contains code from StreamPETR and other projects used by
StreamPETR. The headers in the source files identify where the code came from
and record its copyright notices and applicable licenses. Third-party license
texts are collected in
[`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).

ACCV-Lab as a whole is distributed under the Apache License 2.0; see
[`LICENSE`](../../LICENSE). Individual files also contain code covered by
third-party licenses, as recorded in their headers.
