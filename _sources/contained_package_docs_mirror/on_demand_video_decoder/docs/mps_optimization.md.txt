# MPS Optimization

## Why MPS

A typical PyTorch training setup uses a `DataLoader` with multiple worker processes for video decoding.
Each worker process creates its own CUDA context. When several CUDA contexts share the same GPU, the
driver serializes them through time-slicing: only one context runs at a time, and switching between
them has measurable overhead.

During the forward and backward pass, the training process and the decoder workers compete for the GPU.
The decoder workers interrupt the training compute, and the training process interrupts the decoder
workers in return. The result is that both sides run slower than they would in isolation.

CUDA MPS (Multi-Process Service) addresses this by running a single server-side CUDA context on behalf
of all client processes. The MPS server multiplexes their work onto the GPU without time-slicing,
allowing the NVDEC decoder and the SM compute to overlap genuinely.

## Quick Start

### Managed by accvlab (recommended)

`accvlab-mps` starts the MPS daemon before the training process and shuts it down when training exits.
No changes to the training script are needed.

```bash
# Single-GPU
accvlab-mps python train.py

# Multi-GPU with torchrun
accvlab-mps torchrun --nproc_per_node=8 train.py
```

### Manual startup

Start and stop the MPS daemon yourself, without relying on accvlab:

```bash
# Start
export CUDA_MPS_PIPE_DIRECTORY=/tmp/my-mps
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
nvidia-cuda-mps-control -d

# Run training — the process picks up CUDA_MPS_PIPE_DIRECTORY automatically
python train.py

# Stop
echo quit | nvidia-cuda-mps-control
```

> **⚠️ Warning**: If the training process exits without running the `quit` command, the MPS daemon
> keeps running in the background. Subsequent processes that inherit `CUDA_MPS_PIPE_DIRECTORY` will
> continue to connect to it. Stop it explicitly when it is no longer needed.

## Requirements

- `nvidia-cuda-mps-control` in `PATH` (ships with the CUDA toolkit)
- GPU compute mode set to **Default** or **Exclusive_Process**
  (`nvidia-smi -c 0` for Default, `nvidia-smi -c 3` for Exclusive_Process)
- Linux only

## Limitations

- **Windows is not supported.** `accvlab-mps` relies on `fork`/`exec`, which is not available on Windows.
- MPS has the following general constraints; consult the official documentation for the full list:
  - CUDA debuggers (`cuda-gdb`, `compute-sanitizer`) are not compatible with MPS.
  - All MPS clients must run as the same OS user.
  - If the MPS server crashes, all client processes lose their CUDA context and cannot recover.
  - A small subset of CUDA APIs are not supported under MPS.

## References

- [NVIDIA MPS documentation](https://docs.nvidia.com/deploy/mps/index.html)
