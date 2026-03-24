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

"""
Evaluation script for measuring the runtime of multi_tensor_copier vs. naive PyTorch .to() copying.

Uses the same data structure as the example (multi-camera 3D detection meta-data).
"""

import time

import torch
import numpy as np

import accvlab.multi_tensor_copier as mtc

try:
    from example import create_batch_meta_data
except ImportError:
    raise ImportError(
        "Could not import 'example'. Please run this script from the "
        "'packages/multi_tensor_copier/example/' directory."
    ) from None


NUM_RUNS = 10
NUM_WARMUP = 100
NUM_ITERATIONS = 1000
DEVICE = "cuda:0"


def copy_batch_meta_data_naive(batch_meta_data, device):
    """Copy batch meta-data to device using per-tensor .to() calls with known structure.

    This is representative of what a manual implementation would look like when the data layout
    is known at development time (as is typically the case in a training pipeline).
    """
    gpu_batch = []
    for sample in batch_meta_data:
        gpu_cams = []
        for cam in sample["cams_gt"]:
            gpu_cams.append(
                {
                    "bounding_boxes": cam["bounding_boxes"].to(device),
                    "class_ids": cam["class_ids"].to(device),
                    "active": cam["active"].to(device),
                    "depths": cam["depths"].to(device),
                    "proj_mat": cam["proj_mat"].to(device),
                }
            )
        gpu_sample = {
            "cams_gt": gpu_cams,
            "gt_data": {
                "bounding_boxes_3d": sample["gt_data"]["bounding_boxes_3d"].to(device),
                "class_ids": sample["gt_data"]["class_ids"].to(device),
                "active": sample["gt_data"]["active"].to(device),
            },
        }
        gpu_batch.append(gpu_sample)
    return gpu_batch


def copy_nested_to_device_generic(data, device):
    """Recursively copy all tensors in a nested structure to `device` using .to().

    Generic version that walks the structure without knowing its layout.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [copy_nested_to_device_generic(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(copy_nested_to_device_generic(item, device) for item in data)
    if isinstance(data, dict):
        return {k: copy_nested_to_device_generic(v, device) for k, v in data.items()}
    return data


def benchmark(copy_fn, batch_meta_data, device, num_warmup, num_iterations):
    for _ in range(num_warmup):
        _ = copy_fn(batch_meta_data, device)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = copy_fn(batch_meta_data, device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def benchmark_mtc(batch_meta_data, device, num_warmup, num_iterations):
    for _ in range(num_warmup):
        handle = mtc.start_copy(batch_meta_data, device)
        _ = handle.get()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        handle = mtc.start_copy(batch_meta_data, device)
        _ = handle.get()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def mean_ms(times_s):
    return np.mean(times_s) * 1000


def count_tensors(data):
    """Count the total number of tensors in a nested structure."""
    if isinstance(data, torch.Tensor):
        return 1
    if isinstance(data, (list, tuple)):
        return sum(count_tensors(item) for item in data)
    if isinstance(data, dict):
        return sum(count_tensors(v) for v in data.values())
    return 0


def total_bytes(data):
    """Compute total bytes of all tensors in a nested structure."""
    if isinstance(data, torch.Tensor):
        return data.numel() * data.element_size()
    if isinstance(data, (list, tuple)):
        return sum(total_bytes(item) for item in data)
    if isinstance(data, dict):
        return sum(total_bytes(v) for v in data.values())
    return 0


def main():
    batch_meta_data = create_batch_meta_data()

    n_samples = len(batch_meta_data)
    n_tensors = count_tensors(batch_meta_data)
    n_bytes = total_bytes(batch_meta_data)

    print("=" * 70)
    print("multi_tensor_copier evaluation")
    print("=" * 70)
    print(f"  Batch size:      {n_samples} samples")
    print(f"  Total tensors:   {n_tensors}")
    print(f"  Total bytes:     {n_bytes} ({n_bytes / 1024:.1f} KB)")
    print(f"  Target device:   {DEVICE}")
    print(f"  Runs:            {NUM_RUNS}")
    print(f"  Warmup iters:    {NUM_WARMUP} (per run)")
    print(f"  Measured iters:  {NUM_ITERATIONS} (per run)")
    print()

    hardcoded_means = []
    generic_means = []
    mtc_means = []

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}...")

        hardcoded_times = benchmark(
            copy_batch_meta_data_naive, batch_meta_data, DEVICE, NUM_WARMUP, NUM_ITERATIONS
        )
        hardcoded_means.append(mean_ms(hardcoded_times))

        generic_times = benchmark(
            copy_nested_to_device_generic, batch_meta_data, DEVICE, NUM_WARMUP, NUM_ITERATIONS
        )
        generic_means.append(mean_ms(generic_times))

        mtc_times = benchmark_mtc(batch_meta_data, DEVICE, NUM_WARMUP, NUM_ITERATIONS)
        mtc_means.append(mean_ms(mtc_times))

    hardcoded_means = np.array(hardcoded_means)
    generic_means = np.array(generic_means)
    mtc_means = np.array(mtc_means)

    speedups_hardcoded = hardcoded_means / mtc_means
    speedups_generic = generic_means / mtc_means

    print()
    print("Results (mean runtime per run, aggregated over runs):")
    print("-" * 70)
    print(f"  .to() hardcoded:     {hardcoded_means.mean():.3f} +/- {hardcoded_means.std():.3f} ms")
    print(f"  .to() generic:       {generic_means.mean():.3f} +/- {generic_means.std():.3f} ms")
    print(f"  multi_tensor_copier: {mtc_means.mean():.3f} +/- {mtc_means.std():.3f} ms")
    print()
    print(
        f"  Speedup vs .to() hardcoded: {speedups_hardcoded.mean():.2f}x +/- {speedups_hardcoded.std():.2f}x"
    )
    print(f"  Speedup vs .to() generic:   {speedups_generic.mean():.2f}x +/- {speedups_generic.std():.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
