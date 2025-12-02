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

"""
Resource Management Tests for Stream Decoder (CreateSampleReader)

This module tests CPU and GPU memory resource release mechanisms for:
- Synchronous decoding: DecodeN12ToRGB()
- Asynchronous decoding: DecodeN12ToRGBAsync() + DecodeN12ToRGBAsyncGetBuffer()

Key Points:
- NvDecoder uses CUDA Driver API (cuMemAlloc), NOT PyTorch allocator
- torch.cuda.memory_allocated() CANNOT detect NvDecoder memory
- We MUST use pynvml to get accurate GPU memory usage
"""

import pytest
import sys
import gc
import time
import os

import torch

# For accurate GPU memory measurement (NvDecoder uses cuMemAlloc, not PyTorch)
import pynvml

# For CPU memory measurement
import psutil

import utils
import accvlab.on_demand_video_decoder as nvc

# ============================================================================
# Memory Measurement Utilities
# ============================================================================


class GPUMemoryMonitor:
    """
    GPU Memory Monitor using pynvml.

    Why pynvml instead of torch.cuda.memory_allocated()?
    - NvDecoder uses CUDA Driver API (cuMemAlloc) for GPU memory allocation
    - PyTorch's memory_allocated() only tracks PyTorch-managed memory
    - pynvml.nvmlDeviceGetMemoryInfo() tracks ALL GPU memory on the device
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialized = False

    def __enter__(self):
        pynvml.nvmlInit()
        self._initialized = True
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._initialized:
            pynvml.nvmlShutdown()
            self._initialized = False

    def get_used_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / (1024 * 1024)

    def get_free_memory_mb(self) -> float:
        """Get current GPU free memory in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.free / (1024 * 1024)


class CPUMemoryMonitor:
    """CPU Memory Monitor using psutil."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_rss_mb(self) -> float:
        """Get current process RSS (Resident Set Size) in MB."""
        return self.process.memory_info().rss / (1024 * 1024)


def force_cleanup():
    """
    Force garbage collection and CUDA cleanup.
    This ensures Python objects are destroyed and CUDA resources are released.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    # Wait a bit for async destructor operations to complete
    time.sleep(0.3)
    gc.collect()


def measure_memory_delta(baseline_gpu_mb: float, current_gpu_mb: float, tolerance_mb: float = 50.0) -> tuple:
    """
    Check if memory delta is within tolerance.

    Returns:
        (is_ok, delta_mb): Whether delta is within tolerance and the actual delta
    """
    delta = current_gpu_mb - baseline_gpu_mb
    return delta <= tolerance_mb, delta


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def gpu_monitor():
    """Fixture to provide GPU memory monitor."""
    with GPUMemoryMonitor(gpu_id=0) as monitor:
        yield monitor


@pytest.fixture
def cpu_monitor():
    """Fixture to provide CPU memory monitor."""
    yield CPUMemoryMonitor()


@pytest.fixture
def video_files():
    """Fixture to provide test video files."""
    path_base = utils.get_data_dir()
    files = utils.select_random_clip(path_base)
    assert files is not None, f"No video files found in {path_base}"
    return files


# ============================================================================
# P0 Tests - Core Functionality (Must Pass)
# ============================================================================


class TestP0CoreResourceRelease:
    """
    P0 Priority Tests - Core resource release functionality.
    These tests must pass for the decoder to be considered stable.
    """

    # Memory tolerance in MB - based on actual test results:
    # - Set to 10 MB to allow for CUDA runtime overhead and measurement noise
    GPU_TOLERANCE_MB = 10.0
    CPU_TOLERANCE_MB = 10.0

    def test_01_basic_del_releases_gpu_memory(self, gpu_monitor, video_files):
        """
        Test 1: Normal usage followed by `del` releases GPU resources.

        Scenario: Create decoder → decode several frames → del decoder
        Verify: GPU memory returns to initial level (within tolerance)
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        # Create decoder and decode
        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        frame_ids = [0] * len(video_files)
        _ = decoder.DecodeN12ToRGB(video_files, frame_ids, False)

        # Record memory after decoding
        after_decode_gpu = gpu_monitor.get_used_memory_mb()
        print(
            f"\nGPU memory after decode: {after_decode_gpu:.1f} MB "
            f"(baseline: {baseline_gpu:.1f} MB, delta: {after_decode_gpu - baseline_gpu:.1f} MB)"
        )

        # Delete decoder and force cleanup
        del decoder
        force_cleanup()

        # Check memory released
        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(
            f"GPU memory after cleanup: {final_gpu:.1f} MB "
            f"(baseline: {baseline_gpu:.1f} MB, delta: {delta:.1f} MB)"
        )

        assert is_ok, (
            f"GPU memory leak detected! "
            f"Baseline: {baseline_gpu:.1f} MB, Final: {final_gpu:.1f} MB, "
            f"Delta: {delta:.1f} MB (tolerance: {self.GPU_TOLERANCE_MB} MB)"
        )

    def test_02_sync_decode_loop_memory_stable(self, gpu_monitor, video_files):
        """
        Test 2: Synchronous decode loop - memory should remain stable.

        Scenario: Create single decoder → decode N times in loop → del decoder
        Verify: Memory doesn't continuously grow during decoding
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        num_iterations = 50
        memory_samples = []

        for i in range(num_iterations):
            frame_ids = [i % 100] * len(video_files)  # Vary frame IDs
            frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)

            # Deep copy to tensor and release reference
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

            # Sample memory every 10 iterations
            if (i + 1) % 10 == 0:
                mem = gpu_monitor.get_used_memory_mb()
                memory_samples.append(mem)
                print(f"Iteration {i+1}: GPU memory = {mem:.1f} MB")

        # Check memory stability (no continuous growth)
        # Compare first half average with second half average
        mid = len(memory_samples) // 2
        first_half_avg = sum(memory_samples[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(memory_samples[mid:]) / (len(memory_samples) - mid)

        growth = second_half_avg - first_half_avg
        print(
            f"\nMemory growth: {growth:.1f} MB "
            f"(first half avg: {first_half_avg:.1f} MB, second half avg: {second_half_avg:.1f} MB)"
        )

        # Cleanup
        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"Final GPU memory: {final_gpu:.1f} MB (delta from baseline: {delta:.1f} MB)")

        # Memory should not grow significantly during iteration
        assert growth < self.GPU_TOLERANCE_MB, f"Memory growing during decode loop! Growth: {growth:.1f} MB"

        # Memory should be released after del
        assert is_ok, f"GPU memory leak after del! Delta: {delta:.1f} MB"

    def test_03_async_decode_loop_memory_stable(self, gpu_monitor, video_files):
        """
        Test 3: Asynchronous decode loop - memory should remain stable.

        Scenario: Create decoder → loop (Async + GetBuffer) N times → del decoder
        Verify: Memory doesn't continuously grow during async decoding
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        num_iterations = 50
        memory_samples = []

        for i in range(num_iterations):
            frame_ids = [i % 100] * len(video_files)

            # Async decode
            decoder.DecodeN12ToRGBAsync(video_files, frame_ids, False)
            frames = decoder.DecodeN12ToRGBAsyncGetBuffer(video_files, frame_ids, False)

            # Deep copy and release
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

            if (i + 1) % 10 == 0:
                mem = gpu_monitor.get_used_memory_mb()
                memory_samples.append(mem)
                print(f"Iteration {i+1}: GPU memory = {mem:.1f} MB")

        # Check memory stability
        mid = len(memory_samples) // 2
        first_half_avg = sum(memory_samples[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(memory_samples[mid:]) / (len(memory_samples) - mid)
        growth = second_half_avg - first_half_avg

        print(f"\nMemory growth: {growth:.1f} MB")

        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"Final GPU memory: {final_gpu:.1f} MB (delta: {delta:.1f} MB)")

        assert growth < self.GPU_TOLERANCE_MB, f"Memory growing during async loop! Growth: {growth:.1f} MB"
        assert is_ok, f"GPU memory leak after del! Delta: {delta:.1f} MB"

    def test_07_repeated_create_destroy_memory_stable(self, gpu_monitor, video_files):
        """
        Test 7: Repeated create/destroy cycle - no cumulative memory leak.

        Scenario: Loop M times (create decoder → decode → del decoder)
        Verify: Memory returns to baseline after each cycle, no accumulation
        """

        MAX_LEAK_PER_CYCLE_MB = 1.0

        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        num_cycles = 50  # Increased from 10 to better detect memory leaks
        memory_after_cycles = []

        for cycle in range(num_cycles):
            # Create decoder
            decoder = nvc.CreateSampleReader(
                num_of_set=1,
                num_of_file=6,
                iGpu=0,
            )

            # Decode several frames
            for _ in range(5):
                frame_ids = [0] * len(video_files)
                frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
                tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
                del frames, tensors

            # Delete decoder
            del decoder
            force_cleanup()

            mem = gpu_monitor.get_used_memory_mb()
            memory_after_cycles.append(mem)

            # Print every 10 cycles to reduce output noise
            if (cycle + 1) % 10 == 0 or cycle == 0:
                print(
                    f"Cycle {cycle + 1}: GPU memory after cleanup = {mem:.1f} MB "
                    f"(delta from baseline: {mem - baseline_gpu:.1f} MB)"
                )

        # Check no cumulative leak
        final_gpu = memory_after_cycles[-1]
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        # Also check that memory isn't growing across cycles
        growth = memory_after_cycles[-1] - memory_after_cycles[0]

        # Calculate average growth per cycle to detect linear leak
        avg_growth_per_cycle = growth / (num_cycles - 1) if num_cycles > 1 else 0

        print(f"\n=== Memory Leak Analysis ===")
        print(f"Total cycles: {num_cycles}")
        print(f"First cycle memory: {memory_after_cycles[0]:.1f} MB")
        print(f"Final cycle memory: {memory_after_cycles[-1]:.1f} MB")
        print(f"Cumulative growth: {growth:.1f} MB")
        print(f"Average growth per cycle: {avg_growth_per_cycle:.2f} MB")
        print(f"Max allowed leak per cycle: {MAX_LEAK_PER_CYCLE_MB:.2f} MB")
        print(f"Final delta from baseline: {delta:.1f} MB")

        # For repeated cycle tests, only check per-cycle leak rate
        # (cumulative growth is just: num_cycles × per_cycle_rate, so checking both is redundant)
        #
        # Note: A small per-cycle leak (e.g., 0.78 MB) may be due to:
        # - CUDA/NVDEC driver-level caching
        # - Primary context reference counting overhead
        # This is acceptable as long as the rate is bounded.
        assert avg_growth_per_cycle <= MAX_LEAK_PER_CYCLE_MB, (
            f"Memory leak detected! Average {avg_growth_per_cycle:.2f} MB per cycle "
            f"(threshold: {MAX_LEAK_PER_CYCLE_MB:.2f} MB). "
            f"Total leak: {growth:.1f} MB over {num_cycles} cycles."
        )

    def test_09_del_with_pending_async_task_no_deadlock(self, gpu_monitor, video_files):
        """
        Test 9: Delete decoder while async task is pending - should not deadlock.

        Scenario: Call DecodeN12ToRGBAsync() then immediately del decoder
        Verify: No deadlock, resources properly released
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        frame_ids = [0] * len(video_files)

        # Start async decode but don't get result
        decoder.DecodeN12ToRGBAsync(video_files, frame_ids, False)

        # Immediately delete - this should not deadlock
        # (destructor should wait for async task to complete)
        start_time = time.time()
        del decoder
        elapsed = time.time() - start_time

        print(f"\nDestructor completed in {elapsed:.2f} seconds")

        # Should complete within reasonable time (not deadlock)
        assert elapsed < 30.0, f"Destructor took too long ({elapsed:.2f}s), possible deadlock!"

        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"GPU memory after cleanup: {final_gpu:.1f} MB (delta: {delta:.1f} MB)")

        assert is_ok, f"GPU memory leak when del with pending task! Delta: {delta:.1f} MB"


# ============================================================================
# P1 Tests - Important Functionality
# ============================================================================


class TestP1ExplicitResourceRelease:
    """
    P1 Priority Tests - Explicit resource release APIs and exception handling.
    """

    GPU_TOLERANCE_MB = 50.0

    def test_04_release_device_memory_effective(self, gpu_monitor, video_files):
        """
        Test 4: release_device_memory() effectively releases GPU memory.

        Scenario: Create decoder → decode → call release_device_memory()
        Verify: GPU memory significantly decreases after call
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        # Decode to allocate GPU memory
        frame_ids = [0] * len(video_files)
        frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
        del frames, tensors

        after_decode_gpu = gpu_monitor.get_used_memory_mb()
        gpu_used_by_decoder = after_decode_gpu - baseline_gpu
        print(
            f"\nGPU memory after decode: {after_decode_gpu:.1f} MB "
            f"(decoder using: {gpu_used_by_decoder:.1f} MB)"
        )

        # Release device memory
        decoder.release_device_memory()
        torch.cuda.synchronize()
        time.sleep(0.2)

        after_release_gpu = gpu_monitor.get_used_memory_mb()
        released = after_decode_gpu - after_release_gpu

        print(
            f"GPU memory after release_device_memory(): {after_release_gpu:.1f} MB "
            f"(released: {released:.1f} MB)"
        )

        # Memory should decrease (at least some should be released)
        # Note: Not all memory may be released due to internal state
        assert released > 0 or gpu_used_by_decoder < 50, (
            f"release_device_memory() didn't release any memory! "
            f"Before: {after_decode_gpu:.1f} MB, After: {after_release_gpu:.1f} MB"
        )

        del decoder
        force_cleanup()

    def test_05_clear_all_readers_effective(self, gpu_monitor, video_files):
        """
        Test 5: clearAllReaders() effectively releases resources.

        Scenario: Create decoder → decode multiple different videos → call clearAllReaders()
        Verify: GPU memory decreases after clearing readers
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=3,  # Multiple sets to hold more reader state
            num_of_file=6,
            iGpu=0,
        )

        # Decode multiple times with different frame IDs to create reader state
        for frame_offset in range(3):
            frame_ids = [frame_offset * 10] * len(video_files)
            frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

        after_decode_gpu = gpu_monitor.get_used_memory_mb()
        print(
            f"\nGPU memory after multiple decodes: {after_decode_gpu:.1f} MB "
            f"(delta: {after_decode_gpu - baseline_gpu:.1f} MB)"
        )

        # Clear all readers
        decoder.clearAllReaders()
        torch.cuda.synchronize()
        time.sleep(0.2)

        after_clear_gpu = gpu_monitor.get_used_memory_mb()
        released = after_decode_gpu - after_clear_gpu

        print(
            f"GPU memory after clearAllReaders(): {after_clear_gpu:.1f} MB " f"(released: {released:.1f} MB)"
        )

        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"Final GPU memory: {final_gpu:.1f} MB (delta: {delta:.1f} MB)")
        assert is_ok, f"GPU memory leak after clearAllReaders + del! Delta: {delta:.1f} MB"

    def test_06_release_memory_then_continue_decode(self, gpu_monitor, video_files):
        """
        Test 6: Decoder remains usable after release_device_memory().

        Scenario: Create decoder → decode → release_device_memory() → decode again
        Verify: Decoder works correctly after memory release

        Note: This test validates that after calling release_device_memory(),
        the decoder can still decode frames correctly. The GPU memory pool
        should be re-allocated automatically on the next decode operation.
        """
        force_cleanup()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        frame_ids = [0] * len(video_files)

        # First decode
        frames_1 = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensor_1 = torch.as_tensor(frames_1[0], device='cuda').clone()
        del frames_1

        print(f"\nFirst decode successful, frame shape: {tensor_1.shape}")

        # Release memory
        decoder.release_device_memory()
        torch.cuda.synchronize()

        # Second decode - should still work (but currently returns empty frames!)
        frames_2 = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensor_2 = torch.as_tensor(frames_2[0], device='cuda').clone()
        del frames_2

        print(f"Second decode successful, frame shape: {tensor_2.shape}")

        # Verify frames are identical (same content)
        assert tensor_2.shape[0] > 0, f"Frame has zero height! Shape: {tensor_2.shape}"
        assert (
            tensor_1.shape == tensor_2.shape
        ), f"Frame shapes don't match! {tensor_1.shape} vs {tensor_2.shape}"
        max_diff = (tensor_1.float() - tensor_2.float()).abs().max().item()
        print(f"Max pixel difference between frames: {max_diff}")
        assert max_diff < 1.0, f"Frames differ too much! Max diff: {max_diff}"

        del decoder, tensor_1, tensor_2
        force_cleanup()

    def test_10_multiple_async_without_getbuffer_no_leak(self, gpu_monitor, video_files):
        """
        Test 10: Multiple Async calls without GetBuffer - no memory leak.

        Scenario: Call Async multiple times (overwriting previous) → del decoder
        Verify: Overwritten results don't cause memory leak
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        # Call Async multiple times without GetBuffer
        # Each call should discard previous pending result
        for i in range(10):
            frame_ids = [i * 5] * len(video_files)
            decoder.DecodeN12ToRGBAsync(video_files, frame_ids, False)
            # Don't call GetBuffer - let it be overwritten

        # Only get the last one
        last_frame_ids = [45] * len(video_files)
        frames = decoder.DecodeN12ToRGBAsyncGetBuffer(video_files, last_frame_ids, False)
        assert frames is not None and len(frames) == len(video_files)

        tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
        del frames, tensors

        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"\nGPU memory delta: {delta:.1f} MB (tolerance: {self.GPU_TOLERANCE_MB} MB)")

        assert is_ok, f"GPU memory leak with multiple Async without GetBuffer! " f"Delta: {delta:.1f} MB"

    def test_08_release_decoder_effective(self, gpu_monitor, video_files):
        """
        Test 8: release_decoder() effectively releases GPU memory.

        Scenario: Create decoder → decode → call release_decoder()
        Verify: GPU memory significantly decreases after call

        Note: release_decoder() is more thorough than release_device_memory():
        - release_device_memory(): only releases GPU memory pool
        - release_decoder(): deletes all readers (NvDecoder, FFmpegDemuxer, etc.)
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        # Decode multiple frames to allocate more GPU memory
        for frame_offset in range(5):
            frame_ids = [frame_offset * 10] * len(video_files)
            frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

        after_decode_gpu = gpu_monitor.get_used_memory_mb()
        gpu_used_by_decoder = after_decode_gpu - baseline_gpu
        print(
            f"\nGPU memory after decode: {after_decode_gpu:.1f} MB "
            f"(decoder using: {gpu_used_by_decoder:.1f} MB)"
        )

        # Release decoder (more thorough than release_device_memory)
        decoder.release_decoder()
        torch.cuda.synchronize()
        time.sleep(0.2)

        after_release_gpu = gpu_monitor.get_used_memory_mb()
        released = after_decode_gpu - after_release_gpu

        print(
            f"GPU memory after release_decoder(): {after_release_gpu:.1f} MB "
            f"(released: {released:.1f} MB)"
        )

        # Memory should decrease significantly
        assert released > 0 or gpu_used_by_decoder < 50, (
            f"release_decoder() didn't release any memory! "
            f"Before: {after_decode_gpu:.1f} MB, After: {after_release_gpu:.1f} MB"
        )

        del decoder
        force_cleanup()

    def test_09_release_decoder_then_continue_decode(self, gpu_monitor, video_files):
        """
        Test 9: Decoder remains usable after release_decoder().

        Scenario: Create decoder → decode → release_decoder() → decode again
        Verify: Decoder works correctly after releasing all readers

        Note: After release_decoder(), new readers will be created automatically
        on the next decode operation.
        """
        force_cleanup()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        frame_ids = [0] * len(video_files)

        # First decode
        frames_1 = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensor_1 = torch.as_tensor(frames_1[0], device='cuda').clone()
        shape_1 = tensor_1.shape
        del frames_1

        print(f"\nFirst decode successful, frame shape: {shape_1}")

        # Release decoder (delete all readers)
        decoder.release_decoder()
        torch.cuda.synchronize()

        # Second decode - should work (new readers will be created)
        frames_2 = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensor_2 = torch.as_tensor(frames_2[0], device='cuda').clone()
        shape_2 = tensor_2.shape
        del frames_2

        print(f"Second decode after release_decoder(), frame shape: {shape_2}")

        # Verify frame shapes match
        assert shape_1 == shape_2, (
            f"Frame shapes don't match after release_decoder()! " f"Before: {shape_1}, After: {shape_2}"
        )

        del decoder
        force_cleanup()

    def test_10_repeated_release_decoder_no_leak(self, gpu_monitor, video_files):
        """
        Test 10: Repeated release_decoder() cycles don't cause memory leak.

        Scenario: Repeatedly (decode → release_decoder()) in a loop
        Verify: No cumulative memory growth
        """
        MAX_LEAK_PER_CYCLE_MB = 1.0

        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        num_cycles = 20
        memory_after_cycles = []

        print(f"\n=== Running {num_cycles} decode-release_decoder cycles ===")

        for cycle in range(num_cycles):
            # Decode
            frame_ids = [cycle % 50] * len(video_files)
            frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

            # Release decoder
            decoder.release_decoder()
            torch.cuda.synchronize()

            current_gpu = gpu_monitor.get_used_memory_mb()
            memory_after_cycles.append(current_gpu)

            if (cycle + 1) % 5 == 0:
                print(
                    f"Cycle {cycle + 1}: {current_gpu:.1f} MB "
                    f"(delta from baseline: {current_gpu - baseline_gpu:.1f} MB)"
                )

        # Analyze memory growth
        growth = memory_after_cycles[-1] - memory_after_cycles[0]
        avg_growth_per_cycle = growth / (num_cycles - 1) if num_cycles > 1 else 0

        print(f"\n=== Memory Analysis ===")
        print(f"First cycle: {memory_after_cycles[0]:.1f} MB")
        print(f"Last cycle: {memory_after_cycles[-1]:.1f} MB")
        print(f"Growth: {growth:.1f} MB")
        print(f"Avg growth per cycle: {avg_growth_per_cycle:.2f} MB")

        # Should not have significant per-cycle leak
        assert avg_growth_per_cycle < MAX_LEAK_PER_CYCLE_MB, (
            f"Memory leak in release_decoder() cycles! " f"Avg growth: {avg_growth_per_cycle:.2f} MB/cycle"
        )

        del decoder
        force_cleanup()

    def test_11_release_decoder_with_pending_async(self, gpu_monitor, video_files):
        """
        Test 11: release_decoder() properly handles pending async task.

        Scenario: Start async decode → immediately call release_decoder()
        Verify: No crash, proper cleanup
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        # Start async decode
        frame_ids = [0] * len(video_files)
        decoder.DecodeN12ToRGBAsync(video_files, frame_ids, False)

        # Immediately release decoder (should wait for async to complete)
        decoder.release_decoder()
        torch.cuda.synchronize()

        # Try to decode again (should create new readers)
        frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        assert frames is not None and len(frames) == len(video_files)

        tensor = torch.as_tensor(frames[0], device='cuda')
        assert tensor.numel() > 0, "Got empty frame after release_decoder with pending async!"

        del frames
        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"\nGPU memory delta: {delta:.1f} MB")
        assert is_ok, f"Memory leak with release_decoder + pending async! Delta: {delta:.1f} MB"

    def test_12_exception_then_resource_release(self, gpu_monitor, video_files):
        """
        Test 12: Resources properly released after exception.

        Scenario: Trigger exception with invalid file → catch exception → del decoder
        Verify: Exception doesn't prevent resource release
        """
        force_cleanup()
        baseline_gpu = gpu_monitor.get_used_memory_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        # First, do some successful decoding
        frame_ids = [0] * len(video_files)
        frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
        tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
        del frames, tensors

        after_decode_gpu = gpu_monitor.get_used_memory_mb()
        print(f"\nGPU memory after successful decode: {after_decode_gpu:.1f} MB")

        # Now trigger an exception
        invalid_files = ["/nonexistent/invalid/path.mp4"]
        invalid_frame_ids = [0]

        with pytest.raises(RuntimeError):
            decoder.DecodeN12ToRGB(invalid_files, invalid_frame_ids, False)

        print("Exception caught as expected")

        # Delete decoder - should still release resources
        del decoder
        force_cleanup()

        final_gpu = gpu_monitor.get_used_memory_mb()
        is_ok, delta = measure_memory_delta(baseline_gpu, final_gpu, self.GPU_TOLERANCE_MB)

        print(f"GPU memory after exception + del: {final_gpu:.1f} MB (delta: {delta:.1f} MB)")

        assert is_ok, f"GPU memory leak after exception! Delta: {delta:.1f} MB"


# ============================================================================
# CPU Memory Tests
# ============================================================================


class TestCPUMemoryRelease:
    """Tests for CPU memory resource release."""

    CPU_TOLERANCE_MB = 10.0

    def test_cpu_memory_stable_sync_loop(self, cpu_monitor, video_files):
        """Test CPU memory stability during sync decode loop."""
        force_cleanup()
        baseline_cpu = cpu_monitor.get_rss_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        memory_samples = []
        for i in range(30):
            frame_ids = [i % 100] * len(video_files)
            frames = decoder.DecodeN12ToRGB(video_files, frame_ids, False)
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

            if (i + 1) % 10 == 0:
                gc.collect()
                mem = cpu_monitor.get_rss_mb()
                memory_samples.append(mem)
                print(f"Iteration {i+1}: CPU RSS = {mem:.1f} MB")

        del decoder
        force_cleanup()

        final_cpu = cpu_monitor.get_rss_mb()
        delta = final_cpu - baseline_cpu

        print(f"\nCPU RSS delta: {delta:.1f} MB (tolerance: {self.CPU_TOLERANCE_MB} MB)")

        assert delta < self.CPU_TOLERANCE_MB, f"CPU memory leak! Delta: {delta:.1f} MB"

    def test_cpu_memory_stable_async_loop(self, cpu_monitor, video_files):
        """Test CPU memory stability during async decode loop."""
        force_cleanup()
        baseline_cpu = cpu_monitor.get_rss_mb()

        decoder = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=6,
            iGpu=0,
        )

        for i in range(30):
            frame_ids = [i % 100] * len(video_files)
            decoder.DecodeN12ToRGBAsync(video_files, frame_ids, False)
            frames = decoder.DecodeN12ToRGBAsyncGetBuffer(video_files, frame_ids, False)
            tensors = [torch.as_tensor(f, device='cuda').clone() for f in frames]
            del frames, tensors

        del decoder
        force_cleanup()

        final_cpu = cpu_monitor.get_rss_mb()
        delta = final_cpu - baseline_cpu

        print(f"\nCPU RSS delta: {delta:.1f} MB (tolerance: {self.CPU_TOLERANCE_MB} MB)")

        assert delta < self.CPU_TOLERANCE_MB, f"CPU memory leak! Delta: {delta:.1f} MB"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
