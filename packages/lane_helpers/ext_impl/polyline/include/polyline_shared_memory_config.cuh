/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LANE_HELPERS_POLYLINE_SHARED_MEMORY_CONFIG_CUH
#define LANE_HELPERS_POLYLINE_SHARED_MEMORY_CONFIG_CUH

#include <cstddef>
#include <mutex>
#include <stdexcept>

#include <cuda_runtime.h>

#include "helper_macros.cuh"
#include "polyline_kernels.cuh"

namespace polyline {

static constexpr int MAX_CACHED_CUDA_DEVICES = 64;

static void check_non_negative_cuda_device(int device) {
    if (device < 0) {
        throw std::runtime_error("CUDA device index must be non-negative.");
    }
}

static size_t query_polyline_max_shared_full_for_device(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    size_t max_shared_full = static_cast<size_t>(prop.sharedMemPerBlock);
    if (prop.sharedMemPerBlockOptin != 0) {
        max_shared_full = static_cast<size_t>(prop.sharedMemPerBlockOptin);
    }
    return max_shared_full;
}

static size_t polyline_max_shared_full_for_device(int device) {
    static std::once_flag configured_devices[MAX_CACHED_CUDA_DEVICES];
    static size_t max_shared_full_by_device[MAX_CACHED_CUDA_DEVICES] = {};

    check_non_negative_cuda_device(device);
    // Fallback if there are more devices than the maximum number of cached devices we use.
    if (device >= MAX_CACHED_CUDA_DEVICES) {
        const size_t max_shared_full = query_polyline_max_shared_full_for_device(device);
        return max_shared_full;
    }

    std::call_once(configured_devices[device], [device]() {
        max_shared_full_by_device[device] = query_polyline_max_shared_full_for_device(device);
    });
    const size_t max_shared_full = max_shared_full_by_device[device];
    return max_shared_full;
}

template <typename dtype, bool use_shared_distances>
static void configure_polyline_sampling_kernel(size_t max_shared_full) {
    CUDA_CHECK(cudaFuncSetAttribute(polyline_sampling_fully_shared_kernel<dtype, use_shared_distances>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(max_shared_full)));
    CUDA_CHECK(cudaFuncSetAttribute(polyline_sampling_fully_shared_kernel<dtype, use_shared_distances>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
}

template <typename dtype, bool use_shared_distances>
static void configure_polyline_sampling_kernel_once(int device, size_t max_shared_full) {
    static std::once_flag configured_devices[MAX_CACHED_CUDA_DEVICES];

    check_non_negative_cuda_device(device);
    if (device >= MAX_CACHED_CUDA_DEVICES) {
        configure_polyline_sampling_kernel<dtype, use_shared_distances>(max_shared_full);
        return;
    }

    std::call_once(configured_devices[device], [max_shared_full]() {
        configure_polyline_sampling_kernel<dtype, use_shared_distances>(max_shared_full);
    });
}

template <typename dtype, typename sample_size_dtype, bool use_shared_distances>
static void configure_polyline_sampling_var_batch_kernel(size_t max_shared_full) {
    CUDA_CHECK(cudaFuncSetAttribute(
        polyline_sampling_fully_shared_var_batch_kernel<dtype, sample_size_dtype, use_shared_distances>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_shared_full)));
    CUDA_CHECK(cudaFuncSetAttribute(
        polyline_sampling_fully_shared_var_batch_kernel<dtype, sample_size_dtype, use_shared_distances>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
}

template <typename dtype, typename sample_size_dtype, bool use_shared_distances>
static void configure_polyline_sampling_var_batch_kernel_once(int device, size_t max_shared_full) {
    static std::once_flag configured_devices[MAX_CACHED_CUDA_DEVICES];

    check_non_negative_cuda_device(device);
    if (device >= MAX_CACHED_CUDA_DEVICES) {
        configure_polyline_sampling_var_batch_kernel<dtype, sample_size_dtype, use_shared_distances>(
            max_shared_full);
        return;
    }

    std::call_once(configured_devices[device], [max_shared_full]() {
        configure_polyline_sampling_var_batch_kernel<dtype, sample_size_dtype, use_shared_distances>(
            max_shared_full);
    });
}

}  // namespace polyline

#endif  // LANE_HELPERS_POLYLINE_SHARED_MEMORY_CONFIG_CUH
