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

#include <iostream>
#include <limits>
#include <cstdint>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "helper_macros.cuh"

#include "polyline.cuh"
#include "polyline_kernels.cuh"
#include "polyline_shared_memory_config.cuh"

namespace polyline {

// Return the largest power of two that is <= n.
// For n <= 1, this returns 0 for n == 0 and 1 for n == 1.
static inline int last_power_of_2(int n) {
    if (n <= 0) {
        return 0;
    }
    unsigned int v = static_cast<unsigned int>(n);
    // Propagate highest set bit to all lower bits.
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    // Now (v + 1) >> 1 is the highest power of two <= original n.
    const int power_of_two = static_cast<int>((v + 1u) >> 1);
    return power_of_two;
}

template <typename dtype>
struct PolylineLengthLaunchConfig {
    dim3 block_dim;
    dim3 grid_dim;
    size_t shared_mem_size;
};

static int polyline_launch_threads_x(int num_samples_per_block) {
    const int max_num_threads = 1024;

    const int max_threads_x_for_y = max_num_threads / num_samples_per_block;
    // Round down to a multiple of 32, but keep at least one warp.
    int threads_x = (max_threads_x_for_y / 32) * 32;
    if (threads_x < 32) {
        threads_x = 32;
    }
    return threads_x;
}

template <typename dtype>
static PolylineLaunchConfig<dtype> make_polyline_launch_config_for_y(int num_points, int num_samples,
                                                                     int num_samples_per_block) {
    const int threads_x = polyline_launch_threads_x(num_samples_per_block);

    const dim3 block_dim(threads_x, num_samples_per_block, 1);
    const dim3 grid_dim(1, (num_samples + block_dim.y - 1) / block_dim.y, 1);
    const int num_points_full_blocks = ((num_points + block_dim.x - 1) / block_dim.x) * block_dim.x;
    const int num_warps_per_sample = (block_dim.x + 31) / 32;
    const size_t scratch_buffer_size_elems = block_dim.y * (num_warps_per_sample + 1);
    const size_t distances_buffer_size_elems_shared = static_cast<size_t>(block_dim.y) * num_points;

    PolylineLaunchConfig<dtype> cfg;
    cfg.block_dim = block_dim;
    cfg.grid_dim = grid_dim;
    cfg.num_points_full_blocks = num_points_full_blocks;
    cfg.shared_mem_size = (distances_buffer_size_elems_shared + scratch_buffer_size_elems) * sizeof(dtype);
    cfg.distance_buffer_ext_size_elems = 0;
    cfg.use_shared_distances = true;
    cfg.max_shared_full = 0;
    return cfg;
}

template <typename dtype>
static size_t polyline_external_distance_buffer_size_elems(const PolylineLaunchConfig<dtype>& cfg,
                                                           int num_points) {
    const size_t buffer_size_elems = static_cast<size_t>(cfg.grid_dim.y) * cfg.block_dim.y * num_points;
    return buffer_size_elems;
}

template <typename dtype>
static size_t polyline_scratch_shared_mem_size(const PolylineLaunchConfig<dtype>& cfg) {
    const int num_warps_per_sample = (cfg.block_dim.x + 31) / 32;
    const size_t shared_mem_size =
        static_cast<size_t>(cfg.block_dim.y) * (num_warps_per_sample + 1) * sizeof(dtype);
    return shared_mem_size;
}

template <typename dtype>
PolylineLaunchConfig<dtype> make_polyline_launch_config(int num_points, int num_samples, int device) {
    // Keep blockDim.y at 1 so blockDim.x can use the full thread block for each sample.
    PolylineLaunchConfig<dtype> cfg = make_polyline_launch_config_for_y<dtype>(num_points, num_samples, 1);

    // Determine whether we can stay in the shared‑memory path using the opt‑in
    // limit (`sharedMemPerBlockOptin`) instead of falling back to the external
    // buffer.
    const size_t max_shared_full = polyline_max_shared_full_for_device(device);

    if (cfg.shared_mem_size <= max_shared_full) {
        cfg.max_shared_full = max_shared_full;
        return cfg;
    }

    cfg.shared_mem_size = polyline_scratch_shared_mem_size(cfg);
    cfg.use_shared_distances = false;
    cfg.max_shared_full = max_shared_full;
    cfg.distance_buffer_ext_size_elems = polyline_external_distance_buffer_size_elems(cfg, num_points);
    return cfg;
}

template <typename dtype>
static PolylineLengthLaunchConfig<dtype> make_polyline_length_launch_config(int num_samples) {
    const int max_num_threads = 1024;
    const int max_y_by_threads = max_num_threads / 32;  // assuming at least one warp in x
    const int max_y_candidate = min(num_samples, max_y_by_threads);
    int num_samples_per_block = last_power_of_2(max_y_candidate);
    if (num_samples_per_block < 1) {
        num_samples_per_block = 1;
    }

    const int max_threads_x_for_y = max_num_threads / num_samples_per_block;
    int threads_x = (max_threads_x_for_y / 32) * 32;
    if (threads_x < 32) {
        threads_x = 32;
    }

    const dim3 block_dim(threads_x, num_samples_per_block, 1);
    const dim3 grid_dim(1, (num_samples + block_dim.y - 1) / block_dim.y, 1);
    const int num_warps_per_sample = (block_dim.x + 31) / 32;

    PolylineLengthLaunchConfig<dtype> cfg;
    cfg.block_dim = block_dim;
    cfg.grid_dim = grid_dim;
    cfg.shared_mem_size = static_cast<size_t>(block_dim.y) * num_warps_per_sample * sizeof(dtype);
    return cfg;
}

template <typename dtype>
void polyline_interpolation(dtype* points, int num_points, int num_dims, dtype* distances, int num_distances,
                            dtype* result_points, int num_samples, bool relative_distances, int device,
                            const PolylineLaunchConfig<dtype>& cfg, dtype* distance_buffer_ext,
                            cudaStream_t stream) {
    if (cfg.use_shared_distances) {
        configure_polyline_sampling_kernel_once<dtype, true>(device, cfg.max_shared_full);
        polyline_sampling_fully_shared_kernel<dtype, true>
            <<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
                points, distances, result_points, num_points, cfg.num_points_full_blocks, num_dims,
                num_distances, num_samples, relative_distances, nullptr);
    } else {
        polyline_sampling_fully_shared_kernel<dtype, false>
            <<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
                points, distances, result_points, num_points, cfg.num_points_full_blocks, num_dims,
                num_distances, num_samples, relative_distances, distance_buffer_ext);
    }
    CUDA_CHECK_LAST();
}

template <typename dtype>
void polyline_lengths(dtype* points, int num_points, int num_dims, dtype* lengths, int num_samples,
                      cudaStream_t stream) {
    auto cfg = make_polyline_length_launch_config<dtype>(num_samples);
    polyline_lengths_kernel<dtype><<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
        points, lengths, num_points, num_dims, num_samples);
    CUDA_CHECK_LAST();
}

template <typename dtype, typename sample_size_dtype>
void polyline_interpolation_var_size_batch(dtype* points, int max_num_points, int num_dims, dtype* distances,
                                           int num_distances, dtype* result_points, int num_samples,
                                           sample_size_dtype* sample_sizes_points,
                                           sample_size_dtype* sample_sizes_distances_to_sample,
                                           bool relative_distances, int device,
                                           const PolylineLaunchConfig<dtype>& cfg, dtype* distance_buffer_ext,
                                           cudaStream_t stream) {
    if (cfg.use_shared_distances) {
        configure_polyline_sampling_var_batch_kernel_once<dtype, sample_size_dtype, true>(
            device, cfg.max_shared_full);
        polyline_sampling_fully_shared_var_batch_kernel<dtype, sample_size_dtype, true>
            <<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
                points, distances, result_points, max_num_points, cfg.num_points_full_blocks, num_dims,
                num_distances, num_samples, sample_sizes_points, sample_sizes_distances_to_sample,
                relative_distances, nullptr);
    } else {
        polyline_sampling_fully_shared_var_batch_kernel<dtype, sample_size_dtype, false>
            <<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
                points, distances, result_points, max_num_points, cfg.num_points_full_blocks, num_dims,
                num_distances, num_samples, sample_sizes_points, sample_sizes_distances_to_sample,
                relative_distances, distance_buffer_ext);
    }
    CUDA_CHECK_LAST();
}

template <typename dtype, typename sample_size_dtype>
void polyline_lengths_var_size_batch(dtype* points, int max_num_points, int num_dims, dtype* lengths,
                                     int num_samples, sample_size_dtype* sample_sizes_points,
                                     cudaStream_t stream) {
    auto cfg = make_polyline_length_launch_config<dtype>(num_samples);
    polyline_lengths_var_batch_kernel<dtype, sample_size_dtype>
        <<<cfg.grid_dim, cfg.block_dim, cfg.shared_mem_size, stream>>>(
            points, lengths, max_num_points, num_dims, num_samples, sample_sizes_points);
    CUDA_CHECK_LAST();
}

#define INSTANTIATE_POLYLINE_INTERPOLATION(DTYPE)                                          \
    template void polyline_interpolation<DTYPE>(                                           \
        DTYPE * points, int num_points, int num_dims, DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, bool relative_distances, int device,        \
        const PolylineLaunchConfig<DTYPE>& cfg, DTYPE* distance_buffer_ext, cudaStream_t stream);

#define INSTANTIATE_POLYLINE_LAUNCH_CONFIG(DTYPE)                                                            \
    template PolylineLaunchConfig<DTYPE> make_polyline_launch_config<DTYPE>(int num_points, int num_samples, \
                                                                            int device);

#define INSTANTIATE_POLYLINE_LENGTHS(DTYPE)                                                             \
    template void polyline_lengths<DTYPE>(DTYPE * points, int num_points, int num_dims, DTYPE* lengths, \
                                          int num_samples, cudaStream_t stream);

#define INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH(DTYPE, SAMPLE_SIZE_DTYPE)               \
    template void polyline_interpolation_var_size_batch<DTYPE, SAMPLE_SIZE_DTYPE>(                \
        DTYPE * points, int max_num_points, int num_dims, DTYPE* distances, int num_distances,    \
        DTYPE* result_points, int num_samples, SAMPLE_SIZE_DTYPE* sample_sizes_points,            \
        SAMPLE_SIZE_DTYPE* sample_sizes_distances_to_sample, bool relative_distances, int device, \
        const PolylineLaunchConfig<DTYPE>& cfg, DTYPE* distance_buffer_ext, cudaStream_t stream);

#define INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH(DTYPE, SAMPLE_SIZE_DTYPE)              \
    template void polyline_lengths_var_size_batch<DTYPE, SAMPLE_SIZE_DTYPE>(               \
        DTYPE * points, int max_num_points, int num_dims, DTYPE* lengths, int num_samples, \
        SAMPLE_SIZE_DTYPE* sample_sizes_points, cudaStream_t stream);

#define INSTANTIATE_POLYLINE_CUDA_DTYPE(DTYPE)                        \
    INSTANTIATE_POLYLINE_LAUNCH_CONFIG(DTYPE)                         \
    INSTANTIATE_POLYLINE_INTERPOLATION(DTYPE)                         \
    INSTANTIATE_POLYLINE_LENGTHS(DTYPE)                               \
    INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH(DTYPE, int)     \
    INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH(DTYPE, int64_t) \
    INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH(DTYPE, int)           \
    INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH(DTYPE, int64_t)

INSTANTIATE_POLYLINE_CUDA_DTYPE(float)
INSTANTIATE_POLYLINE_CUDA_DTYPE(double)
INSTANTIATE_POLYLINE_CUDA_DTYPE(c10::Half)
INSTANTIATE_POLYLINE_CUDA_DTYPE(c10::BFloat16)

#undef INSTANTIATE_POLYLINE_CUDA_DTYPE
#undef INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH
#undef INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH
#undef INSTANTIATE_POLYLINE_LENGTHS
#undef INSTANTIATE_POLYLINE_LAUNCH_CONFIG
#undef INSTANTIATE_POLYLINE_INTERPOLATION
}  // namespace polyline