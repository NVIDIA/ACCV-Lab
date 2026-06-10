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

// Host-visible interface for the polyline interpolation CUDA
// implementation. This header is intentionally free of CUDA device intrinsics
// so it can be included from both C++ and CUDA translation units.

#ifndef LANE_HELPERS_POLYLINE_CUH
#define LANE_HELPERS_POLYLINE_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace polyline {

template <typename dtype>
struct PolylineLaunchConfig {
    dim3 block_dim;
    dim3 grid_dim;
    int num_points_full_blocks;
    size_t shared_mem_size;
    size_t distance_buffer_ext_size_elems;
    bool use_shared_distances;
    size_t max_shared_full;
};

template <typename dtype>
PolylineLaunchConfig<dtype> make_polyline_launch_config(int num_points, int num_samples, int device);

template <typename dtype>
void polyline_interpolation(dtype* points, int num_points, int num_dims, dtype* distances, int num_distances,
                            dtype* result_points, int num_samples, bool relative_distances, int device,
                            const PolylineLaunchConfig<dtype>& cfg, dtype* distance_buffer_ext,
                            cudaStream_t stream);

template <typename dtype>
void polyline_lengths(dtype* points, int num_points, int num_dims, dtype* lengths, int num_samples,
                      cudaStream_t stream);

template <typename dtype, typename sample_size_dtype>
void polyline_interpolation_var_size_batch(dtype* points, int max_num_points, int num_dims, dtype* distances,
                                           int num_distances, dtype* result_points, int num_samples,
                                           sample_size_dtype* sample_sizes_points,
                                           sample_size_dtype* sample_sizes_distances_to_sample,
                                           bool relative_distances, int device,
                                           const PolylineLaunchConfig<dtype>& cfg, dtype* distance_buffer_ext,
                                           cudaStream_t stream);

template <typename dtype, typename sample_size_dtype>
void polyline_lengths_var_size_batch(dtype* points, int max_num_points, int num_dims, dtype* lengths,
                                     int num_samples, sample_size_dtype* sample_sizes_points,
                                     cudaStream_t stream);

template <typename dtype>
void polyline_interpolation_cpu(const dtype* points, int num_points, int num_dims, const dtype* distances,
                                int num_distances, dtype* result_points, int num_samples,
                                bool relative_distances);

template <typename dtype>
void polyline_lengths_cpu(const dtype* points, int num_points, int num_dims, dtype* lengths, int num_samples);

template <typename dtype, typename sample_size_dtype>
void polyline_interpolation_var_size_batch_cpu(const dtype* points, int max_num_points, int num_dims,
                                               const dtype* distances, int num_distances,
                                               dtype* result_points, int num_samples,
                                               const sample_size_dtype* sample_sizes_points,
                                               const sample_size_dtype* sample_sizes_distances_to_sample,
                                               bool relative_distances);

template <typename dtype, typename sample_size_dtype>
void polyline_lengths_var_size_batch_cpu(const dtype* points, int max_num_points, int num_dims,
                                         dtype* lengths, int num_samples,
                                         const sample_size_dtype* sample_sizes_points);

// Explicit instantiations are provided in polyline.cu and polyline_cpu.cpp.
#define DECLARE_POLYLINE_LAUNCH_CONFIG_EXTERN(DTYPE)                                \
    extern template PolylineLaunchConfig<DTYPE> make_polyline_launch_config<DTYPE>( \
        int num_points, int num_samples, int device);

#define DECLARE_POLYLINE_INTERPOLATION_EXTERN(DTYPE)                                       \
    extern template void polyline_interpolation<DTYPE>(                                    \
        DTYPE * points, int num_points, int num_dims, DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, bool relative_distances, int device,        \
        const PolylineLaunchConfig<DTYPE>& cfg, DTYPE* distance_buffer_ext, cudaStream_t stream);

#define DECLARE_POLYLINE_LENGTHS_EXTERN(DTYPE)                                                 \
    extern template void polyline_lengths<DTYPE>(DTYPE * points, int num_points, int num_dims, \
                                                 DTYPE* lengths, int num_samples, cudaStream_t stream);

#define DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_EXTERN(DTYPE, SAMPLE_SIZE_DTYPE)            \
    extern template void polyline_interpolation_var_size_batch<DTYPE, SAMPLE_SIZE_DTYPE>(         \
        DTYPE * points, int max_num_points, int num_dims, DTYPE* distances, int num_distances,    \
        DTYPE* result_points, int num_samples, SAMPLE_SIZE_DTYPE* sample_sizes_points,            \
        SAMPLE_SIZE_DTYPE* sample_sizes_distances_to_sample, bool relative_distances, int device, \
        const PolylineLaunchConfig<DTYPE>& cfg, DTYPE* distance_buffer_ext, cudaStream_t stream);

#define DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_EXTERN(DTYPE, SAMPLE_SIZE_DTYPE)           \
    extern template void polyline_lengths_var_size_batch<DTYPE, SAMPLE_SIZE_DTYPE>(        \
        DTYPE * points, int max_num_points, int num_dims, DTYPE* lengths, int num_samples, \
        SAMPLE_SIZE_DTYPE* sample_sizes_points, cudaStream_t stream);

#define DECLARE_POLYLINE_CUDA_DTYPE_EXTERN(DTYPE)                        \
    DECLARE_POLYLINE_LAUNCH_CONFIG_EXTERN(DTYPE)                         \
    DECLARE_POLYLINE_INTERPOLATION_EXTERN(DTYPE)                         \
    DECLARE_POLYLINE_LENGTHS_EXTERN(DTYPE)                               \
    DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_EXTERN(DTYPE, int)     \
    DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_EXTERN(DTYPE, int64_t) \
    DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_EXTERN(DTYPE, int)           \
    DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_EXTERN(DTYPE, int64_t)

DECLARE_POLYLINE_CUDA_DTYPE_EXTERN(float)
DECLARE_POLYLINE_CUDA_DTYPE_EXTERN(double)
DECLARE_POLYLINE_CUDA_DTYPE_EXTERN(c10::Half)
DECLARE_POLYLINE_CUDA_DTYPE_EXTERN(c10::BFloat16)

#undef DECLARE_POLYLINE_CUDA_DTYPE_EXTERN
#undef DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_EXTERN
#undef DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_EXTERN
#undef DECLARE_POLYLINE_LENGTHS_EXTERN
#undef DECLARE_POLYLINE_INTERPOLATION_EXTERN
#undef DECLARE_POLYLINE_LAUNCH_CONFIG_EXTERN

#define DECLARE_POLYLINE_INTERPOLATION_CPU_EXTERN(DTYPE)                                              \
    extern template void polyline_interpolation_cpu<DTYPE>(                                           \
        const DTYPE* points, int num_points, int num_dims, const DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, bool relative_distances);

#define DECLARE_POLYLINE_LENGTHS_CPU_EXTERN(DTYPE)                                                      \
    extern template void polyline_lengths_cpu<DTYPE>(const DTYPE* points, int num_points, int num_dims, \
                                                     DTYPE* lengths, int num_samples);

#define DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, SAMPLE_SIZE_DTYPE)                \
    extern template void polyline_interpolation_var_size_batch_cpu<DTYPE, SAMPLE_SIZE_DTYPE>(             \
        const DTYPE* points, int max_num_points, int num_dims, const DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, const SAMPLE_SIZE_DTYPE* sample_sizes_points,              \
        const SAMPLE_SIZE_DTYPE* sample_sizes_distances_to_sample, bool relative_distances);

#define DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, SAMPLE_SIZE_DTYPE)            \
    extern template void polyline_lengths_var_size_batch_cpu<DTYPE, SAMPLE_SIZE_DTYPE>(         \
        const DTYPE* points, int max_num_points, int num_dims, DTYPE* lengths, int num_samples, \
        const SAMPLE_SIZE_DTYPE* sample_sizes_points);

#define DECLARE_POLYLINE_CPU_DTYPE_EXTERN(DTYPE)                             \
    DECLARE_POLYLINE_INTERPOLATION_CPU_EXTERN(DTYPE)                         \
    DECLARE_POLYLINE_LENGTHS_CPU_EXTERN(DTYPE)                               \
    DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, int)     \
    DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, int64_t) \
    DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, int)           \
    DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU_EXTERN(DTYPE, int64_t)

DECLARE_POLYLINE_CPU_DTYPE_EXTERN(float)
DECLARE_POLYLINE_CPU_DTYPE_EXTERN(double)

#undef DECLARE_POLYLINE_CPU_DTYPE_EXTERN
#undef DECLARE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU_EXTERN
#undef DECLARE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU_EXTERN
#undef DECLARE_POLYLINE_LENGTHS_CPU_EXTERN
#undef DECLARE_POLYLINE_INTERPOLATION_CPU_EXTERN
}  // namespace polyline

#endif  // LANE_HELPERS_POLYLINE_CUH
