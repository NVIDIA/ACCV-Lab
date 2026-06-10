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

#include <cstdint>
#include <vector>

#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>

#include "polyline_common.cuh"

namespace polyline {

template <typename dtype>
using cpu_acc_t = at::acc_type<dtype, false>;

template <typename dtype>
static void compute_accumulated_distances_cpu(const dtype* points_sample, int num_points, int num_dims,
                                              cpu_acc_t<dtype>* accum_distances) {
    using acc_t = cpu_acc_t<dtype>;
    accum_distances[0] = static_cast<acc_t>(0.0);
    for (int point_idx = 0; point_idx < num_points - 1; ++point_idx) {
        accum_distances[point_idx + 1] =
            accum_distances[point_idx] +
            compute_segment_length_common<dtype, acc_t>(points_sample, point_idx, num_dims);
    }
}

template <typename dtype>
static void sample_polyline_cpu(const dtype* points_sample, const dtype* distances_sample, int num_points,
                                int num_dims, int num_distances, dtype* result_sample,
                                bool relative_distances, std::vector<cpu_acc_t<dtype>>& accum_distances) {
    using acc_t = cpu_acc_t<dtype>;
    if (num_distances == 0) {
        return;
    }
    if (num_points == 0) {
        dtype* result_sample_i = result_sample;
        for (int distance_idx = 0; distance_idx < num_distances;
             ++distance_idx, result_sample_i += num_dims) {
            fill_point_with_nan_common<dtype>(result_sample_i, num_dims);
        }
        return;
    }
    compute_accumulated_distances_cpu<dtype>(points_sample, num_points, num_dims, accum_distances.data());
    const acc_t total_length = accum_distances[num_points - 1];
    for (int distance_idx = 0; distance_idx < num_distances; ++distance_idx) {
        const acc_t distance_to_sample =
            relative_distances ? static_cast<acc_t>(distances_sample[distance_idx]) * total_length
                               : static_cast<acc_t>(distances_sample[distance_idx]);
        sample_at_distance_common<dtype, acc_t>(points_sample, accum_distances.data(), distance_to_sample,
                                                num_points, num_dims,
                                                result_sample + distance_idx * num_dims);
    }
}

template <typename dtype>
void polyline_interpolation_cpu(const dtype* points, int num_points, int num_dims, const dtype* distances,
                                int num_distances, dtype* result_points, int num_samples,
                                bool relative_distances) {
    using acc_t = cpu_acc_t<dtype>;
    const size_t stride_points = static_cast<size_t>(num_points) * static_cast<size_t>(num_dims);
    const size_t stride_distances = static_cast<size_t>(num_distances);
    const size_t stride_result = static_cast<size_t>(num_distances) * static_cast<size_t>(num_dims);
    at::parallel_for(0, num_samples, 0, [&](int64_t start, int64_t end) {
        std::vector<acc_t> accum_distances(num_points);
        for (int64_t sample_idx = start; sample_idx < end; ++sample_idx) {
            const dtype* points_sample = points + sample_idx * stride_points;
            const dtype* distances_sample = distances + sample_idx * stride_distances;
            dtype* result_sample = result_points + sample_idx * stride_result;
            sample_polyline_cpu<dtype>(points_sample, distances_sample, num_points, num_dims, num_distances,
                                       result_sample, relative_distances, accum_distances);
        }
    });
}

template <typename dtype>
void polyline_lengths_cpu(const dtype* points, int num_points, int num_dims, dtype* lengths,
                          int num_samples) {
    using acc_t = cpu_acc_t<dtype>;
    const size_t stride_points = static_cast<size_t>(num_points) * static_cast<size_t>(num_dims);
    at::parallel_for(0, num_samples, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_idx = start; sample_idx < end; ++sample_idx) {
            const dtype* points_sample = points + sample_idx * stride_points;
            acc_t length = static_cast<acc_t>(0.0);
            if (num_points == 0) {
                length = polyline_nan<acc_t>();
            } else {
                for (int point_idx = 0; point_idx < num_points - 1; ++point_idx) {
                    length += compute_segment_length_common<dtype, acc_t>(points_sample, point_idx, num_dims);
                }
            }
            lengths[sample_idx] = static_cast<dtype>(length);
        }
    });
}

template <typename dtype, typename sample_size_dtype>
void polyline_interpolation_var_size_batch_cpu(const dtype* points, int max_num_points, int num_dims,
                                               const dtype* distances, int num_distances,
                                               dtype* result_points, int num_samples,
                                               const sample_size_dtype* sample_sizes_points,
                                               const sample_size_dtype* sample_sizes_distances_to_sample,
                                               bool relative_distances) {
    using acc_t = cpu_acc_t<dtype>;
    at::parallel_for(0, num_samples, 0, [&](int64_t start, int64_t end) {
        std::vector<acc_t> accum_distances(max_num_points);
        for (int64_t sample_idx = start; sample_idx < end; ++sample_idx) {
            const int curr_num_points = static_cast<int>(sample_sizes_points[sample_idx]);
            const int curr_num_distances = static_cast<int>(sample_sizes_distances_to_sample[sample_idx]);
            const dtype* points_sample = points + sample_idx * max_num_points * num_dims;
            const dtype* distances_sample = distances + sample_idx * num_distances;
            dtype* result_sample = result_points + sample_idx * num_distances * num_dims;
            sample_polyline_cpu<dtype>(points_sample, distances_sample, curr_num_points, num_dims,
                                       curr_num_distances, result_sample, relative_distances,
                                       accum_distances);
        }
    });
}

template <typename dtype, typename sample_size_dtype>
void polyline_lengths_var_size_batch_cpu(const dtype* points, int max_num_points, int num_dims,
                                         dtype* lengths, int num_samples,
                                         const sample_size_dtype* sample_sizes_points) {
    using acc_t = cpu_acc_t<dtype>;
    at::parallel_for(0, num_samples, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_idx = start; sample_idx < end; ++sample_idx) {
            const int curr_num_points = static_cast<int>(sample_sizes_points[sample_idx]);
            const dtype* points_sample = points + sample_idx * max_num_points * num_dims;
            acc_t length = static_cast<acc_t>(0.0);
            if (curr_num_points == 0) {
                length = polyline_nan<acc_t>();
            } else {
                for (int point_idx = 0; point_idx < curr_num_points - 1; ++point_idx) {
                    length += compute_segment_length_common<dtype, acc_t>(points_sample, point_idx, num_dims);
                }
            }
            lengths[sample_idx] = static_cast<dtype>(length);
        }
    });
}

#define INSTANTIATE_POLYLINE_INTERPOLATION_CPU(DTYPE)                                                 \
    template void polyline_interpolation_cpu<DTYPE>(                                                  \
        const DTYPE* points, int num_points, int num_dims, const DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, bool relative_distances);

#define INSTANTIATE_POLYLINE_LENGTHS_CPU(DTYPE)                                                  \
    template void polyline_lengths_cpu<DTYPE>(const DTYPE* points, int num_points, int num_dims, \
                                              DTYPE* lengths, int num_samples);

#define INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU(DTYPE, SAMPLE_SIZE_DTYPE)                   \
    template void polyline_interpolation_var_size_batch_cpu<DTYPE, SAMPLE_SIZE_DTYPE>(                    \
        const DTYPE* points, int max_num_points, int num_dims, const DTYPE* distances, int num_distances, \
        DTYPE* result_points, int num_samples, const SAMPLE_SIZE_DTYPE* sample_sizes_points,              \
        const SAMPLE_SIZE_DTYPE* sample_sizes_distances_to_sample, bool relative_distances);

#define INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU(DTYPE, SAMPLE_SIZE_DTYPE)               \
    template void polyline_lengths_var_size_batch_cpu<DTYPE, SAMPLE_SIZE_DTYPE>(                \
        const DTYPE* points, int max_num_points, int num_dims, DTYPE* lengths, int num_samples, \
        const SAMPLE_SIZE_DTYPE* sample_sizes_points);

#define INSTANTIATE_POLYLINE_CPU_DTYPE(DTYPE)                             \
    INSTANTIATE_POLYLINE_INTERPOLATION_CPU(DTYPE)                         \
    INSTANTIATE_POLYLINE_LENGTHS_CPU(DTYPE)                               \
    INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU(DTYPE, int)     \
    INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU(DTYPE, int64_t) \
    INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU(DTYPE, int)           \
    INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU(DTYPE, int64_t)

INSTANTIATE_POLYLINE_CPU_DTYPE(float)
INSTANTIATE_POLYLINE_CPU_DTYPE(double)

#undef INSTANTIATE_POLYLINE_CPU_DTYPE
#undef INSTANTIATE_POLYLINE_LENGTHS_VAR_SIZE_BATCH_CPU
#undef INSTANTIATE_POLYLINE_INTERPOLATION_VAR_SIZE_BATCH_CPU
#undef INSTANTIATE_POLYLINE_LENGTHS_CPU
#undef INSTANTIATE_POLYLINE_INTERPOLATION_CPU

}  // namespace polyline
