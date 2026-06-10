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

#ifndef LANE_HELPERS_POLYLINE_COMMON_CUH
#define LANE_HELPERS_POLYLINE_COMMON_CUH

#include <cmath>
#include <limits>

#include "polyline_dtype_compat.cuh"

#ifdef __CUDACC__
// Keep scalar helpers callable from both CUDA kernels and CPU translation units.
#define POLYLINE_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define POLYLINE_HOST_DEVICE_INLINE inline
#endif

namespace polyline {

template <typename dtype>
POLYLINE_HOST_DEVICE_INLINE dtype polyline_nan() {
    const dtype nan_value = static_cast<dtype>(NAN);
    return nan_value;
}

template <typename point_dtype>
POLYLINE_HOST_DEVICE_INLINE void fill_point_with_nan_common(point_dtype* res_point, int num_dims) {
    const point_dtype nan_value = polyline_nan<point_dtype>();
    for (int d = 0; d < num_dims; ++d) {
        res_point[d] = nan_value;
    }
}

/**
 * @brief Compute the Euclidean length of one polyline segment.
 *
 * @details
 * `segment_idx` refers to the segment between points `segment_idx` and
 * `segment_idx + 1`. The point coordinates are laid out consecutively as
 * `(num_points, num_dims)`.
 *
 * The point dtype and accumulation dtype are intentionally separate so the CPU
 * path can accumulate in a wider type while the CUDA path preserves its
 * existing dtype behavior.
 */
template <typename point_dtype, typename accum_dtype>
POLYLINE_HOST_DEVICE_INLINE accum_dtype compute_segment_length_common(const point_dtype* points_sample,
                                                                      int segment_idx, int num_dims) {
    const point_dtype* first_point = points_sample + segment_idx * num_dims;
    const point_dtype* second_point = points_sample + (segment_idx + 1) * num_dims;
    accum_dtype accum_sqr = static_cast<accum_dtype>(0.0);
    for (int d = 0; d < num_dims; ++d) {
        const accum_dtype diff =
            static_cast<accum_dtype>(first_point[d]) - static_cast<accum_dtype>(second_point[d]);
        accum_sqr += diff * diff;
    }
    const accum_dtype segment_length = polyline_sqrt(accum_sqr);
    return segment_length;
}

/**
 * @brief Find the last index whose value is lower than or equal to `value`.
 *
 * @details
 * The input sequence is expected to be monotonically non-decreasing cumulative
 * distances. The return value can be:
 * - `-1` when `value` lies before the first point.
 * - `sequence_length - 1` when `value` lies at or beyond the last point.
 * - Any valid lower segment endpoint otherwise.
 *
 * This is used to locate the segment containing the requested interpolation
 * distance.
 */
template <typename accum_dtype>
POLYLINE_HOST_DEVICE_INLINE int get_index_of_last_lower_or_equal_to_common(const accum_dtype* sequence,
                                                                           accum_dtype value,
                                                                           int sequence_length) {
    int min_idx = 0;
    int max_idx = sequence_length - 1;

    if (polyline_value_gt(sequence[0], value)) {
        return -1;
    }
    if (polyline_value_lt(sequence[sequence_length - 1], value)) {
        return sequence_length - 1;
    }

    while (max_idx - min_idx > 1) {
        const int curr_idx = (max_idx + min_idx) >> 1;
        const accum_dtype curr_val = sequence[curr_idx];
        if (polyline_value_lt(curr_val, value)) {
            min_idx = curr_idx;
        } else if (polyline_value_gt(curr_val, value)) {
            max_idx = curr_idx;
        } else {
            min_idx = curr_idx;
            max_idx = curr_idx;
        }
    }
    return min_idx;
}

/**
 * @brief Sample one point on a polyline at a requested absolute distance.
 *
 * @details
 * `accum_distances` stores the distance from the start of the polyline to each
 * point. Distances outside the polyline are clamped to the first or last point.
 * Degenerate zero-length segments return the lower endpoint.
 */
template <typename point_dtype, typename accum_dtype>
POLYLINE_HOST_DEVICE_INLINE void sample_at_distance_common(const point_dtype* points,
                                                           const accum_dtype* accum_distances,
                                                           accum_dtype distance_to_sample_at, int num_points,
                                                           int num_dims, point_dtype* res_point) {
    const int index_min = get_index_of_last_lower_or_equal_to_common<accum_dtype>(
        accum_distances, distance_to_sample_at, num_points);
    if (index_min >= 0 && index_min < num_points - 1) {
        const int index_max = index_min + 1;
        const point_dtype* min_point = points + index_min * num_dims;
        const point_dtype* max_point = points + index_max * num_dims;
        const accum_dtype dist_min = accum_distances[index_min];
        const accum_dtype dist_max = accum_distances[index_max];
        const accum_dtype dist = dist_max - dist_min;
        if (polyline_value_ge(dist, static_cast<accum_dtype>(std::numeric_limits<accum_dtype>::epsilon()))) {
            const accum_dtype weight_max = (distance_to_sample_at - dist_min) / dist;
            const accum_dtype weight_min = (dist_max - distance_to_sample_at) / dist;
            for (int d = 0; d < num_dims; ++d) {
                const accum_dtype interpolated = static_cast<accum_dtype>(min_point[d]) * weight_min +
                                                 static_cast<accum_dtype>(max_point[d]) * weight_max;
                res_point[d] = static_cast<point_dtype>(interpolated);
            }
        } else {
            for (int d = 0; d < num_dims; ++d) {
                res_point[d] = min_point[d];
            }
        }
    } else if (index_min == -1) {
        for (int d = 0; d < num_dims; ++d) {
            // Note that we are accessing the first point, so that points[d] corresponds to the element we
            // want to access, and no offset is needed.
            res_point[d] = points[d];
        }
    } else if (index_min == num_points - 1) {
        for (int d = 0; d < num_dims; ++d) {
            res_point[d] = points[(num_points - 1) * num_dims + d];
        }
    }
}

}  // namespace polyline

#undef POLYLINE_HOST_DEVICE_INLINE

#endif  // LANE_HELPERS_POLYLINE_COMMON_CUH
