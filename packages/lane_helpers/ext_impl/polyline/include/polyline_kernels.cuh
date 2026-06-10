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

#ifndef LANE_HELPERS_POLYLINE_KERNELS_CUH
#define LANE_HELPERS_POLYLINE_KERNELS_CUH

#include "polyline_common.cuh"
#include "polyline_dtype_compat.cuh"

namespace polyline {

template <typename dtype>
__device__ __forceinline__ void prefix_sum_warp(int index, dtype value, int num_values_in_scan,
                                                dtype& scan_value, dtype& sum_all) {
    sum_all = value;
    scan_value = static_cast<dtype>(0.0);
    for (int i = 1; i < num_values_in_scan; i <<= 1) {
        dtype sum_other = shfl_xor_sync_compat<dtype>(0xffffffffu, sum_all, i);
        scan_value += ((static_cast<uint32_t>(index) & static_cast<uint32_t>(i)) > 0) * sum_other;
        sum_all += sum_other;
    }
}

/**
 * @brief Perform a prefix sum on a block of values.
 *
 * @details
 * The buffer `warp_scan_buffer` is used to store the sums of the individual warps, which is then used
 * to compute the offsets to add to each warp. For that, a prefix sum is performed on the buffer in a second
 * step (in-place). The size of the buffer is (in elements): `blockDim.y * num_warps_per_sample`.
 * 
 * @tparam dtype The type of the values to prefix sum
 *
 * @param value The value to prefix sum
 * @param num_warps_per_sample The number of warps per sample
 * @param warp_scan_buffer The buffer to store the partial sums of the iterations so far for each sample
 *
 * @return The prefix sum for the current thread
 */
template <typename dtype>
__device__ __forceinline__ dtype prefix_sum_block(dtype value, int num_warps_per_sample,
                                                  dtype* warp_scan_buffer) {
    // ix corresponds to the thread index inside a single sample
    const int ix = threadIdx.x;
    const int iwx = threadIdx.x / 32;                         // index of the warp inside the sample
    const int iw = threadIdx.y * num_warps_per_sample + iwx;  // index of the warp in the block
    // Get thread id (consecutive IDs correspond to consecutive values in the array)
    //const int tid_shared = threadIdx.y * bxsize + ix;

    // Using warp shuffles iteratively, in two stages

    // First stage: perform warp scans
    dtype warp_scan1;
    dtype warp_sum1;
    prefix_sum_warp(ix, value, 32, warp_scan1, warp_sum1);
    // If this is the first thread in the warp, it is responsible for storing the partial sum for the warp
    if (ix % 32 == 0) warp_scan_buffer[iw] = warp_sum1;
    __syncthreads();

    // Warp scan for for the partial sums to obtain the offsets for each warp
    // The first warp (i.e. ix < 32) participates in step 2 of the warp scan.
    // Note that the whole first warp always participates in step 2, even if num_warps_per_sample < 32.
    // This is to avoid a deadlock without using a complex mask generation method for `__shfl_xor_sync()`.
    if (ix < 32) {
        const int wid_shared = threadIdx.y * num_warps_per_sample + ix;
        const bool is_inside = ix < num_warps_per_sample;
        const dtype value = is_inside ? warp_scan_buffer[wid_shared] : static_cast<dtype>(0.0);
        dtype warp_scan2;
        dtype warp_sum2;
        prefix_sum_warp(ix, value, num_warps_per_sample, warp_scan2, warp_sum2);
        if (ix < num_warps_per_sample) {
            warp_scan_buffer[wid_shared] = warp_scan2;
        }
    }
    __syncthreads();

    // Apply offsets to the partial sums to obtain the final values
    warp_scan1 += warp_scan_buffer[iw];

    return warp_scan1;
}

/**
 * @brief Perform a prefix sum on a block of values.
 *
 * @details
 * The buffer is split into 2 parts:
 * - The first part (`blockDim.y` elements) stores the running sums of all
 *   elements processed so far for each sample in y (`sum_buffer`).
 * - The second part (`warp_scan_buffer`) is scratch space for the intra‑block
 *   scan performed by `prefix_sum_block`. The size is: `blockDim.y * num_warps_per_sample`
 *   (see `prefix_sum_block` for more details)
 * Hence, the total buffer size in elements is (in elements):
 * `blockDim.y + blockDim.y * num_warps_per_sample`.
 * or
 * `blockDim.y * (1 + num_warps_per_sample)`
 * 
 * The results are stored in the sequence array, overriding the input values.
 * Note that the results are the accumulated values including the current value, i.e. the operation can be expressed as:
 * `sequences[i] = sum(sequences[0:(i+1)])`, where the slicing is defined as in Python.
 *
 * 
 *
 * @tparam dtype The type of the values to prefix sum
 *
 * @param sequences The sequences to prefix sum for the current thread block. Note that this means that 
 *   the first sequence is the one corresponding to threadIdx.y == 0 of the current block, not necessarily the 
 *   first sequence in the global array.
 * @param buffer Combined temporary storage used by the prefix-sum.
 * @param numel_x The number of elements in the x dimension
 * @param numel_x_full_blocks The number of elements in the x dimension extended to a multiple of blockDim.x
 * @param numel_y The number of sequences in the y dimension
 * @param offset The initial offset to add to the prefix sum of each sequence
 */
template <typename dtype>
__device__ __forceinline__ void prefix_sum_looped(dtype* sequence, dtype* buffer_block, int numel_x,
                                                  int numel_x_full_blocks, int numel_y, dtype offset) {
    const int ix = threadIdx.x;
    const int iy = threadIdx.y;

    // Buffer for keeping the sums of the iterations so far for each sample
    dtype* sum_buffer = buffer_block;
    // Buffer as needed for the prefix sum implementation
    dtype* warp_scan_buffer = buffer_block + blockDim.y;

    int num_warps_per_sample = (blockDim.x + 31) / 32;

    // Initialize the buffer containing the partial sums of the iterations so far for each sample
    if (ix == 0) {
        sum_buffer[iy] = static_cast<dtype>(0.0);
    }
    // Compute the sum one `bxsize` at a time for each sample
    for (int i = ix; i < numel_x_full_blocks; i += blockDim.x) {
        // Make sure that sum_buffer is written to (either initially or in the previous iteration)
        __syncthreads();
        const dtype value = i < numel_x ? sequence[i] : static_cast<dtype>(0.0);
        const dtype value_out =
            prefix_sum_block<dtype>(value, num_warps_per_sample, warp_scan_buffer) + sum_buffer[iy] + offset;
        const dtype value_out_incl_current = value_out + value;
        // Make sure that
        //   - sequences are not written to before they are read from for the current iteration
        //   - sum_buffer is not written to before it is read from for the current iteration
        __syncthreads();
        if (i < numel_x) {
            sequence[i] = value_out_incl_current;
        }
        // Update the sum buffer for the next iteration to the current value of the last processed sample.
        // Note that the last thread may be out of bounds and not correspond to the last element. However,
        // in this case, the value us not needed (and also still is the correct value as the values are
        // extended with zeros, so that the cumulative sum (computed as including the current value) is
        // the same as for the last element)
        if (ix == blockDim.x - 1) {
            sum_buffer[iy] = value_out_incl_current;
        }
        // Offset is only applied in the first iteration. Afterwards, the offset is already included in the
        // partial sum as stored in `sum_buffer` and the offset must not be applied again.
        offset = static_cast<dtype>(0.0);
    }
    __syncthreads();
}

template <typename dtype>
__device__ __forceinline__ dtype warp_reduce_sum(dtype value, int num_vals_per_partial) {
    const int ix = threadIdx.x;
    for (int i = 1; i < num_vals_per_partial; i <<= 1) {
        const dtype val_other = shfl_xor_sync_compat<dtype>(0xffffffffu, value, i);
        value += val_other;
    }
    return value;
}

template <typename dtype>
__device__ __forceinline__ dtype sample_reduce_sum(dtype value, int num_warps_per_sample,
                                                   dtype* warp_temp_and_result_buffer) {
    const int ix = threadIdx.x;                               // index of thread in the block
    const int iwx = threadIdx.x / 32;                         //index of warp in the sample
    const int iw = threadIdx.y * num_warps_per_sample + iwx;  // index of the warp in the block

    const dtype warp_sum = warp_reduce_sum(value, 32);
    // The first thread in the warp writes the result for the warp
    if (ix % 32 == 0) {
        warp_temp_and_result_buffer[iw] = warp_sum;
    }
    // Make sure all warps have written their results
    __syncthreads();

    dtype sample_sum = static_cast<dtype>(0.0);
    // The first warp reduces the results of the first stage
    // Note that from now on, ix corresponds to the index of the warp (from stage 1) in the block (previously iw)
    if (ix < 32) {
        const int iw_base = threadIdx.y * num_warps_per_sample;
        const dtype warp_sum_phase_1 =
            ix < num_warps_per_sample ? warp_temp_and_result_buffer[iw_base + ix] : static_cast<dtype>(0.0);
        // The partial sum will have constant segments, each segment corresponding to one sample (and containing as many values as
        // there are warps per sample).
        sample_sum = warp_reduce_sum(warp_sum_phase_1, num_warps_per_sample);
    }
    __syncthreads();
    return sample_sum;
}

template <typename dtype>
__device__ __forceinline__ void sample_distances(const dtype* points_sample,
                                                 const dtype* accum_distances_sample,
                                                 const dtype* distances_to_sample_sample,
                                                 int num_distances_to_sample, int num_points, int num_dims,
                                                 dtype* res_points_sample, bool relative_distances) {
    const int x = threadIdx.x;
    if (num_points == 0) {
        const int result_stride = blockDim.x * num_dims;

        dtype* res_points_current =
            x < num_distances_to_sample ? res_points_sample + x * num_dims : res_points_sample;
        for (int i = x; i < num_distances_to_sample; i += blockDim.x, res_points_current += result_stride) {
            fill_point_with_nan_common<dtype>(res_points_current, num_dims);
        }
        return;
    }

    dtype total_length_if_needed = static_cast<dtype>(0.0);
    if (relative_distances) {
        total_length_if_needed = accum_distances_sample[num_points - 1];
    }
    for (int i = x; i < num_distances_to_sample; i += blockDim.x) {
        dtype* res_points_current = res_points_sample + i * num_dims;
        const dtype distance_to_sample = relative_distances
                                             ? distances_to_sample_sample[i] * total_length_if_needed
                                             : distances_to_sample_sample[i];
        sample_at_distance_common<dtype, dtype>(points_sample, accum_distances_sample, distance_to_sample,
                                                num_points, num_dims, res_points_current);
    }
}

template <typename dtype>
__device__ __forceinline__ void compute_distances(dtype* points_sample, int num_points, int num_dims,
                                                  dtype* distances_sample) {
    const int x = threadIdx.x;
    if (num_points == 0) {
        return;
    }

    if (x == 0) {
        // Distance from the start to the first point is zero.
        distances_sample[0] = static_cast<dtype>(0.0);
    }
    // Store segment lengths starting at index 1 so that an *inclusive* prefix
    // sum over `distances_sample` yields distances to points:
    //   distances_sample[j] = distance from start to point j.
    for (int i = x; i < num_points - 1; i += blockDim.x) {
        distances_sample[i + 1] = compute_segment_length_common<dtype, dtype>(points_sample, i, num_dims);
    }
}

/**
 * @brief Shared implementation for both fixed-size and variable-size batch kernels.
 *
 * @details
 * This routine implements the common logic used by:
 *  - `polyline_sampling_fully_shared_kernel` (fixed-size batches), and
 *  - `polyline_sampling_fully_shared_var_batch_kernel` (variable-size batches).
 *
 * The shared memory is split into two parts:
 * - The first part stores the distances and accumulated distances
 *   (conversion in-place) for all points and has size (in elements):
 *   `blockDim.y * max_num_points`.
 * - The second part stores the temporary buffer used by
 *   `prefix_sum_looped` and has size (in elements):
 *   `blockDim.y * (num_warps_per_sample + 1)`.
 *   (see the documentation of `prefix_sum_looped` for details).
 * The total shared memory size is therefore (in elements):
 * `(blockDim.y * max_num_points + blockDim.y * (num_warps_per_sample + 1))`.
 *
 *
 * @tparam dtype The type of the points
 *
 * @param points The points to sample
 * @param distances_to_sample The distances to sample at
 * @param res_points The resulting sampled points
 * @param max_num_points The maximum number of points per polyline in the batch
 * @param max_num_points_full_blocks The maximum number of points extended to a multiple of blockDim.x
 * @param num_dims The number of dimensions of the points
 * @param max_num_distances_to_sample The maximum number of distances to sample at per polyline
 * @param num_samples The number of samples (batch size)
 * @param sample_sizes_points (optional) Per-sample number of points (variable-size batches)
 * @param sample_sizes_distances_to_sample (optional) Per-sample number of distances (variable-size batches)
 * @param relative_distances Interpret distances to sample as fractions of each polyline's total length
 * @param distance_buffer_ext Optional external buffer for distances when shared memory is insufficient
 */
template <typename dtype, typename sample_size_dtype, bool use_shared_distances, bool use_variable_size_batch>
__device__ __forceinline__ void polyline_sampling_fully_shared_common(
    dtype* points, dtype* distances_to_sample, dtype* res_points, int max_num_points,
    int max_num_points_full_blocks, int num_dims, int max_num_distances_to_sample, int num_samples,
    sample_size_dtype* sample_sizes_points, sample_size_dtype* sample_sizes_distances_to_sample,
    bool relative_distances, dtype* distance_buffer_ext) {
    extern __shared__ uint8_t shared_mem[];
    dtype* distances;
    dtype* buffer;
    if (use_shared_distances) {
        // Shared-memory layout per block:
        //   distances: [blockDim.y][max_num_points]
        //   buffer   : [blockDim.y * (1 + num_warps_per_sample)]
        distances = reinterpret_cast<dtype*>(shared_mem);
        buffer = reinterpret_cast<dtype*>(shared_mem + blockDim.y * max_num_points * sizeof(dtype));
    } else {
        // External distances buffer is laid out per block as
        //   [blockIdx.y][blockDim.y][max_num_points]
        // so each block gets its own contiguous slice. The scratch `buffer`
        // always starts at the beginning of this block's shared memory.
        distances = distance_buffer_ext + blockIdx.y * blockDim.y * max_num_points;
        buffer = reinterpret_cast<dtype*>(shared_mem);
    }

    const int y = threadIdx.y;
    const int y_global = blockIdx.y * blockDim.y + y;
    const bool is_active_sample = (y_global < num_samples);

    // 1) Compute per-point distances only for valid samples. Inactive rows in
    // the final block still participate in sync-heavy code paths with zero work.
    int curr_num_points = 0;
    int curr_num_distances_to_sample = 0;
    if (is_active_sample) {
        if (use_variable_size_batch) {
            curr_num_points = sample_sizes_points[y_global];
            curr_num_distances_to_sample = sample_sizes_distances_to_sample[y_global];
        } else {
            curr_num_points = max_num_points;
            curr_num_distances_to_sample = max_num_distances_to_sample;
        }

        // Global index for points in device memory; distances remain indexed by the
        // local y within the block because they live in shared memory.
        dtype* points_sample = points + y_global * max_num_points * num_dims;
        dtype* distances_sample = distances + y * max_num_points;
        if (curr_num_points > 0) {
            compute_distances<dtype>(points_sample, curr_num_points, num_dims, distances_sample);
        }
    }

    // 2) Prefix-sum over distances for all rows in this block-local buffer.
    //    This operates purely on (shared or external) distances, so it is
    //    safe even for rows that don't correspond to a real sample; their
    //    results are never used.
    // The `distances` are per-block, so we use the local index `y` to access the distances for the current block.
    dtype* distance = distances + y * max_num_points;
    prefix_sum_looped<dtype>(distance,                    // sequences
                             buffer,                      // buffer (sum_buffer + warp_scan_buffer)
                             curr_num_points,             // numel_x
                             max_num_points_full_blocks,  // numel_x_full_blocks (extended to full blocks)
                             blockDim.y,                  // numel_y (number of samples per block)
                             static_cast<dtype>(0.0)      // offset
    );

    // 3) Sample only for valid samples, using their (possibly shared or
    //    external) accumulated distances.
    if (is_active_sample) {
        // Get the points for the current sample (use of global offset)
        const dtype* points_sample = points + y_global * max_num_points * num_dims;
        // Get the distances for the current sample (use of block-local offset, as distances are stored in
        // shared memory (or in an external buffer with `points` referring to points for this block))
        const dtype* distances_sample = distances + y * max_num_points;
        // Get the distances to sample at for the current sample (use of global offset)
        const dtype* distances_to_sample_sample =
            distances_to_sample + y_global * max_num_distances_to_sample;
        sample_distances<dtype>(points_sample, distances_sample, distances_to_sample_sample,
                                curr_num_distances_to_sample, curr_num_points, num_dims,
                                res_points + y_global * max_num_distances_to_sample * num_dims,
                                relative_distances);
    }
}

/**
 * @brief Sample the points at the distances (fixed-size batches).
 *
 * See `polyline_sampling_fully_shared_common` for implementation details.
 */
template <typename dtype, bool use_shared_distances>
__global__ void polyline_sampling_fully_shared_kernel(dtype* points, dtype* distances_to_sample,
                                                      dtype* res_points, int num_points,
                                                      int num_points_full_blocks, int num_dims,
                                                      int num_distances_to_sample, int num_samples,
                                                      bool relative_distances, dtype* distance_buffer_ext) {
    polyline_sampling_fully_shared_common<dtype, int, use_shared_distances, false>(
        points, distances_to_sample, res_points,
        num_points,              // max_num_points
        num_points_full_blocks,  // max_num_points_full_blocks
        num_dims,
        num_distances_to_sample,  // max_num_distances_to_sample
        num_samples,
        /*sample_sizes_points=*/nullptr,
        /*sample_sizes_distances_to_sample=*/nullptr, relative_distances, distance_buffer_ext);
}

// Variable-size batch version of the kernel.
template <typename dtype, typename sample_size_dtype, bool use_shared_distances>
__global__ void polyline_sampling_fully_shared_var_batch_kernel(
    dtype* points, dtype* distances_to_sample, dtype* res_points, int max_num_points,
    int max_num_points_full_blocks, int num_dims, int max_num_distances_to_sample, int num_samples,
    sample_size_dtype* sample_sizes_points, sample_size_dtype* sample_sizes_distances_to_sample,
    bool relative_distances, dtype* distance_buffer_ext) {
    polyline_sampling_fully_shared_common<dtype, sample_size_dtype, use_shared_distances, true>(
        points, distances_to_sample, res_points, max_num_points, max_num_points_full_blocks, num_dims,
        max_num_distances_to_sample, num_samples, sample_sizes_points, sample_sizes_distances_to_sample,
        relative_distances, distance_buffer_ext);
}

template <typename dtype, typename sample_size_dtype, bool use_variable_size_batch>
__device__ __forceinline__ void polyline_lengths_common(dtype* points, dtype* lengths, int max_num_points,
                                                        int num_dims, int num_samples,
                                                        sample_size_dtype* sample_sizes_points,
                                                        dtype* reduction_buffer) {
    const int x = threadIdx.x;
    const int y_global = blockIdx.y * blockDim.y + threadIdx.y;
    const bool is_active_sample = (y_global < num_samples);

    int curr_num_points = 0;
    dtype local_length = static_cast<dtype>(0.0);
    if (is_active_sample) {
        curr_num_points = use_variable_size_batch ? sample_sizes_points[y_global] : max_num_points;
        const dtype* points_sample = points + y_global * max_num_points * num_dims;
        for (int i = x; i < curr_num_points - 1; i += blockDim.x) {
            local_length += compute_segment_length_common<dtype, dtype>(points_sample, i, num_dims);
        }
    }

    const int num_warps_per_sample = (blockDim.x + 31) / 32;
    const dtype length = sample_reduce_sum<dtype>(local_length, num_warps_per_sample, reduction_buffer);
    if (is_active_sample && x == 0) {
        lengths[y_global] = curr_num_points == 0 ? polyline_nan<dtype>() : length;
    }
}

template <typename dtype>
__global__ void polyline_lengths_kernel(dtype* points, dtype* lengths, int num_points, int num_dims,
                                        int num_samples) {
    extern __shared__ uint8_t shared_mem[];
    dtype* reduction_buffer = reinterpret_cast<dtype*>(shared_mem);
    polyline_lengths_common<dtype, int, false>(points, lengths, num_points, num_dims, num_samples,
                                               /*sample_sizes_points=*/nullptr, reduction_buffer);
}

template <typename dtype, typename sample_size_dtype>
__global__ void polyline_lengths_var_batch_kernel(dtype* points, dtype* lengths, int max_num_points,
                                                  int num_dims, int num_samples,
                                                  sample_size_dtype* sample_sizes_points) {
    extern __shared__ uint8_t shared_mem[];
    dtype* reduction_buffer = reinterpret_cast<dtype*>(shared_mem);
    polyline_lengths_common<dtype, sample_size_dtype, true>(
        points, lengths, max_num_points, num_dims, num_samples, sample_sizes_points, reduction_buffer);
}

}  // namespace polyline

#endif  // LANE_HELPERS_POLYLINE_KERNELS_CUH
