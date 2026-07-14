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
#include <cmath>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "polyline_kernels.cuh"

using namespace polyline;

// Kernel used for testing `prefix_sum_looped`
// Note that there is a buffer passed to the kernel. In the actual use-case,
// shared memory is used for the buffer instead.
template <typename dtype>
static __global__ void compute_distances_kernel(dtype* points, int num_points, int num_dims, int num_samples,
                                                dtype* distances_res) {
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= num_samples) {
        return;
    }
    dtype* points_sample = points + iy * num_points * num_dims;
    dtype* distances_res_sample = distances_res + iy * num_points;
    compute_distances<dtype>(points_sample, num_points, num_dims, distances_res_sample);
}

void run_test(int threads_per_block, int num_points, int num_samples) {
    const static int num_dims = 3;

    float* points = nullptr;
    cudaMallocManaged<float>(&points, num_points * num_dims * num_samples * sizeof(float));

    float* distances_res = nullptr;
    cudaMallocManaged<float>(&distances_res, num_points * num_samples * sizeof(float));

    float* distances_expected = new float[num_points * num_samples];

    for (int a = 0; a < num_samples; a++) {
        for (int i = 0; i < num_points; i++) {
            const float aa = static_cast<float>(a);
            const float ii = static_cast<float>(i);
            const size_t offset = a * num_points * num_dims + i * num_dims;
            points[offset + 0] = std::sin(ii + aa * 0.1f);
            points[offset + 1] = std::cos(ii + aa * 0.1f);
            points[offset + 2] = std::log(ii + aa * 0.1f + 1.0f);
        }
    }

    for (int a = 0; a < num_samples; a++) {
        const size_t offset_dist = a * num_points;
        distances_expected[offset_dist + 0] = 0.0f;  // distance to first point
        for (int i = 1; i < num_points; i++) {
            float diff = 0.0f;
            const size_t offset_point = (offset_dist + i - 1) * num_dims;
            for (int d = 0; d < num_dims; d++) {
                const float diff_d = points[offset_point + d] - points[offset_point + d + num_dims];
                diff += diff_d * diff_d;
            }
            distances_expected[offset_dist + i] = std::sqrt(diff);
        }
    }

    const dim3 block_dim(threads_per_block, 1);
    const dim3 grid_dim(1, num_samples);

    compute_distances_kernel<float>
        <<<grid_dim, block_dim>>>(points, num_points, num_dims, num_samples, distances_res);
    cudaDeviceSynchronize();

    for (int a = 0; a < num_samples; a++) {
        const size_t offset_dist = a * num_points;
        for (int i = 0; i < num_points; i++) {
            EXPECT_NEAR(distances_res[offset_dist + i], distances_expected[offset_dist + i], 1e-4f);
        }
    }

    cudaFree(points);
    cudaFree(distances_res);
    delete[] distances_expected;
}

TEST(SingleKernelComputeDistancesTest, SingleBlockXSingleSequenceTest) { run_test(1024, 1024, 1); }

TEST(SingleKernelComputeDistancesTest, MultiBlockXMultiSequenceTest) { run_test(1024, 2028, 4); }