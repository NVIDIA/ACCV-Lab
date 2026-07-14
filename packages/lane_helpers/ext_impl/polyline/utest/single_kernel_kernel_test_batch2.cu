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

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "polyline_kernels.cuh"

using namespace polyline;

TEST(PolylineSamplingFullySharedKernelTestBatch2, SimpleRectanglePolylineBatch2) {
    // Simple axis-aligned rectangle as in `single_kernel_kernel_test.cu`, but with batch size 2
    constexpr int NUM_POINTS = 5;
    constexpr int NUM_DIMS = 2;
    constexpr int NUM_DISTANCES = 11;
    constexpr int NUM_SAMPLES = 2;

    const float rectangle[NUM_POINTS][NUM_DIMS] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 2.0f}, {0.0f, 2.0f}, {0.0f, 0.0f}};

    const float distances_to_sample[NUM_SAMPLES][NUM_DISTANCES] = {
        {0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f},
        {0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f}};

    const float expected[NUM_DISTANCES][NUM_DIMS] = {{0.0f, 0.0f}, {0.5f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
                                                     {1.0f, 2.0f}, {0.5f, 2.0f}, {0.0f, 2.0f}, {0.0f, 1.5f},
                                                     {0.0f, 1.0f}, {0.0f, 0.5f}, {0.0f, 0.0f}};

    constexpr float eps = 1e-5f;

    // Allocate unified memory
    float* points = nullptr;
    cudaMallocManaged(&points, NUM_POINTS * NUM_DIMS * NUM_SAMPLES * sizeof(float));

    float* distances_to_sample_gpu = nullptr;
    cudaMallocManaged(&distances_to_sample_gpu, NUM_SAMPLES * NUM_DISTANCES * sizeof(float));

    float* res_points = nullptr;
    cudaMallocManaged(&res_points, NUM_SAMPLES * NUM_DISTANCES * NUM_DIMS * sizeof(float));

    // Initialize input polylines (two identical samples)
    for (int s = 0; s < NUM_SAMPLES; ++s) {
        for (int i = 0; i < NUM_POINTS; ++i) {
            for (int d = 0; d < NUM_DIMS; ++d) {
                points[s * NUM_POINTS * NUM_DIMS + i * NUM_DIMS + d] = rectangle[i][d];
            }
        }
    }

    // Initialize per-sample distances to sample
    for (int s = 0; s < NUM_SAMPLES; ++s) {
        for (int i = 0; i < NUM_DISTANCES; ++i) {
            distances_to_sample_gpu[s * NUM_DISTANCES + i] = distances_to_sample[s][i];
        }
    }

    // Launch kernel
    const dim3 block_dim(64, 2);
    const dim3 grid_dim(1, (NUM_SAMPLES + block_dim.y - 1) / block_dim.y);

    // Extend number of points to a multiple of block_dim.x so that
    // `prefix_sum_looped` can iterate in full blocks without leaving
    // some threads skipping iterations that contain `__syncthreads()`.
    const int num_points_full_blocks = ((NUM_POINTS + block_dim.x - 1) / block_dim.x) * block_dim.x;
    const size_t shared_mem_size =
        (block_dim.y * NUM_POINTS + block_dim.x * (block_dim.y + 1) / 32 + block_dim.y) * sizeof(float);

    polyline_sampling_fully_shared_kernel<float, true><<<grid_dim, block_dim, shared_mem_size>>>(
        points, distances_to_sample_gpu, res_points, NUM_POINTS, num_points_full_blocks, NUM_DIMS,
        NUM_DISTANCES, NUM_SAMPLES, false, nullptr);

    cudaDeviceSynchronize();

    // Verify output for both samples
    for (int s = 0; s < NUM_SAMPLES; ++s) {
        for (int i = 0; i < NUM_DISTANCES; ++i) {
            for (int d = 0; d < NUM_DIMS; ++d) {
                const int idx = (s * NUM_DISTANCES + i) * NUM_DIMS + d;
                EXPECT_NEAR(res_points[idx], expected[i][d], eps);
            }
        }
    }

    cudaFree(points);
    cudaFree(distances_to_sample_gpu);
    cudaFree(res_points);
}
