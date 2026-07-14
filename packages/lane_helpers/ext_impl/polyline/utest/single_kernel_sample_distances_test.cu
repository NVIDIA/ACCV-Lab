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
#include <algorithm>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "polyline_common.cuh"

using namespace polyline;

template <typename dtype>
static __global__ void sample_at_distance_test_kernel(dtype* points, dtype* accum_distances,
                                                      dtype* distances_to_sample, int num_points,
                                                      int num_dims, int num_distances, dtype* res_points) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < num_distances) {
        sample_at_distance_common<dtype, dtype>(points, accum_distances, distances_to_sample[ix], num_points,
                                                num_dims, res_points + ix * num_dims);
    }
}

// Reference implementation for testing
template <typename dtype>
void sample_at_distance_reference(dtype* points, dtype* accum_distances, dtype distance_to_sample_at,
                                  int num_points, int num_dims, dtype* res_point) {
    int index_min = -1;
    for (int i = 0; i < num_points - 1; ++i) {
        if (accum_distances[i] <= distance_to_sample_at && accum_distances[i + 1] > distance_to_sample_at) {
            index_min = i;
            break;
        }
    }

    if (index_min >= 0 && index_min < num_points - 1) {
        const int index_max = std::min(index_min + 1, num_points - 1);
        const dtype* min_point = points + index_min * num_dims;
        const dtype* max_point = points + index_max * num_dims;
        const dtype dist_min = accum_distances[index_min];
        const dtype dist_max = accum_distances[index_max];
        const dtype dist = dist_max - dist_min;

        if (dist >= std::numeric_limits<dtype>::epsilon()) {
            const dtype weight_max = (distance_to_sample_at - dist_min) / dist;
            const dtype weight_min = (dist_max - distance_to_sample_at) / dist;
            for (int d = 0; d < num_dims; ++d) {
                res_point[d] = min_point[d] * weight_min + max_point[d] * weight_max;
            }
        } else {
            for (int d = 0; d < num_dims; ++d) {
                res_point[d] = min_point[d];
            }
        }
    } else if (distance_to_sample_at <= accum_distances[0]) {
        for (int d = 0; d < num_dims; ++d) {
            res_point[d] = points[d];
        }
    } else {
        for (int d = 0; d < num_dims; ++d) {
            res_point[d] = points[(num_points - 1) * num_dims + d];
        }
    }
}

void run_test(int threads_per_block, int num_points, int num_dims, int num_distances) {
    // Allocate memory
    float* points = nullptr;
    cudaMallocManaged(&points, num_points * num_dims * sizeof(float));

    float* accum_distances = nullptr;
    cudaMallocManaged(&accum_distances, num_points * sizeof(float));

    float* distances_to_sample = nullptr;
    cudaMallocManaged(&distances_to_sample, num_distances * sizeof(float));

    float* res_points = nullptr;
    cudaMallocManaged(&res_points, num_distances * num_dims * sizeof(float));

    // Initialize points in a simple pattern (e.g., a line in 2D)
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < num_dims; ++d) {
            points[i * num_dims + d] = static_cast<float>(i);
        }
    }

    // Initialize accumulated distances
    accum_distances[0] = 0.0f;
    for (int i = 1; i < num_points; ++i) {
        float dist = 0.0f;
        for (int d = 0; d < num_dims; ++d) {
            float diff = points[i * num_dims + d] - points[(i - 1) * num_dims + d];
            dist += diff * diff;
        }
        accum_distances[i] = accum_distances[i - 1] + std::sqrt(dist);
    }

    // Generate test distances
    const float max_dist = accum_distances[num_points - 1];
    for (int i = 0; i < num_distances; ++i) {
        distances_to_sample[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * max_dist;
    }

    // Compute expected results using reference implementation
    std::vector<float> expected_points(num_distances * num_dims);
    for (int i = 0; i < num_distances; ++i) {
        sample_at_distance_reference<float>(points, accum_distances, distances_to_sample[i], num_points,
                                            num_dims, &expected_points[i * num_dims]);
    }

    // Run CUDA kernel
    const int num_blocks = (num_distances + threads_per_block - 1) / threads_per_block;
    sample_at_distance_test_kernel<float><<<num_blocks, threads_per_block>>>(
        points, accum_distances, distances_to_sample, num_points, num_dims, num_distances, res_points);
    cudaDeviceSynchronize();

    // Verify results
    const float epsilon = 1e-4f;
    for (int i = 0; i < num_distances; ++i) {
        for (int d = 0; d < num_dims; ++d) {
            EXPECT_NEAR(res_points[i * num_dims + d], expected_points[i * num_dims + d], epsilon)
                << "Mismatch at distance " << i << ", dimension " << d;
        }
    }

    // Cleanup
    cudaFree(points);
    cudaFree(accum_distances);
    cudaFree(distances_to_sample);
    cudaFree(res_points);
}

TEST(SingleKernelSampleDistancesTest, BasicTest) { run_test(1024, 100, 2, 128); }

TEST(SingleKernelSampleDistancesTest, LargeTest) { run_test(1024, 1000, 3, 256); }

TEST(SingleKernelSampleDistancesTest, EdgeCases) {
    // Test with very few points
    run_test(1024, 2, 2, 10);

    // Test with many dimensions
    run_test(1024, 50, 10, 64);
}
