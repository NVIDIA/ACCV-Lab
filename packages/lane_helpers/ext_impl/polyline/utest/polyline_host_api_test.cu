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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "polyline.cuh"

namespace {

constexpr float EPS = 1e-5f;

void expect_near_point(const float* values, const float* expected, int num_dims) {
    for (int d = 0; d < num_dims; ++d) {
        EXPECT_NEAR(values[d], expected[d], EPS);
    }
}

}  // namespace

TEST(PolylineHostApiTest, NativeCudaAndCpuEntryPointsAreCallable) {
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int device = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));

    constexpr int NUM_DIMS = 2;

    float* fixed_points = nullptr;
    float* fixed_distances = nullptr;
    float* fixed_result = nullptr;
    float* fixed_lengths = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&fixed_points, 3 * NUM_DIMS * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&fixed_distances, 3 * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&fixed_result, 3 * NUM_DIMS * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&fixed_lengths, sizeof(float)));

    fixed_points[0] = 0.0f;
    fixed_points[1] = 0.0f;
    fixed_points[2] = 2.0f;
    fixed_points[3] = 0.0f;
    fixed_points[4] = 2.0f;
    fixed_points[5] = 2.0f;
    fixed_distances[0] = 0.0f;
    fixed_distances[1] = 1.0f;
    fixed_distances[2] = 4.0f;

    const auto fixed_cfg = polyline::make_polyline_launch_config<float>(3, 1, device);
    float* fixed_distance_buffer = nullptr;
    if (fixed_cfg.distance_buffer_ext_size_elems > 0) {
        ASSERT_EQ(cudaSuccess, cudaMallocManaged(&fixed_distance_buffer,
                                                 fixed_cfg.distance_buffer_ext_size_elems * sizeof(float)));
    }
    polyline::polyline_interpolation<float>(fixed_points, 3, NUM_DIMS, fixed_distances, 3, fixed_result, 1,
                                            false, device, fixed_cfg, fixed_distance_buffer, stream);
    polyline::polyline_lengths<float>(fixed_points, 3, NUM_DIMS, fixed_lengths, 1, stream);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const float fixed_expected0[NUM_DIMS] = {0.0f, 0.0f};
    const float fixed_expected1[NUM_DIMS] = {1.0f, 0.0f};
    const float fixed_expected2[NUM_DIMS] = {2.0f, 2.0f};
    expect_near_point(fixed_result + 0 * NUM_DIMS, fixed_expected0, NUM_DIMS);
    expect_near_point(fixed_result + 1 * NUM_DIMS, fixed_expected1, NUM_DIMS);
    expect_near_point(fixed_result + 2 * NUM_DIMS, fixed_expected2, NUM_DIMS);
    EXPECT_NEAR(fixed_lengths[0], 4.0f, EPS);

    float* ragged_points = nullptr;
    float* ragged_distances = nullptr;
    float* ragged_result = nullptr;
    float* ragged_lengths = nullptr;
    int64_t* sample_sizes_points = nullptr;
    int64_t* sample_sizes_distances = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&ragged_points, 2 * NUM_DIMS * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&ragged_distances, 2 * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&ragged_result, 2 * NUM_DIMS * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&ragged_lengths, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&sample_sizes_points, sizeof(int64_t)));
    ASSERT_EQ(cudaSuccess, cudaMallocManaged(&sample_sizes_distances, sizeof(int64_t)));

    ragged_points[0] = 10.0f;
    ragged_points[1] = 0.0f;
    ragged_points[2] = 12.0f;
    ragged_points[3] = 0.0f;
    ragged_distances[0] = 0.0f;
    ragged_distances[1] = 0.5f;
    sample_sizes_points[0] = 2;
    sample_sizes_distances[0] = 2;

    const auto ragged_cfg = polyline::make_polyline_launch_config<float>(2, 1, device);
    float* ragged_distance_buffer = nullptr;
    if (ragged_cfg.distance_buffer_ext_size_elems > 0) {
        ASSERT_EQ(cudaSuccess, cudaMallocManaged(&ragged_distance_buffer,
                                                 ragged_cfg.distance_buffer_ext_size_elems * sizeof(float)));
    }
    polyline::polyline_interpolation_var_size_batch<float, int64_t>(
        ragged_points, 2, NUM_DIMS, ragged_distances, 2, ragged_result, 1, sample_sizes_points,
        sample_sizes_distances, true, device, ragged_cfg, ragged_distance_buffer, stream);
    polyline::polyline_lengths_var_size_batch<float, int64_t>(ragged_points, 2, NUM_DIMS, ragged_lengths, 1,
                                                              sample_sizes_points, stream);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const float ragged_expected0[NUM_DIMS] = {10.0f, 0.0f};
    const float ragged_expected1[NUM_DIMS] = {11.0f, 0.0f};
    expect_near_point(ragged_result + 0 * NUM_DIMS, ragged_expected0, NUM_DIMS);
    expect_near_point(ragged_result + 1 * NUM_DIMS, ragged_expected1, NUM_DIMS);
    EXPECT_NEAR(ragged_lengths[0], 2.0f, EPS);

    float cpu_points[3 * NUM_DIMS] = {0.0f, 0.0f, 2.0f, 0.0f, 2.0f, 2.0f};
    float cpu_distances[2] = {0.0f, 0.5f};
    float cpu_result[2 * NUM_DIMS] = {};
    float cpu_lengths[1] = {};
    polyline::polyline_interpolation_cpu<float>(cpu_points, 3, NUM_DIMS, cpu_distances, 2, cpu_result, 1,
                                                true);
    polyline::polyline_lengths_cpu<float>(cpu_points, 3, NUM_DIMS, cpu_lengths, 1);

    const float cpu_expected1[NUM_DIMS] = {2.0f, 0.0f};
    expect_near_point(cpu_result + 0 * NUM_DIMS, fixed_expected0, NUM_DIMS);
    expect_near_point(cpu_result + 1 * NUM_DIMS, cpu_expected1, NUM_DIMS);
    EXPECT_NEAR(cpu_lengths[0], 4.0f, EPS);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    cudaFree(fixed_points);
    cudaFree(fixed_distances);
    cudaFree(fixed_result);
    cudaFree(fixed_lengths);
    if (fixed_distance_buffer != nullptr) {
        cudaFree(fixed_distance_buffer);
    }
    cudaFree(ragged_points);
    cudaFree(ragged_distances);
    cudaFree(ragged_result);
    cudaFree(ragged_lengths);
    cudaFree(sample_sizes_points);
    cudaFree(sample_sizes_distances);
    if (ragged_distance_buffer != nullptr) {
        cudaFree(ragged_distance_buffer);
    }
}
