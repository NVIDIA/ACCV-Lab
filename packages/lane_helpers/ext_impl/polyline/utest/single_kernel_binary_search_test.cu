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

#include "polyline_common.cuh"

using namespace polyline;

template <typename dtype>
static __global__ void binary_search_test_kernel(dtype* values, int num_values, dtype* to_search,
                                                 int num_to_search, int* results) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < num_to_search) {
        const int idx = get_index_of_last_lower_or_equal_to_common<dtype>(values, to_search[ix], num_values);
        results[ix] = idx;
    }
}

static int get_index_of_last_lower_or_equal_to_reference(float* values, float to_search, int num_values) {
    if (to_search < values[0]) {
        return -1;
    } else if (to_search > values[num_values - 1]) {
        return num_values - 1;
    } else {
        for (int i = 0; i < num_values; ++i) {
            if (values[i] > to_search) {
                return i - 1;
            }
        }
    }
    // If we reach here, all values are <= to_search, so return last index.
    return num_values - 1;
}

void run_test(int threads_per_block, int num_values, int num_to_search) {
    float* values = nullptr;
    cudaMallocManaged<float>(&values, num_values * sizeof(float));

    float* to_search = nullptr;
    cudaMallocManaged<float>(&to_search, num_to_search * sizeof(float));

    int* results = nullptr;
    cudaMallocManaged<int>(&results, num_to_search * sizeof(int));

    for (int i = 0; i < num_values; i++) {
        const float ii = static_cast<float>(i);
        values[i] = ii * ii;
    }

    const float max_val = values[num_values - 1];
    for (int i = 0; i < num_to_search; i++) {
        to_search[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * max_val;
    }

    int* results_expected = new int[num_to_search];

    for (int a = 0; a < num_to_search; ++a) {
        results_expected[a] = get_index_of_last_lower_or_equal_to_reference(values, to_search[a], num_values);
    }

    const int num_blocks = (num_to_search + threads_per_block - 1) / threads_per_block;

    binary_search_test_kernel<float>
        <<<num_blocks, threads_per_block>>>(values, num_values, to_search, num_to_search, results);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_to_search; i++) {
        EXPECT_EQ(results[i], results_expected[i]);
    }

    cudaFree(values);
    cudaFree(to_search);
    cudaFree(results);
    delete[] results_expected;
}

TEST(SingleKernelBinarySearchTest, SearchTest) { run_test(1024, 1024, 128); }