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
static __global__ void prefix_sum_kernel(dtype* sequences, dtype* buffer, int numel_x, int numel_y) {
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= numel_y) {
        return;
    }

    // Extend the number of elements to the next multiple of the block size in x.
    // This is done on the CPU in the actual implementation, but is done here for simplicity.
    const int num_x_full_blocks = ((numel_x + blockDim.x - 1) / blockDim.x) * blockDim.x;

    const int num_warps_per_sample = (blockDim.x + 31) / 32;
    dtype* sequence = sequences + iy * numel_x;
    dtype* buffer_block = buffer + blockIdx.y * blockDim.y * (num_warps_per_sample + 1);
    prefix_sum_looped<dtype>(sequence, buffer_block, numel_x, num_x_full_blocks, numel_y, 0.0f);
}

void run_test(int block_size_x, int block_size_y, int numel_x, int numel_y) {
    const int num_warps_per_sample = (block_size_x + 31) / 32;
    const int grid_y = (numel_y + block_size_y - 1) / block_size_y;
    size_t buffer_size = grid_y * block_size_y * (num_warps_per_sample + 1);

    // Use managed memory in test for simplicity
    float* sequences = nullptr;
    cudaMallocManaged<float>(&sequences, numel_x * numel_y * sizeof(float));
    float* expected = new float[numel_x * numel_y];

    float* buffer = nullptr;
    cudaMallocManaged<float>(&buffer, buffer_size * sizeof(float));

    for (int j = 0; j < numel_y; j++) {
        const size_t sequence_offset = j * numel_x;
        for (int i = 0; i < numel_x; i++) {
            float ii = static_cast<float>(i);
            sequences[sequence_offset + i] = std::sin(ii) + std::log(ii + 1.0f);
            expected[sequence_offset + i] =
                sequences[sequence_offset + i] + (i == 0 ? 0.0f : expected[sequence_offset + i - 1]);
        }
    }

    const dim3 block_dim(block_size_x, block_size_y);
    // Note that the grid dimension is always 1, as the prefix sum is performed in a single block
    // for all samples in the batch.
    const dim3 grid_dim(1, grid_y);

    prefix_sum_kernel<float><<<grid_dim, block_dim>>>(sequences, buffer, numel_x, numel_y);
    cudaDeviceSynchronize();

    for (int j = 0; j < numel_y; j++) {
        const size_t sequence_offset = j * numel_x;
        for (int i = 0; i < numel_x; i++) {
            const float max_abs_error = 1e-4f * std::abs(sequences[sequence_offset + i]);
            EXPECT_NEAR(sequences[sequence_offset + i], expected[sequence_offset + i], max_abs_error);
        }
    }

    cudaFree(sequences);
    cudaFree(buffer);
    delete[] expected;
}

TEST(PrefixScanTest, SingleBlockXSingleSequenceTest) { run_test(1024, 1, 1024, 1); }

TEST(PrefixScanTest, SingleBlockXSingleSequenceNumWarps2Test) { run_test(64, 1, 64, 1); }

TEST(PrefixScanTest, SingleBlockXSmallerBlockSizeTest) { run_test(64, 8, 64, 32); }

TEST(PrefixScanTest, MultiBlockXSingleSequenceTest) { run_test(1024, 1, 8192, 1); }

TEST(PrefixScanTest, MultiBlockXMultiSequenceSingleBlockYTest) { run_test(1024, 1, 2048, 2); }

TEST(PrefixScanTest, MultiBlockXMultiSequenceMultiBlockYTest) { run_test(512, 2, 1024, 16); }

TEST(PrefixScanTest, NonPowerOf2ProblemSizes) {
    run_test(512, 2, 1024 - 1, 16);
    run_test(512, 2, 1024 + 1, 16);
    run_test(512, 2, 512 + 256, 16);
    run_test(512, 2, 512 + 256 - 1, 16);
    run_test(512, 2, 512 + 256 + 1, 16);
    run_test(512, 2, 1, 16);
    run_test(512, 2, 3, 16);
}