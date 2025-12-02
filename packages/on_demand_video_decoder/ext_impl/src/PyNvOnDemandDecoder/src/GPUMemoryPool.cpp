/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "GPUMemoryPool.hpp"

#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "NvCodecUtils.h"

static void cuda_err_check() {
    const cudaError last_error = cudaGetLastError();
    if (last_error != cudaError::cudaSuccess) {
        throw std::runtime_error("Cuda error encountered; Error code: " + std::to_string(last_error));
    }
}

GPUMemoryPool::GPUMemoryPool(size_t num_bytes_to_store) { EnsureSizeAndSoftReset(num_bytes_to_store, false); }

void* GPUMemoryPool::AddElement(size_t num_bytes) {
    const size_t size_needed = curr_size_ + num_bytes;
    if (size_needed > allocated_size_) {
        throw std::out_of_range(
            "Memory pool does not have enough allocated "
            "memory. Allow re-allocating when calling this "
            "method or pre-allocate enough memory.");
    }
    uint8_t* pointer_to_new_element = data_ + curr_size_;
    curr_size_ += num_bytes;
    return static_cast<void*>(pointer_to_new_element);
}

void GPUMemoryPool::EnsureSizeAndSoftReset(size_t num_bytes_to_store, bool shrink_if_smaller) {
    const bool curr_too_small = curr_size_ < num_bytes_to_store;
    if (curr_too_small || shrink_if_smaller) {
        if (data_ != nullptr) {
            cuMemFree(reinterpret_cast<CUdeviceptr>(data_));
            cuda_err_check();
        }
        CUDA_DRVAPI_CALL(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&data_), num_bytes_to_store));
        cuda_err_check();

        allocated_size_ = num_bytes_to_store;
    }
    curr_size_ = 0;
}

void GPUMemoryPool::SoftRelease() { curr_size_ = 0; }

void GPUMemoryPool::HardRelease() {
    // Only free if memory was actually allocated
    // cuMemFree(0) is undefined behavior, unlike free(NULL) in C
    if (data_ != nullptr) {
        cuMemFree(reinterpret_cast<CUdeviceptr>(data_));
    }
    data_ = nullptr;
    curr_size_ = 0;
    allocated_size_ = 0;
}

GPUMemoryPool::~GPUMemoryPool() { HardRelease(); }
