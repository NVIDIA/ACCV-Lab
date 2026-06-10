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

#ifndef BATCHING_HELPERS_CPP_IMPL_BATCHED_INDEXING_ACCESS_HELPERS_H
#define BATCHING_HELPERS_CPP_IMPL_BATCHED_INDEXING_ACCESS_HELPERS_H

#include <vector>

#include <torch/torch.h>

// Compability macros for older PyTorch versions
// clang-format off
#ifndef AT_DISPATCH_FLOATING_TYPES_AND3
#define AT_DISPATCH_FLOATING_TYPES_AND3(TYPE1, TYPE2, TYPE3, SCALAR_TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(SCALAR_TYPE, NAME, \
        AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__); \
        AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE1, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE2, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE3, __VA_ARGS__))
#endif

#ifndef AT_DISPATCH_FLOATING_TYPES_AND4
#define AT_DISPATCH_FLOATING_TYPES_AND4(TYPE1, TYPE2, TYPE3, TYPE4, SCALAR_TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(SCALAR_TYPE, NAME, \
        AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__); \
        AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE1, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE2, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE3, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE4, __VA_ARGS__))
#endif

#ifndef AT_DISPATCH_FLOATING_TYPES_AND5
#define AT_DISPATCH_FLOATING_TYPES_AND5(TYPE1, TYPE2, TYPE3, TYPE4, TYPE5, SCALAR_TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(SCALAR_TYPE, NAME, \
        AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__); \
        AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE1, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE2, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE3, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE4, __VA_ARGS__); \
        AT_DISPATCH_CASE(TYPE5, __VA_ARGS__))
#endif
// clang-format on

#define DISPATCH_CASE_INDEX_TYPES(...)                 \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INDEX_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INDEX_TYPES(__VA_ARGS__))

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_CUDA_DEVICE(tensors_list...)                                                         \
    {                                                                                                   \
        const std::vector<torch::Tensor> tensors = {tensors_list};                                      \
        CHECK_CUDA(tensors[0]);                                                                         \
        const auto& device = tensors[0].device();                                                       \
        for (size_t i = 1; i < tensors.size(); ++i) {                                                   \
            TORCH_CHECK(tensors[i].device() == device, "All input tensors must be on the same device"); \
        }                                                                                               \
    }
#define CHECK_SAME_DTYPE(error_msg, tensors_list...)                                      \
    {                                                                                     \
        const std::vector<torch::Tensor> tensors = {tensors_list};                        \
        for (size_t i = 1; i < tensors.size(); ++i) {                                     \
            TORCH_CHECK(tensors[i].scalar_type() == tensors[0].scalar_type(), error_msg); \
        }                                                                                 \
    }

#define CHECK_SIZE_MATCH(tensor1, tensor2)                                                     \
    {                                                                                          \
        /* If the tensors are empty, the actual sizes are not relevant */                      \
        if (!((tensor1).numel() == 0 && (tensor2).numel() == 0)) {                             \
            TORCH_CHECK((tensor1).dim() == (tensor2).dim(),                                    \
                        #tensor1 " and " #tensor2 " must have the same number of dimensions"); \
            for (size_t i = 0; i < (tensor1).dim(); ++i) {                                     \
                TORCH_CHECK((tensor1).size(i) == (tensor2).size(i),                            \
                            #tensor1 " and " #tensor2 " must have the same size");             \
            }                                                                                  \
        }                                                                                      \
    }

#define CHECK_SIZE_MATCH_FIRST_DIMS(tensor1, tensor2, num_dims_to_check)                                     \
    {                                                                                                        \
        /* If the tensors are empty, the actual sizes are not relevant */                                    \
        if (!((tensor1).numel() == 0 && (tensor2).numel() == 0)) {                                           \
            TORCH_CHECK((tensor1).dim() >= (num_dims_to_check) && (tensor2).dim() >= (num_dims_to_check),    \
                        #tensor1 " and " #tensor2 " must have at least " +                                   \
                            std::to_string(num_dims_to_check) + " dimensions");                              \
            for (size_t i = 0; i < (num_dims_to_check); ++i) {                                               \
                TORCH_CHECK(                                                                                 \
                    (tensor1).size(i) == (tensor2).size(i),                                                  \
                    #tensor1 " and " #tensor2 " must have the same size in dimension " + std::to_string(i)); \
            }                                                                                                \
        }                                                                                                    \
    }

#define CHECK_SIZE_MATCH_EXCEPT_DIM(tensor1, tensor2, dim_to_allow_mismatch)                                 \
    {                                                                                                        \
        /* If the tensors are empty, the actual sizes are not relevant */                                    \
        if (!((tensor1).numel() == 0 && (tensor2).numel() == 0)) {                                           \
            TORCH_CHECK((tensor1).dim() == (tensor2).dim(),                                                  \
                        #tensor1 " and " #tensor2 " must have the same number of dimensions");               \
            for (size_t i = 0; i < (tensor1).dim(); ++i) {                                                   \
                if (i == (dim_to_allow_mismatch)) {                                                          \
                    continue;                                                                                \
                }                                                                                            \
                TORCH_CHECK(                                                                                 \
                    (tensor1).size(i) == (tensor2).size(i),                                                  \
                    #tensor1 " and " #tensor2 " must have the same size in dimension " + std::to_string(i)); \
            }                                                                                                \
        }                                                                                                    \
    }

#define CHECK_NUM_DIMS(tensor, num_dims)                                                   \
    {                                                                                      \
        /* If the tensor is empty, the number of dimensions is not relevant */             \
        if (!((tensor).numel() == 0)) {                                                    \
            TORCH_CHECK((tensor).dim() == (num_dims),                                      \
                        #tensor " must have " + std::to_string(num_dims) + " dimensions"); \
        }                                                                                  \
    }

#define CHECK_NUM_DIMS_AT_LEAST(tensor, num_dims)                                                   \
    {                                                                                               \
        /* If the tensor is empty, the number of dimensions is not relevant */                      \
        if (!((tensor).numel() == 0)) {                                                             \
            TORCH_CHECK((tensor).dim() >= (num_dims),                                               \
                        #tensor " must have at least " + std::to_string(num_dims) + " dimensions"); \
        }                                                                                           \
    }

static inline int64_t get_number_data_elements_per_index(const torch::Tensor& input_data,
                                                         int64_t num_batch_and_index_dims = 2) {
    const int64_t num_extra_dims_data = input_data.dim() - num_batch_and_index_dims;
    int64_t num_data_elements_per_index = 1;
    for (int64_t i = 0; i < num_extra_dims_data; ++i) {
        num_data_elements_per_index *= input_data.size(i + num_batch_and_index_dims);
    }
    return num_data_elements_per_index;
}

#endif  // BATCHING_HELPERS_CPP_IMPL_BATCHED_INDEXING_ACCESS_HELPERS_H