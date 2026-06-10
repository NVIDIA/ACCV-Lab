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

#include "draw_heatmap_cuda_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x);

void draw_heatmap_cuda(cudaStream_t stream, float* heatmap, const int* centers, const int* radii,
                       const int* heatmap_idxes, int height, int width, float diameter_to_sigma_factor,
                       float k_scale, int num_targets, int num_heatmaps) {
    int grid_dim = (num_targets + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    draw_heatmap_cuda_kernel<<<grid_dim, THREADS_PER_BLOCK, 0, stream>>>(
        heatmap, centers, radii, heatmap_idxes, height, width, diameter_to_sigma_factor, k_scale, num_targets,
        num_heatmaps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in draw_heatmap_cuda: %s\n", cudaGetErrorString(err));
    }
}

void draw_heatmap_batched_cuda(cudaStream_t stream, float* heatmap, const int* centers, const int* radii,
                               const int* nums_targets, int height, int width, int max_num_targets,
                               float diameter_to_sigma_factor, float k_scale, int batch_size,
                               const int max_num_classes = 0, const int* labels = nullptr) {
    dim3 grid_dim;
    grid_dim.x = batch_size + THREADS_PER_BLOCK_BATCHED.x - 1 / THREADS_PER_BLOCK_BATCHED.x;
    grid_dim.y = max_num_targets + THREADS_PER_BLOCK_BATCHED.y - 1 / THREADS_PER_BLOCK_BATCHED.y;
    grid_dim.z = 1;

    draw_heatmap_batched_cuda_kernel<<<grid_dim, THREADS_PER_BLOCK_BATCHED, 0, stream>>>(
        heatmap, centers, radii, nums_targets, height, width, max_num_targets, diameter_to_sigma_factor,
        k_scale, batch_size, max_num_classes, labels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in draw_heatmap_cuda: %s\n", cudaGetErrorString(err));
    }
}

void draw_heatmap_launcher(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                           const at::Tensor& heatmap_idxes, float diameter_to_sigma_factor, float k_scale) {
    at::DeviceGuard guard(heatmap.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    CHECK_INPUT(heatmap);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    CHECK_INPUT(heatmap_idxes);

    TORCH_CHECK(centers.size(0) == radii.size(0), "centers and radii must have the same size at dim0");
    TORCH_CHECK(centers.size(0) == heatmap_idxes.size(0),
                "centers and heatmap_idxes must have the same size at dim0");
    TORCH_CHECK(heatmap.dim() == 3, "heatmap must be of shape [num_heatmaps, height, width]");
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 2, "centers must be of shape [num_targets, 2]");

    const int num_targets = centers.size(0);
    const int num_heatmaps = heatmap.size(0);
    const int height = heatmap.size(1);
    const int width = heatmap.size(2);

    AT_DISPATCH_FLOATING_TYPES(
        heatmap.scalar_type(), "draw_heatmap_cuda", ([&] {
            draw_heatmap_cuda(at::cuda::getCurrentCUDAStream(), heatmap.data_ptr<float>(),
                              centers.data_ptr<int>(), radii.data_ptr<int>(), heatmap_idxes.data_ptr<int>(),
                              height, width, diameter_to_sigma_factor, k_scale, num_targets, num_heatmaps);
        }));
}

void draw_heatmap_batched_launcher(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                                   const at::Tensor& nums_targets, float diameter_to_sigma_factor,
                                   float k_scale) {
    at::DeviceGuard guard(heatmap.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    CHECK_INPUT(heatmap);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    CHECK_INPUT(nums_targets);

    const int batch_size = heatmap.size(0);
    const int num_targets = radii.size(1);
    TORCH_CHECK(
        batch_size == radii.size(0) && batch_size == centers.size(0) && batch_size == nums_targets.size(0),
        "batch_size (dim 0) need to be the same for all inputs");
    TORCH_CHECK(num_targets == centers.size(1),
                "maximum number of targets (dim 1) need to be the same centers and radii");
    TORCH_CHECK(heatmap.dim() == 3, "heatmap must be of shape [batch_size, height, width]");
    TORCH_CHECK(centers.dim() == 3 && centers.size(2) == 2,
                "centers must be of shape [batch_size, num_targets, 2]");
    TORCH_CHECK(radii.dim() == 2, "radii must be of shape [batch_size, num_targets]");

    const int height = heatmap.size(1);
    const int width = heatmap.size(2);

    AT_DISPATCH_FLOATING_TYPES(heatmap.scalar_type(), "draw_heatmap_cuda_batched", ([&] {
                                   draw_heatmap_batched_cuda(
                                       at::cuda::getCurrentCUDAStream(), heatmap.data_ptr<float>(),
                                       centers.data_ptr<int>(), radii.data_ptr<int>(),
                                       nums_targets.data_ptr<int>(), height, width, num_targets,
                                       diameter_to_sigma_factor, k_scale, batch_size);
                               }));
}

void draw_heatmap_batched_classwise_launcher(at::Tensor& heatmap, const at::Tensor& centers,
                                             const at::Tensor& radii, const at::Tensor& nums_targets,
                                             const at::Tensor& labels, float diameter_to_sigma_factor,
                                             float k_scale) {
    at::DeviceGuard guard(heatmap.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    CHECK_INPUT(heatmap);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    CHECK_INPUT(nums_targets);
    CHECK_INPUT(labels);

    const int batch_size = heatmap.size(0);
    const int num_targets = radii.size(1);
    TORCH_CHECK(
        batch_size == radii.size(0) && batch_size == centers.size(0) && batch_size == nums_targets.size(0),
        "batch_size (dim 0) need to be the same for all inputs");
    TORCH_CHECK(num_targets == centers.size(1),
                "maximum number of targets (dim 1) need to be the same centers and radii");
    TORCH_CHECK(heatmap.dim() == 4, "heatmap must be of shape [batch_size, max_num_classes, height, width]");
    TORCH_CHECK(centers.dim() == 3 && centers.size(2) == 2,
                "centers must be of shape [batch_size, num_targets, 2]");
    TORCH_CHECK(radii.dim() == 2, "radii must be of shape [batch_size, num_targets]");

    const int height = heatmap.size(2);
    const int width = heatmap.size(3);
    const int max_num_classes = heatmap.size(1);
    // Validate labels tensor shape and range before launching the kernel
    TORCH_CHECK(labels.dim() == 2, "labels must be of shape [batch_size, radii.size(1)]");
    TORCH_CHECK(labels.size(0) == batch_size && labels.size(1) == num_targets,
                "labels shape must be [batch_size, radii.size(1)]");
    AT_DISPATCH_FLOATING_TYPES(
        heatmap.scalar_type(), "draw_heatmap_cuda_batched", ([&] {
            draw_heatmap_batched_cuda(
                at::cuda::getCurrentCUDAStream(), heatmap.data_ptr<float>(), centers.data_ptr<int>(),
                radii.data_ptr<int>(), nums_targets.data_ptr<int>(), height, width, num_targets,
                diameter_to_sigma_factor, k_scale, batch_size, max_num_classes, labels.data_ptr<int>());
        }));
}