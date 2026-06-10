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

#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "polyline.cuh"
#include "helper_macros.cuh"

//#define PROFILE_AND_SYNC

#ifdef PROFILE_AND_SYNC
#include <nvtx3/nvToolsExt.h>
#endif

namespace polyline {

#define CHECK_DEVICE(x) check_device(x, #x)
#define CHECK_CONTIGUOUS(x) check_contiguous(x, #x)
#define CHECK_TYPE(x) check_type(x, #x)
#define CHECK_INPUT(x)   \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_TYPE(x);
inline void check_device(const at::Tensor& tensor, const char* description) {
    TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(), description, " must be on CPU or CUDA");
}

inline void check_contiguous(const at::Tensor& tensor, const char* description) {
    TORCH_CHECK(tensor.is_contiguous(), description, " must be contiguous");
}

inline void check_type(const at::Tensor& tensor, const char* description) {
    if (tensor.is_cuda()) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64 ||
                        tensor.scalar_type() == torch::kFloat16 || tensor.scalar_type() == torch::kBFloat16,
                    description, " must have dtype float16, float32, float64, or bfloat16 on CUDA");
    } else {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64,
                    description, " must have dtype float32 or float64 on CPU");
    }
}

inline void check_same_device(const at::Tensor& lhs, const at::Tensor& rhs, const char* message) {
    TORCH_CHECK(lhs.device() == rhs.device(), message);
}

inline void check_sample_size_type(const at::Tensor& sample_sizes, const char* description) {
    TORCH_CHECK(sample_sizes.scalar_type() == at::kInt || sample_sizes.scalar_type() == at::kLong,
                description, " must have dtype int32 or int64");
}

inline void check_sample_sizes(const at::Tensor& sample_sizes, int max_size, const char* description) {
    if (sample_sizes.numel() == 0) {
        return;
    }
    TORCH_CHECK(
        !torch::any(sample_sizes < 0).item<bool>() && !torch::any(sample_sizes > max_size).item<bool>(),
        description, " values must be in [0, ", max_size, "]");
}

at::Tensor make_external_distance_buffer(size_t size_elems, const at::TensorOptions& options) {
    // Keep external CUDA scratch memory owned by PyTorch's stream-aware allocator.
    // A raw cudaFree here can race with the asynchronous custom kernel that uses this buffer.

    // Return an empty tensor if no external distance buffer is needed.
    if (size_elems == 0) {
        return at::Tensor();
    }

    // Check that the size is not too large to allocate as a tensor.
    TORCH_CHECK(size_elems <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "external polyline distance buffer is too large to allocate as a tensor");

    // Allocate the buffer and return it.
    at::Tensor buffer = at::empty({static_cast<int64_t>(size_elems)}, options);
    return buffer;
}

at::Tensor polyline_interpolation(at::Tensor points, at::Tensor distances, bool relative_distances) {
    CHECK_DEVICE(points);
    CHECK_DEVICE(distances);
    CHECK_TYPE(points);
    CHECK_TYPE(distances);
    TORCH_CHECK(points.ndimension() == 3, "points must have shape (batch, num_points, num_dims)");
    TORCH_CHECK(distances.ndimension() == 2, "distances must have shape (batch, num_distances)");
    TORCH_CHECK(points.size(0) == distances.size(0),
                "points and distances must contain the same number of polylines");
    TORCH_CHECK(points.scalar_type() == distances.scalar_type(),
                "points and distances must have the same dtype");
    check_same_device(points, distances, "points and distances must be on the same device");

    const int num_samples = points.size(0);
    const int num_points = points.size(1);
    const int num_distances = distances.size(1);
    const int num_dims = points.size(2);
    // Result has shape (batch, num_distances, point_dim) and otherwise
    // matches `distances` (device, dtype).
    auto res = at::empty({num_samples, num_distances, num_dims}, distances.options());
    if (num_distances == 0) {
        return res;
    }
    const at::Tensor points_contiguous = points.contiguous();
    const at::Tensor distances_contiguous = distances.contiguous();

    if (points.is_cuda()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, points.scalar_type(), "polyline_interpolation", [&] {
                const int device = points.get_device();
                c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
                const auto stream = at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device));
                at::cuda::CUDAStreamGuard stream_guard(stream);
                const auto cfg = make_polyline_launch_config<scalar_t>(num_points, num_samples, device);
                // Allocate under the same stream used for the kernel launch so the caching allocator
                // does not recycle this temporary scratch buffer before queued kernel work consumes it.
                const at::Tensor distance_buffer_ext =
                    make_external_distance_buffer(cfg.distance_buffer_ext_size_elems, points.options());
                scalar_t* distance_buffer_ext_ptr =
                    distance_buffer_ext.defined() ? distance_buffer_ext.data_ptr<scalar_t>() : nullptr;
                polyline_interpolation<scalar_t>(points_contiguous.data_ptr<scalar_t>(), num_points, num_dims,
                                                 distances_contiguous.data_ptr<scalar_t>(), num_distances,
                                                 res.data_ptr<scalar_t>(), num_samples, relative_distances,
                                                 device, cfg, distance_buffer_ext_ptr, stream.stream());
                CUDA_CHECK_LAST();
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "polyline_interpolation_cpu", [&] {
            polyline_interpolation_cpu<scalar_t>(points_contiguous.data_ptr<scalar_t>(), num_points, num_dims,
                                                 distances_contiguous.data_ptr<scalar_t>(), num_distances,
                                                 res.data_ptr<scalar_t>(), num_samples, relative_distances);
        });
    }

    return res;
}

at::Tensor polyline_lengths(at::Tensor points) {
    CHECK_DEVICE(points);
    CHECK_TYPE(points);
    TORCH_CHECK(points.ndimension() == 3, "points must have shape (batch, num_points, num_dims)");

    const int num_samples = points.size(0);
    const int num_points = points.size(1);
    const int num_dims = points.size(2);
    auto res = at::empty({num_samples}, points.options());
    const at::Tensor points_contiguous = points.contiguous();

    if (points.is_cuda()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, points.scalar_type(), "polyline_lengths", [&] {
                cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                polyline_lengths<scalar_t>(points_contiguous.data_ptr<scalar_t>(),  // points
                                           num_points,                              // num_points
                                           num_dims,                                // num_dims
                                           res.data_ptr<scalar_t>(),                // lengths
                                           num_samples,                             // num_samples
                                           stream                                   // stream
                );
                CUDA_CHECK_LAST();
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "polyline_lengths_cpu", [&] {
            polyline_lengths_cpu<scalar_t>(points_contiguous.data_ptr<scalar_t>(),  // points
                                           num_points,                              // num_points
                                           num_dims,                                // num_dims
                                           res.data_ptr<scalar_t>(),                // lengths
                                           num_samples                              // num_samples
            );
        });
    }

    return res;
}

at::Tensor polyline_interpolation_var_size_batch(at::Tensor points, at::Tensor distances,
                                                 at::Tensor sample_sizes_points,
                                                 at::Tensor sample_sizes_distances_to_sample,
                                                 bool relative_distances) {
    CHECK_DEVICE(points);
    CHECK_DEVICE(distances);
    check_device(sample_sizes_points, "points.sample_sizes");
    check_device(sample_sizes_distances_to_sample, "distances.sample_sizes");
    CHECK_TYPE(points);
    CHECK_TYPE(distances);
    check_sample_size_type(sample_sizes_points, "points.sample_sizes");
    check_sample_size_type(sample_sizes_distances_to_sample, "distances.sample_sizes");

    TORCH_CHECK(points.ndimension() == 3, "points must have shape (batch, max_num_points, num_dims)");
    TORCH_CHECK(distances.ndimension() == 2, "distances must have shape (batch, max_num_distances)");
    TORCH_CHECK(points.size(0) == distances.size(0),
                "points and distances must contain the same number of polylines");
    TORCH_CHECK(points.scalar_type() == distances.scalar_type(),
                "points and distances must have the same dtype");
    check_same_device(points, distances, "points and distances must be on the same device");
    TORCH_CHECK(sample_sizes_points.scalar_type() == sample_sizes_distances_to_sample.scalar_type(),
                "points.sample_sizes and distances.sample_sizes must have the same dtype "
                "(both int32 or both int64)");
    check_same_device(sample_sizes_points, points,
                      "points.sample_sizes must be on the same device as points");
    check_same_device(sample_sizes_distances_to_sample, distances,
                      "distances.sample_sizes must be on the same device as distances");
    TORCH_CHECK(sample_sizes_points.ndimension() == 1, "points.sample_sizes must be a 1D tensor");
    TORCH_CHECK(sample_sizes_distances_to_sample.ndimension() == 1,
                "distances.sample_sizes must be a 1D tensor");

    const int num_samples = points.size(0);
    const int max_num_points = points.size(1);
    const int max_num_distances = distances.size(1);
    const int num_dims = points.size(2);
    // Result has shape (batch, num_distances, point_dim) and otherwise
    // matches `distances` (device, dtype).
    auto res = at::empty({num_samples, max_num_distances, num_dims}, distances.options());

    TORCH_CHECK(sample_sizes_points.size(0) == num_samples,
                "points.sample_sizes must contain one count per polyline in points");
    TORCH_CHECK(sample_sizes_distances_to_sample.size(0) == num_samples,
                "distances.sample_sizes must contain one count per polyline in distances");
    check_sample_sizes(sample_sizes_points, max_num_points, "points.sample_sizes");
    check_sample_sizes(sample_sizes_distances_to_sample, max_num_distances, "distances.sample_sizes");
    if (max_num_distances == 0) {
        return res;
    }

    const at::Tensor points_contiguous = points.contiguous();
    const at::Tensor distances_contiguous = distances.contiguous();
    const at::Tensor sample_sizes_points_contiguous = sample_sizes_points.contiguous();
    const at::Tensor sample_sizes_distances_to_sample_contiguous =
        sample_sizes_distances_to_sample.contiguous();

    auto launch = [&](auto sample_size_type_tag) {
        using sample_size_t = decltype(sample_size_type_tag);
        if (points.is_cuda()) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, points.scalar_type(), "polyline_interpolation_var_size_batch", [&] {
                    const int device = points.get_device();
                    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
                    const auto stream = at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device));
                    at::cuda::CUDAStreamGuard stream_guard(stream);
                    const auto cfg =
                        make_polyline_launch_config<scalar_t>(max_num_points, num_samples, device);
                    // Allocate under the same stream used for the kernel launch so the caching allocator
                    // does not recycle this temporary scratch buffer before queued kernel work consumes it.
                    const at::Tensor distance_buffer_ext =
                        make_external_distance_buffer(cfg.distance_buffer_ext_size_elems, points.options());
                    scalar_t* distance_buffer_ext_ptr =
                        distance_buffer_ext.defined() ? distance_buffer_ext.data_ptr<scalar_t>() : nullptr;
                    polyline_interpolation_var_size_batch<scalar_t, sample_size_t>(
                        points_contiguous.data_ptr<scalar_t>(),                    // points
                        max_num_points,                                            // max_num_points
                        num_dims,                                                  // num_dims
                        distances_contiguous.data_ptr<scalar_t>(),                 // distances
                        max_num_distances,                                         // num_distances
                        res.data_ptr<scalar_t>(),                                  // result_points
                        num_samples,                                               // num_samples
                        sample_sizes_points_contiguous.data_ptr<sample_size_t>(),  // sample_sizes_points
                        sample_sizes_distances_to_sample_contiguous
                            .data_ptr<sample_size_t>(),  // sample sizes distances
                        relative_distances,              // relative_distances
                        device,                          // device
                        cfg,                             // launch config
                        distance_buffer_ext_ptr,         // distance_buffer_ext
                        stream.stream()                  // stream
                    );
                    CUDA_CHECK_LAST();
                });
        } else {
            AT_DISPATCH_FLOATING_TYPES(
                points.scalar_type(), "polyline_interpolation_var_size_batch_cpu", [&] {
                    polyline_interpolation_var_size_batch_cpu<scalar_t, sample_size_t>(
                        points_contiguous.data_ptr<scalar_t>(),                    // points
                        max_num_points,                                            // max_num_points
                        num_dims,                                                  // num_dims
                        distances_contiguous.data_ptr<scalar_t>(),                 // distances
                        max_num_distances,                                         // num_distances
                        res.data_ptr<scalar_t>(),                                  // result_points
                        num_samples,                                               // num_samples
                        sample_sizes_points_contiguous.data_ptr<sample_size_t>(),  // sample_sizes_points
                        sample_sizes_distances_to_sample_contiguous
                            .data_ptr<sample_size_t>(),  // sample sizes distances
                        relative_distances               // relative_distances
                    );
                });
        }
    };
    if (sample_sizes_points.scalar_type() == at::kInt) {
        launch(int32_t{});
    } else {
        launch(int64_t{});
    }

    return res;
}

at::Tensor polyline_lengths_var_size_batch(at::Tensor points, at::Tensor sample_sizes_points) {
    CHECK_DEVICE(points);
    check_device(sample_sizes_points, "points.sample_sizes");
    CHECK_TYPE(points);
    check_sample_size_type(sample_sizes_points, "points.sample_sizes");

    TORCH_CHECK(points.ndimension() == 3, "points must have shape (batch, max_num_points, num_dims)");
    TORCH_CHECK(sample_sizes_points.ndimension() == 1, "points.sample_sizes must be a 1D tensor");
    check_same_device(sample_sizes_points, points,
                      "points.sample_sizes must be on the same device as points");

    const int num_samples = points.size(0);
    const int max_num_points = points.size(1);
    const int num_dims = points.size(2);
    auto res = at::empty({num_samples}, points.options());

    TORCH_CHECK(sample_sizes_points.size(0) == num_samples,
                "points.sample_sizes must contain one count per polyline in points");
    check_sample_sizes(sample_sizes_points, max_num_points, "points.sample_sizes");

    const at::Tensor points_contiguous = points.contiguous();
    const at::Tensor sample_sizes_points_contiguous = sample_sizes_points.contiguous();

    auto launch = [&](auto sample_size_type_tag) {
        using sample_size_t = decltype(sample_size_type_tag);
        if (points.is_cuda()) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, points.scalar_type(), "polyline_lengths_var_size_batch", [&] {
                    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                    polyline_lengths_var_size_batch<scalar_t, sample_size_t>(
                        points_contiguous.data_ptr<scalar_t>(),                    // points
                        max_num_points,                                            // max_num_points
                        num_dims,                                                  // num_dims
                        res.data_ptr<scalar_t>(),                                  // lengths
                        num_samples,                                               // num_samples
                        sample_sizes_points_contiguous.data_ptr<sample_size_t>(),  // sample_sizes_points
                        stream                                                     // stream
                    );
                    CUDA_CHECK_LAST();
                });
        } else {
            AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "polyline_lengths_var_size_batch_cpu", [&] {
                polyline_lengths_var_size_batch_cpu<scalar_t, sample_size_t>(
                    points_contiguous.data_ptr<scalar_t>(),                   // points
                    max_num_points,                                           // max_num_points
                    num_dims,                                                 // num_dims
                    res.data_ptr<scalar_t>(),                                 // lengths
                    num_samples,                                              // num_samples
                    sample_sizes_points_contiguous.data_ptr<sample_size_t>()  // sample_sizes_points
                );
            });
        }
    };
    if (sample_sizes_points.scalar_type() == at::kInt) {
        launch(int32_t{});
    } else {
        launch(int64_t{});
    }

    return res;
}

}  // namespace polyline

namespace py = pybind11;
using namespace polyline;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Lane helpers polyline interpolation bindings";
    m.def("polyline_interpolation", (at::Tensor(*)(at::Tensor, at::Tensor, bool)) & polyline_interpolation,
          py::arg("points"), py::arg("distances"), py::arg("relative") = false,
          "Interpolate points along polylines at given distances.");
    m.def("_polyline_lengths", (at::Tensor(*)(at::Tensor)) & polyline_lengths, py::arg("points"),
          "Internal tensor-only entry point for fixed-size polyline length computation.");
    m.def("_polyline_interpolation_var_size_batch",
          (at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool)) &
              polyline_interpolation_var_size_batch,
          py::arg("points"), py::arg("distances"), py::arg("sample_sizes_points"),
          py::arg("sample_sizes_distances_to_sample"), py::arg("relative") = false,
          "Internal tensor-only entry point for variable-length polyline interpolation.");
    m.def("_polyline_lengths_var_size_batch",
          (at::Tensor(*)(at::Tensor, at::Tensor)) & polyline_lengths_var_size_batch, py::arg("points"),
          py::arg("sample_sizes_points"),
          "Internal tensor-only entry point for variable-length polyline length computation.");
}