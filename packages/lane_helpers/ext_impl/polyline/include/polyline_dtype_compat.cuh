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

#ifndef LANE_HELPERS_POLYLINE_DTYPE_COMPAT_CUH
#define LANE_HELPERS_POLYLINE_DTYPE_COMPAT_CUH

#include <cmath>

// CUDA provides native __half/__nv_bfloat16 comparison intrinsics and shuffle
// overloads, while c10 low-precision wrappers add extra conversion paths,
// leading to compilation errors. The CUDA-only specializations below route c10 values
// through the native CUDA operations where available; only scalar math such as sqrt
// intentionally computes via float. Keeping these variants CUDA-only keeps CPU builds
// free of these types.
#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#define POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE inline
#endif

namespace polyline {

#ifdef __CUDACC__
template <typename dtype>
__device__ __forceinline__ dtype shfl_xor_sync_compat(unsigned mask, dtype val, int laneMask) {
    return __shfl_xor_sync(mask, val, laneMask);
}

template <>
__device__ __forceinline__ c10::Half shfl_xor_sync_compat(unsigned mask, c10::Half val, int laneMask) {
    return c10::Half(__shfl_xor_sync(mask, static_cast<__half>(val), laneMask));
}

template <>
__device__ __forceinline__ c10::BFloat16 shfl_xor_sync_compat(unsigned mask, c10::BFloat16 val,
                                                              int laneMask) {
    return c10::BFloat16(__shfl_xor_sync(mask, static_cast<__nv_bfloat16>(val), laneMask));
}
#endif

template <typename dtype>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_lt(dtype lhs, dtype rhs) {
    return lhs < rhs;
}

template <typename dtype>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_gt(dtype lhs, dtype rhs) {
    return lhs > rhs;
}

template <typename dtype>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_ge(dtype lhs, dtype rhs) {
    return lhs >= rhs;
}

#ifdef __CUDACC__
template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_lt<c10::Half>(c10::Half lhs, c10::Half rhs) {
    return __hlt(static_cast<__half>(lhs), static_cast<__half>(rhs));
}

template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_gt<c10::Half>(c10::Half lhs, c10::Half rhs) {
    return __hgt(static_cast<__half>(lhs), static_cast<__half>(rhs));
}

template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_ge<c10::Half>(c10::Half lhs, c10::Half rhs) {
    return __hge(static_cast<__half>(lhs), static_cast<__half>(rhs));
}

template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_lt<c10::BFloat16>(c10::BFloat16 lhs,
                                                                               c10::BFloat16 rhs) {
    return __hlt(static_cast<__nv_bfloat16>(lhs), static_cast<__nv_bfloat16>(rhs));
}

template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_gt<c10::BFloat16>(c10::BFloat16 lhs,
                                                                               c10::BFloat16 rhs) {
    return __hgt(static_cast<__nv_bfloat16>(lhs), static_cast<__nv_bfloat16>(rhs));
}

template <>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE bool polyline_value_ge<c10::BFloat16>(c10::BFloat16 lhs,
                                                                               c10::BFloat16 rhs) {
    return __hge(static_cast<__nv_bfloat16>(lhs), static_cast<__nv_bfloat16>(rhs));
}
#endif

template <typename dtype>
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE dtype polyline_sqrt(dtype value) {
    return sqrt(value);
}

#ifdef __CUDACC__
POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE c10::Half polyline_sqrt(c10::Half value) {
    return static_cast<c10::Half>(sqrtf(static_cast<float>(value)));
}

POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE c10::BFloat16 polyline_sqrt(c10::BFloat16 value) {
    return static_cast<c10::BFloat16>(sqrtf(static_cast<float>(value)));
}
#endif

}  // namespace polyline

#undef POLYLINE_DTYPE_COMPAT_HOST_DEVICE_INLINE

#endif  // LANE_HELPERS_POLYLINE_DTYPE_COMPAT_CUH
