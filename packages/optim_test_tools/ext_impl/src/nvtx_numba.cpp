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
#include <string>

#include <pybind11/pybind11.h>

// NVTX v3 header-only implementation (includes dynamic injection loader).
// We rely on the repo-vendored NVTX headers (see ext_impl/CMakeLists.txt include path).
#include <nvtx3/nvToolsExt.h>

namespace py = pybind11;

namespace {

static inline uintptr_t nvtx_register_string_a(const char* s) noexcept {
    // If domain is NULL, the global domain is used.
    nvtxStringHandle_t h = nvtxDomainRegisterStringA(nullptr, s);
    return reinterpret_cast<uintptr_t>(h);
}

static inline void nvtx_range_push_registered(uintptr_t handle) noexcept {
    if (handle == 0) {
        return;
    }

    nvtxEventAttributes_t a{};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    a.message.registered = reinterpret_cast<nvtxStringHandle_t>(handle);
    (void)nvtxRangePushEx(&a);
}

static inline void nvtx_range_pop() noexcept { (void)nvtxRangePop(); }

}  // namespace

#if defined(_WIN32)
#define ACCVLAB_NVTX_NUMBA_EXPORT extern "C" __declspec(dllexport)
#else
#define ACCVLAB_NVTX_NUMBA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// These exported symbols are what Numba will call from within @njit (via LLVM symbol binding).
ACCVLAB_NVTX_NUMBA_EXPORT void accvlab_nvtx_range_push(std::uint64_t handle) noexcept {
    nvtx_range_push_registered(static_cast<uintptr_t>(handle));
}

ACCVLAB_NVTX_NUMBA_EXPORT void accvlab_nvtx_range_pop() noexcept { nvtx_range_pop(); }

static std::uint64_t py_register_string(const std::string& s) {
    return static_cast<std::uint64_t>(nvtx_register_string_a(s.c_str()));
}

static void py_range_push(std::uint64_t handle) noexcept {
    nvtx_range_push_registered(static_cast<uintptr_t>(handle));
}

static void py_range_pop() noexcept { nvtx_range_pop(); }

PYBIND11_MODULE(_nvtx_numba_ext, m) {
    m.doc() = "NVTX registered-string range push/pop for Numba @njit (CPU) code.";
    m.def("register_string", &py_register_string, py::arg("name"),
          "Register a string with NVTX and return an integer handle (0 if NVTX is inactive).");
    m.def("range_push", &py_range_push, py::arg("handle"),
          "Push an NVTX range using a previously-registered string handle.");
    m.def("range_pop", &py_range_pop, "Pop an NVTX range.");
}
