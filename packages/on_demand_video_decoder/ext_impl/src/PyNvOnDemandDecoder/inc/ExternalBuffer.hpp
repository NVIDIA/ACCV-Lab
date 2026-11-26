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

#ifndef EXTERNAL_BUFFER_HPP
#define EXTERNAL_BUFFER_HPP

#include "DLPackUtils.hpp"

#include <cuda.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class ExternalBuffer final : public std::enable_shared_from_this<ExternalBuffer> {
   public:
    static void Export(py::module& m);

    const DLTensor& dlTensor() const;

    py::tuple shape() const;
    py::tuple strides() const;
    std::string dtype() const;

    void* data() const;

    //bool load(PyObject *o);
    explicit ExternalBuffer(DLPackTensor&& dlTensor);

    ExternalBuffer() = default;
    // __dlpack__ implementation
    py::capsule dlpack(py::object stream) const;
    // __dlpack_device__ implementation
    py::tuple dlpackDevice() const;
    int LoadDLPack(std::vector<size_t> _shape, std::vector<size_t> _stride, std::string _typeStr,
                   size_t _streamid, CUdeviceptr _data, bool _readOnly);

   private:
    friend py::detail::type_caster<ExternalBuffer>;

    DLPackTensor m_dlTensor;
};

#endif  // EXTERNAL_BUFFER_HPP
