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

#pragma once
#include "ExternalBuffer.hpp"
#include "NvCodecUtils.h"
#include "PyCAIMemoryView.hpp"
#include "nvEncodeAPI.h"
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

class RGBFrame {
   public:
    std::tuple<size_t, size_t, size_t> shape;
    std::tuple<size_t, size_t, size_t> stride;
    std::string typestr;
    CUstream stream = nullptr;
    CUdeviceptr data;
    bool readOnly;
    bool isBGR;

    RGBFrame(const std::vector<size_t>& _shape, const std::vector<size_t>& _stride,
             const std::string& _typeStr, size_t _streamid, CUdeviceptr _data, bool _readOnly, bool _isBRG);

    RGBFrame(const CAIMemoryView& to_convert, bool _isBRG);

    RGBFrame();

    void release_data();

    bool is_of_size(size_t height, size_t width);
    static void Export(py::module& m) {
        py::class_<RGBFrame, std::shared_ptr<RGBFrame>>(m, "RGBFrame")
            .def(py::init<std::vector<size_t>, std::vector<size_t>, std::string, size_t, CUdeviceptr, bool,
                          bool>())
            .def_readonly("shape", &RGBFrame::shape)
            .def_readonly("stride", &RGBFrame::stride)
            .def_readonly("dataptr", &RGBFrame::data)
            .def_readonly("isBGR", &RGBFrame::isBGR)
            .def("__repr__",
                 [](std::shared_ptr<RGBFrame>& self) {
                     std::stringstream ss;
                     ss << "<RGBFrame ";
                     ss << py::str(py::cast(self->shape));
                     ss << ">";
                     return ss.str();
                 })
            .def_readonly("data", &RGBFrame::data)
            .def_property_readonly("__cuda_array_interface__", [](std::shared_ptr<RGBFrame>& self) {
                py::dict dict;
                dict["version"] = 3;
                dict["shape"] = self->shape;
                dict["strides"] = self->stride;
                dict["typestr"] = self->typestr;
                // stream field per CAI v3 spec:
                // https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html#python-interface-specification
                //   None  → data already ready, no sync needed
                //   1 / 2 → legacy / per-thread default stream
                //   other → cudaStream_t handle cast to Python int; consumer must sync before use
                dict["stream"] =
                    self->stream ? py::cast(reinterpret_cast<size_t>(self->stream)) : py::cast(2);
                dict["data"] = std::make_pair(self->data, false);
                dict["gpuIdx"] = 0;  // TODO
                return dict;
            });
    }
};
