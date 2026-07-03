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
#include "nvEncodeAPI.h"
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
// using namespace chrono;

namespace py = pybind11;

/**
 * @brief providing seek functionality within demuxer.
 */

#define ThrowOnCudaError_STRINGIFY(s) ThrowOnCudaError_STRINGIFY_(s)
#define ThrowOnCudaError_STRINGIFY_(s) #s
#define ThrowOnCudaError(expr)                                                       \
    {                                                                                \
        auto res = (expr);                                                           \
        if (CUDA_SUCCESS != res) {                                                   \
            std::stringstream ss;                                                    \
            ss << __FILE__ << ":";                                                   \
            ss << __LINE__ << std::endl;                                             \
            const char* errName = nullptr;                                           \
            if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {                     \
                ss << "CUDA error with code " << res << std::endl;                   \
            } else {                                                                 \
                ss << "CUDA error: " << errName << std::endl;                        \
            }                                                                        \
            const char* errDesc = nullptr;                                           \
            cuGetErrorString(res, &errDesc);                                         \
            if (!errDesc) {                                                          \
                ss << "No error string available" << std::endl;                      \
            } else {                                                                 \
                ss << errDesc << std::endl;                                          \
            }                                                                        \
            ss << "while executing: " ThrowOnCudaError_STRINGIFY(expr) << std::endl; \
            throw std::runtime_error(ss.str());                                      \
        }                                                                            \
    }

namespace {

class CuCtxGuard {
    CUcontext m_ctx;

   public:
    CuCtxGuard(CUcontext ctx) : m_ctx(ctx) { cuCtxPushCurrent_v2(ctx); }

    ~CuCtxGuard() { cuCtxPopCurrent(&m_ctx); }
};
}  // namespace

#define ENUM_VALUE_STRINGIFY(s) ENUM_VALUE_STRINGIFY_(s)
#define ENUM_VALUE_STRINGIFY_(s) #s
#define ENUM_VALUE(prefix, s) value(ENUM_VALUE_STRINGIFY(s), prefix##_##s)
#define DEF_CONSTANT(s) attr(ENUM_VALUE_STRINGIFY(s)) = py::cast(s)
#define DEF_READWRITE(type, s) def_readwrite(ENUM_VALUE_STRINGIFY(s), &type::s)

enum Pixel_Format {
    Pixel_Format_UNDEFINED = 0,
    Pixel_Format_NV12 = 3,
    Pixel_Format_YUV444 = 4,
    Pixel_Format_P016 = 5,
    Pixel_Format_YUV444_16Bit = 6

};

struct CAIMemoryView {
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    std::string typestr;
    CUstream stream = nullptr;
    CUdeviceptr data;
    bool readOnly;

    CAIMemoryView(const std::vector<size_t>& _shape, const std::vector<size_t>& _stride,
                  const std::string& _typeStr, size_t _streamid, CUdeviceptr _data, bool _readOnly) {
        shape = _shape;
        stride = _stride;
        typestr = _typeStr;
        data = _data;
        readOnly = _readOnly;
        stream = reinterpret_cast<CUstream>(_streamid);
    }
    CAIMemoryView() {
        shape = {0};
        stride = {1};
        typestr = "|u1";
        data = reinterpret_cast<CUdeviceptr>(nullptr);
        readOnly = true;
        stream = (CUstream)2;
    }
    static void Export(py::module& m) {
        py::class_<CAIMemoryView, std::shared_ptr<CAIMemoryView>>(m, "CAIMemoryView", py::module_local())
            .def(py::init<std::vector<size_t>, std::vector<size_t>, std::string, size_t, CUdeviceptr, bool>())
            .def_readonly("shape", &CAIMemoryView::shape)
            .def_readonly("stride", &CAIMemoryView::stride)
            .def_readonly("dataptr", &CAIMemoryView::data)
            .def("__repr__",
                 [](std::shared_ptr<CAIMemoryView>& self) {
                     std::stringstream ss;
                     ss << "<CAIMemoryView ";
                     ss << py::str(py::cast(self->shape));
                     ss << ">";
                     return ss.str();
                 })
            .def_readonly("data", &CAIMemoryView::data)
            .def_property_readonly("__cuda_array_interface__", [](std::shared_ptr<CAIMemoryView>& self) {
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

struct DecodedFrame {
    int64_t timestamp;
    std::vector<CAIMemoryView> views;
    Pixel_Format format;
    std::shared_ptr<ExternalBuffer> extBuf;
    DecodedFrame() { extBuf = std::make_shared<ExternalBuffer>(); }
    static void Export(py::module& m) {
        py::class_<DecodedFrame, std::shared_ptr<DecodedFrame>>(m, "DecodedFrame", py::module_local())
            .def_readonly("timestamp", &DecodedFrame::timestamp)
            .def_readonly("format", &DecodedFrame::format)
            .def("__repr__",
                 [](std::shared_ptr<DecodedFrame>& self) {
                     std::stringstream ss;
                     ss << "<DecodedFrame [";
                     ss << "timestamp=" << self->timestamp;
                     ss << ", format=" << py::str(py::cast(self->format));
                     ss << ", " << py::str(py::cast(self->views));
                     ss << "]>";
                     return ss.str();
                 })
            .def(
                "framesize",
                [](std::shared_ptr<DecodedFrame>& self) {
                    int height = self->views.at(0).shape.at(0);
                    int width = self->views.at(0).shape.at(1);
                    int framesize = width * height * 1.5;
                    switch (self->format) {
                        case Pixel_Format_NV12:
                            break;
                        case Pixel_Format_P016:
                            framesize = width * height * 3;
                            break;
                        case Pixel_Format_YUV444:
                            framesize = width * height * 3;
                            break;
                        case Pixel_Format_YUV444_16Bit:
                            framesize = width * height * 6;
                            break;
                        default:
                            break;
                    }
                    return framesize;
                },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
            .def(
                "cuda", [](std::shared_ptr<DecodedFrame>& self) { return self->views; },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
            .def(
                "nvcv_image",
                [](std::shared_ptr<DecodedFrame>& self) {
                    switch (self->format) {
                        case Pixel_Format_NV12: {
                            size_t width = self->views.at(0).shape[1];
                            size_t height = self->views.at(0).shape[0] * 1.5;
                            CUdeviceptr data = self->views.at(0).data;
                            CUstream stream = self->views.at(0).stream;
                            self->views.clear();
                            self->views.push_back(
                                CAIMemoryView{{height, width, 1},
                                              {width, 2, 1},
                                              "|u1",
                                              reinterpret_cast<size_t>(stream),
                                              (data),
                                              false});  // hack for cvcuda tensor represenation
                        } break;
                        case Pixel_Format_YUV444: {
                            size_t width = self->views.at(0).shape[1];
                            size_t height = self->views.at(0).shape[0] * 3;
                            CUdeviceptr data = self->views.at(0).data;
                            CUstream stream = self->views.at(0).stream;
                            self->views.clear();
                            self->views.push_back(
                                CAIMemoryView{{height, width, 1},
                                              {width, 3, 1},
                                              "|u1",
                                              reinterpret_cast<size_t>(stream),
                                              (data),
                                              false});  // hack for cvcuda tensor represenation
                        } break;
                        default:
                            throw std::invalid_argument("only nv12 and yuv444 supported as of now");
                            break;
                    }
                    return self->views;
                },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")

            // DL Pack Tensor
            .def_property_readonly(
                "shape", [](std::shared_ptr<DecodedFrame>& self) { return self->extBuf->shape(); },
                "Get the shape of the buffer as an array")
            .def_property_readonly(
                "strides", [](std::shared_ptr<DecodedFrame>& self) { return self->extBuf->strides(); },
                "Get the strides of the buffer")
            .def_property_readonly(
                "dtype", [](std::shared_ptr<DecodedFrame>& self) { return self->extBuf->dtype(); },
                "Get the data type of the buffer")
            .def(
                "__dlpack__",
                [](std::shared_ptr<DecodedFrame>& self, py::object stream) {
                    return self->extBuf->dlpack(stream);
                },
                py::arg("stream") = NULL, "Export the buffer as a DLPack tensor")
            .def(
                "__dlpack_device__",
                [](std::shared_ptr<DecodedFrame>& self) {
                    // DLDevice ctx;
                    // ctx.device_type = DLDeviceType::kDLCUDA;
                    // ctx.device_id = 0;
                    return py::make_tuple(py::int_(static_cast<int>(DLDeviceType::kDLCUDA)),
                                          py::int_(static_cast<int>(0)));
                },
                "Get the device associated with the buffer")
            .def(
                "GetPtrToPlane",

                [](std::shared_ptr<DecodedFrame>& self, int planeIdx) { return self->views[planeIdx].data; },
                R"pbdoc(
            return pointer to base address for plane index
            :param planeIdx : index to the plane
            )pbdoc");
        // TODO add __iter__ interface on DecodedFrame
    }
};

CAIMemoryView coerceToCudaArrayView(py::object cuda_array, NV_ENC_BUFFER_FORMAT bufferFormat, size_t width,
                                    size_t height, int planeIdx = 0);
