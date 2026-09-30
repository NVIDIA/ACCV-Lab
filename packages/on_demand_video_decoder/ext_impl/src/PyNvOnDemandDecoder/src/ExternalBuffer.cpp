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

#include "ExternalBuffer.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <functional>  // for std::multiplies

using namespace py::literals;

static void CheckValidCUDABuffer(const void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    //TBD
    //cudaPointerAttributes attrs = {};
    //cudaError_t           err   = cudaPointerGetAttributes(&attrs, ptr);
    //cudaGetLastError(); // reset the cuda error (if any)
    //if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered)
    //{
    //    throw std::runtime_error("Buffer is not CUDA-accessible");
    //}
}

//static std::string ToFormatString(const DLDataType &dtype)
//{
//    py::dtype dt = ToDType(ToNVCVDataType(dtype));
//    return dt.attr("str").cast<std::string>();
//}

ExternalBuffer::ExternalBuffer(DLPackTensor&& dlTensor) {
    if (!IsCudaAccessible(dlTensor->device.device_type)) {
        throw std::runtime_error("Only CUDA memory buffers can be wrapped");
    }

    if (dlTensor->data != nullptr) {
        CheckValidCUDABuffer(dlTensor->data);
    }

    m_dlTensor = std::move(dlTensor);
}

py::tuple ExternalBuffer::shape() const {
    py::tuple shape(m_dlTensor->ndim);
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = m_dlTensor->shape[i];
    }

    return shape;
}

py::tuple ExternalBuffer::strides() const {
    py::tuple strides(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] = m_dlTensor->strides[i];
    }

    return strides;
}

std::string ExternalBuffer::dtype() const {
    return std::string("|u1");
    //return (m_dlTensor->dtype);
}

void* ExternalBuffer::data() const { return m_dlTensor->data; }

py::capsule ExternalBuffer::dlpack(py::object stream) const {
    struct ManagerCtx {
        DLManagedTensor tensor;
        std::shared_ptr<const ExternalBuffer> extBuffer;
    };

    auto ctx = std::make_unique<ManagerCtx>();

    // Set up tensor deleter to delete the ManagerCtx
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter = [](DLManagedTensor* tensor) {
        auto* ctx = static_cast<ManagerCtx*>(tensor->manager_ctx);
        delete ctx;
    };

    // Copy tensor data
    ctx->tensor.dl_tensor = *m_dlTensor;

    // Manager context holds a reference to this External Buffer so that
    // GC doesn't delete this buffer while the dlpack tensor still refers to it.
    ctx->extBuffer = this->shared_from_this();

    // Creates the python capsule with the DLManagedTensor instance we're returning.
    py::capsule cap(&ctx->tensor, "dltensor", [](PyObject* ptr) {
        if (PyCapsule_IsValid(ptr, "dltensor")) {
            // If consumer didn't delete the tensor,
            if (auto* dlTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, "dltensor"))) {
                // Delete the tensor.
                if (dlTensor->deleter != nullptr) {
                    dlTensor->deleter(dlTensor);
                }
            }
        }
    });

    // Now that the capsule is created and the manager ctx was transfered to it,
    // we can release the unique_ptr.
    ctx.release();

    return cap;
}

py::tuple ExternalBuffer::dlpackDevice() const {
    return py::make_tuple(py::int_(static_cast<int>(m_dlTensor->device.device_type)),
                          py::int_(static_cast<int>(m_dlTensor->device.device_id)));
}

const DLTensor& ExternalBuffer::dlTensor() const { return *m_dlTensor; }

void ExternalBuffer::Export(py::module& m) {
    py::class_<ExternalBuffer, std::shared_ptr<ExternalBuffer>>(m, "ExternalBuffer", py::dynamic_attr())
        .def_property_readonly("shape", &ExternalBuffer::shape, "Get the shape of the buffer as an array")
        .def_property_readonly("strides", &ExternalBuffer::strides, "Get the strides of the buffer")
        .def_property_readonly("dtype", &ExternalBuffer::dtype, "Get the data type of the buffer")
        .def("__dlpack__", &ExternalBuffer::dlpack, "stream"_a = 1, "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", &ExternalBuffer::dlpackDevice, "Get the device associated with the buffer");
}

int ExternalBuffer::LoadDLPack(std::vector<size_t> _shape, std::vector<size_t> _stride, std::string _typeStr,
                               size_t _streamid, CUdeviceptr _data, bool _readOnly) {
    if (_typeStr != "|u1" && _typeStr != "B")  // TODO: can also be other letters
    {
        throw std::runtime_error("Could not create DL Pack tensor! Invalid typstr: " + _typeStr);
    }
    if (_shape.size() != _stride.size()) {
        throw std::invalid_argument("Shape and strides must have the same rank");
    }

    void* ptr = reinterpret_cast<void*>(_data);
    CheckValidCUDABuffer(ptr);

    DLManagedTensor managedTensor{};
    DLTensor& tensor = managedTensor.dl_tensor;
    tensor.data = ptr;
    tensor.byte_offset = 0;
    // TODO: infer the device type and device ID from the memory buffer.
    tensor.device.device_type = kDLCUDA;
    tensor.device.device_id = 0;
    tensor.dtype.code = kDLUInt;
    tensor.dtype.bits = 8;
    tensor.dtype.lanes = 1;
    tensor.ndim = static_cast<int32_t>(_shape.size());

    managedTensor.deleter = [](DLManagedTensor* self) {
        delete[] self->dl_tensor.shape;
        self->dl_tensor.shape = nullptr;
        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
    };

    const int itemSizeDT = sizeof(uint8_t);  // dt.itemsize()
    try {
        tensor.shape = new int64_t[tensor.ndim];
        tensor.strides = new int64_t[tensor.ndim];
        for (int i = 0; i < tensor.ndim; ++i) {
            tensor.shape[i] = _shape[i];
            tensor.strides[i] = _stride[i];
            if (tensor.strides[i] % itemSizeDT != 0) {
                throw std::runtime_error("Stride must be a multiple of the element size in bytes");
            }
            tensor.strides[i] /= itemSizeDT;
        }
    } catch (...) {
        managedTensor.deleter(&managedTensor);
        throw;
    }

    m_dlTensor = DLPackTensor(std::move(managedTensor));
    return 0;
}
