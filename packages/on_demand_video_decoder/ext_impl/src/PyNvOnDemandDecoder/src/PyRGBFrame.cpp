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

#include "../inc/PyRGBFrame.hpp"

RGBFrame::RGBFrame(const std::vector<size_t>& _shape, const std::vector<size_t>& _stride,
                   const std::string& _typeStr, size_t _streamid, CUdeviceptr _data, bool _readOnly,
                   bool _isBRG) {
    assert(_shape.size() == 3 && _stride.size() == 3 && "_shape and _stride need to have size of 3");
    assert(_shape[2] == 3 && "IMage has to have 3 chaneels, i.e. _shape[2] has to be 3");
    shape = {_shape[0], _shape[1], 3};
    stride = {_stride[0], _stride[1], _stride[2]};
    typestr = _typeStr;
    data = _data;
    readOnly = _readOnly;
    stream = reinterpret_cast<CUstream>(_streamid);
    isBGR = isBGR;
}

RGBFrame::RGBFrame(const CAIMemoryView& to_convert, bool _isBRG) {
    assert(to_convert.shape.size() == 3 && to_convert.stride.size() == 3 &&
           "CAIMemoryView need to have 3 dimensions to be converted to RGBFrame");
    assert(to_convert.shape[2] == 3 &&
           "CAIMemoryView need to have 3 channel (i.e. size of 3 in dimension 2)");
    shape = {to_convert.shape[0], to_convert.shape[1], 3};
    stride = {to_convert.stride[0], to_convert.stride[1], to_convert.stride[2]};
    typestr = to_convert.typestr;
    data = to_convert.data;
    readOnly = to_convert.readOnly;
    stream = to_convert.stream;
    isBGR = isBGR;
}

RGBFrame::RGBFrame() {
    shape = {0, 0, 0};
    stride = {1, 1, 1};
    typestr = "|u1";
    data = reinterpret_cast<CUdeviceptr>(nullptr);
    readOnly = true;
    stream = (CUstream)2;
    isBGR = false;
}

void RGBFrame::release_data() {
    CUDA_DRVAPI_CALL(cuMemFree(data));
    data = reinterpret_cast<CUdeviceptr>(nullptr);
}

bool RGBFrame::is_of_size(size_t height, size_t width) {
    const bool shape_correct = (height == std::get<0>(shape)) && (width == std::get<1>(shape));
    return shape_correct;
}