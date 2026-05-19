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

#include "PyNvOnDemandDecoder.hpp"

using namespace std;
using namespace chrono;

namespace py = pybind11;

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
    if (CUDA_SUCCESS != res) {
        stringstream ss;

        if (lineNum > 0) {
            ss << __FILE__ << ":";
            ss << lineNum << endl;
        }

        const char* errName = nullptr;
        if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
            ss << "CUDA error with code " << res << endl;
        } else {
            ss << "CUDA error: " << errName << endl;
        }

        const char* errDesc = nullptr;
        cuGetErrorString(res, &errDesc);

        if (!errDesc) {
            ss << "No error string available" << endl;
        } else {
            ss << errDesc << endl;
        }

        throw runtime_error(ss.str());
    }
};

void Init_PyNvGopDecoder(py::module& m);
void Init_PyNvVideoReader(py::module& m);
void Init_PyNvSampleReader(py::module& m);
void Init_PyNvBatchAsyncStreamReader(py::module& m);
PYBIND11_MODULE(_PyNvOnDemandDecoder, m) {
    Init_PyNvVideoReader(m);
    Init_PyNvGopDecoder(m);
    Init_PyNvSampleReader(m);
    Init_PyNvBatchAsyncStreamReader(m);

    m.doc() = R"pbdoc(
        accvlab.on_demand_video_decoder
        ----------
        .. currentmodule:: PyNvVideoReader
        .. autosummary::
           :toctree: _generate

           PyNvVideoReader
           
    )pbdoc";
}
