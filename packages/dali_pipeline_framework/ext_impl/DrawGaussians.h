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

#ifndef DRAW_GAUSSIANS_H_
#define DRAW_GAUSSIANS_H_

#include <algorithm>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"

namespace custom_operators {

template <typename Backend>
class DrawGaussians : public ::dali::Operator<Backend> {
   public:
    explicit DrawGaussians(const ::dali::OpSpec& spec);

    virtual ~DrawGaussians();

    DrawGaussians(const DrawGaussians&) = delete;
    DrawGaussians& operator=(const DrawGaussians&) = delete;
    DrawGaussians(DrawGaussians&&) = delete;
    DrawGaussians& operator=(DrawGaussians&&) = delete;

   protected:
    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc, const ::dali::Workspace& ws) override {
        output_desc.resize(1);

        const auto& src_input = ws.template Input<Backend>(0);
        const auto& src_input_shape = src_input.shape();
        const int batch_size = src_input_shape.num_samples();

        if (src_input.type() != ::dali::DALIDataType::DALI_FLOAT) {
            DALI_FAIL("heat_map has to be of type FLOAT");
        }
        if (ws.template Input<Backend>(1).type() != ::dali::DALIDataType::DALI_BOOL) {
            DALI_FAIL("active has to be of type BOOL");
        }
        if (ws.template Input<Backend>(2).type() != ::dali::DALIDataType::DALI_INT32) {
            DALI_FAIL("class_ids has to be of type INT32");
        }
        if (ws.template Input<Backend>(3).type() != ::dali::DALIDataType::DALI_INT32) {
            DALI_FAIL("centers has to be of type INT32");
        }
        if (ws.template Input<Backend>(4).type() != ::dali::DALIDataType::DALI_FLOAT) {
            DALI_FAIL("radii has to be of type FLOAT");
        }

        std::vector<dali::TensorShape<>> sample_shapes;
        sample_shapes.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            const size_t num_dims = src_input_shape[i].size();

            size_t curr_dim = 0;
            size_t num_channels = 1;

            const bool has_3_dims = num_dims == 3;
            if (has_3_dims) {
                num_channels = src_input_shape[i][curr_dim++];
            } else if (num_dims != 2) {
                DALI_FAIL("Heatmap has to have 2 or 3 dimensions");
            }
            const size_t height = src_input_shape[i][curr_dim++];
            const size_t width = src_input_shape[i][curr_dim];

            const dali::TensorShape<> single_sample_shape =
                has_3_dims ? dali::TensorShape<>{num_channels, height, width}
                           : dali::TensorShape<>{height, width};
            sample_shapes.push_back(single_sample_shape);
        }

        output_desc[0].shape = dali::TensorListShape<>(sample_shapes);
        output_desc[0].type = DALI_FLOAT;

        //std::cout << "Finished SetupImpl()" << std::endl;

        return true;
    };

    void RunImpl(::dali::Workspace& ws) override;

   private:
    std::vector<float> _k_for_classes;

    float _radius_to_sigma_factor;
};

}  // namespace custom_operators

#endif