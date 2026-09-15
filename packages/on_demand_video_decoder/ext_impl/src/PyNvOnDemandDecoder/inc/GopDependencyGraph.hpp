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

#pragma once

#include "cuviddec.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace accvlab::on_demand_video_decoder::internal {

struct GopDependencyGraph {
    std::vector<uint32_t> row_offsets;
    std::vector<uint32_t> parent_nodes;
    std::vector<uint32_t> coded_positions;
};

/**
 * Build a dependency graph without creating a CUDA context or an NVDEC decoder.
 * A missing value means that the codec has no dependency extractor yet, or
 * that the parser could not prove the GOP safe for selective decoding.
 */
std::optional<GopDependencyGraph> BuildGopDependencyGraph(cudaVideoCodec codec, int first_frame_id,
                                                          int gop_length,
                                                          const std::vector<int>& packet_sizes,
                                                          const std::vector<int>& frame_ids,
                                                          const std::vector<uint8_t>& packet_data);

std::vector<uint8_t> SerializeGopDependencyGraph(const GopDependencyGraph& graph);

std::optional<GopDependencyGraph> ParseGopDependencyGraph(const uint8_t* bytes, size_t size,
                                                          uint32_t expected_node_count);

std::vector<uint8_t> BuildGopDependencyDecodeMask(const GopDependencyGraph& graph,
                                                  const std::vector<uint32_t>& target_nodes);

}  // namespace accvlab::on_demand_video_decoder::internal
