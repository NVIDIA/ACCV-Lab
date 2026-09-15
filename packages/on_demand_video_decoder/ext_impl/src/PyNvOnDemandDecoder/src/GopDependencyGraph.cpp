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

#include "GopDependencyGraph.hpp"

#include "cuvid_dlopen.h"
#include "nvcuvid.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace accvlab::on_demand_video_decoder::internal {
namespace {

constexpr char kGraphMagic[] = "GCSR";
constexpr uint16_t kGraphVersion = 1;
constexpr size_t kGraphHeaderSize = 16;

const char* ParserLibraryName() {
#ifdef _WIN32
    return "nvcuvid.dll";
#else
    return "libnvcuvid.so.1";
#endif
}

std::runtime_error ParserError(const char* operation, CUresult status) {
    std::ostringstream message;
    message << operation << " failed with CUresult " << status;
    return std::runtime_error(message.str());
}

void AppendU16Le(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void AppendU32Le(std::vector<uint8_t>& bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
    bytes.push_back(static_cast<uint8_t>(value >> 16));
    bytes.push_back(static_cast<uint8_t>(value >> 24));
}

uint16_t ReadU16Le(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) | static_cast<uint16_t>(bytes[1]) << 8;
}

uint32_t ReadU32Le(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
           static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

void ValidateGopDependencyGraph(const GopDependencyGraph& graph, uint32_t expected_node_count) {
    const size_t node_count = graph.coded_positions.size();
    if (node_count != expected_node_count) {
        throw std::invalid_argument("[ERROR] dependency CSR node count does not match GOP length");
    }
    if (graph.row_offsets.size() != node_count + 1 || graph.row_offsets.front() != 0 ||
        graph.row_offsets.back() != graph.parent_nodes.size()) {
        throw std::invalid_argument("[ERROR] dependency CSR row offsets are invalid");
    }

    std::vector<uint8_t> seen_coded_positions(node_count, 0);
    for (uint32_t position : graph.coded_positions) {
        if (position >= node_count || seen_coded_positions[position]) {
            throw std::invalid_argument("[ERROR] dependency CSR coded positions are invalid");
        }
        seen_coded_positions[position] = 1;
    }

    for (size_t row = 1; row < graph.row_offsets.size(); ++row) {
        if (graph.row_offsets[row - 1] > graph.row_offsets[row] ||
            graph.row_offsets[row] > graph.parent_nodes.size()) {
            throw std::invalid_argument("[ERROR] dependency CSR row offsets are not ordered");
        }
    }
    for (size_t node = 0; node < node_count; ++node) {
        for (uint32_t edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
            const uint32_t parent = graph.parent_nodes[edge];
            if (parent >= node_count || graph.coded_positions[parent] >= graph.coded_positions[node]) {
                throw std::invalid_argument("[ERROR] dependency CSR edge is not a coded-order ancestor");
            }
        }
    }
}

class CuvidDependencyParserSession final {
   public:
    CuvidDependencyParserSession(cudaVideoCodec codec, int first_frame_id, int gop_length)
        : codec_(codec), first_frame_id_(first_frame_id), gop_length_(gop_length) {
        if (gop_length <= 0) {
            throw std::invalid_argument("dependency parsing requires a positive GOP length");
        }
        parents_.resize(static_cast<size_t>(gop_length));
        coded_positions_.assign(static_cast<size_t>(gop_length), UINT32_MAX);
        seen_nodes_.assign(static_cast<size_t>(gop_length), 0);

        try {
            LoadParserApi();

            CUVIDPARSERPARAMS parameters{};
            parameters.CodecType = codec_;
            parameters.ulMaxNumDecodeSurfaces = 1;
            parameters.ulMaxDisplayDelay = 1;
            parameters.pUserData = this;
            parameters.pfnSequenceCallback = HandleSequence;
            parameters.pfnDecodePicture = HandlePicture;
            parameters.pfnDisplayPicture = HandleDisplay;

            const CUresult status = create_(&parser_, &parameters);
            if (status != CUDA_SUCCESS) {
                throw ParserError("cuvidCreateVideoParser", status);
            }
        } catch (...) {
            CloseParserApi();
            throw;
        }
    }

    ~CuvidDependencyParserSession() { CloseParserApi(); }

    CuvidDependencyParserSession(const CuvidDependencyParserSession&) = delete;
    CuvidDependencyParserSession& operator=(const CuvidDependencyParserSession&) = delete;

    void ParseAccessUnit(const uint8_t* data, size_t size, int64_t frame_id) {
        if (finished_) {
            throw std::logic_error("dependency parser has already been finished");
        }
        if (!data || size == 0) {
            throw std::invalid_argument("dependency parser access unit is empty");
        }
        if (size > std::numeric_limits<unsigned long>::max()) {
            throw std::overflow_error("access unit is too large for the CUVID dependency parser");
        }

        pending_frame_ids_.push_back(frame_id);
        CUVIDSOURCEDATAPACKET packet{};
        packet.payload = data;
        packet.payload_size = static_cast<unsigned long>(size);
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.timestamp = frame_id;
        ParsePacket(packet);
    }

    std::optional<GopDependencyGraph> Finish() {
        if (finished_) {
            throw std::logic_error("dependency parser has already been finished");
        }

        CUVIDSOURCEDATAPACKET end_of_stream{};
        end_of_stream.flags = CUVID_PKT_ENDOFSTREAM;
        ParsePacket(end_of_stream);
        finished_ = true;

        if (!pending_frame_ids_.empty()) {
            valid_ = false;
            pending_frame_ids_.clear();
        }
        if (!valid_) return std::nullopt;

        for (size_t node = 0; node < seen_nodes_.size(); ++node) {
            if (!seen_nodes_[node] || coded_positions_[node] == UINT32_MAX) {
                return std::nullopt;
            }
        }

        GopDependencyGraph graph;
        graph.row_offsets.reserve(parents_.size() + 1);
        graph.row_offsets.push_back(0);
        for (const auto& parents : parents_) {
            graph.parent_nodes.insert(graph.parent_nodes.end(), parents.begin(), parents.end());
            graph.row_offsets.push_back(static_cast<uint32_t>(graph.parent_nodes.size()));
        }
        graph.coded_positions = std::move(coded_positions_);
        return graph;
    }

   private:
    static int CUDAAPI HandleSequence(void* opaque, CUVIDEOFORMAT* format) {
        auto* session = static_cast<CuvidDependencyParserSession*>(opaque);
        try {
            return session->OnSequence(format);
        } catch (const std::exception& error) {
            session->RecordCallbackFailure(error.what());
            return 0;
        } catch (...) {
            session->RecordCallbackFailure("unknown exception in dependency sequence callback");
            return 0;
        }
    }

    static int CUDAAPI HandlePicture(void* opaque, CUVIDPICPARAMS* picture) {
        auto* session = static_cast<CuvidDependencyParserSession*>(opaque);
        try {
            return session->OnPicture(picture);
        } catch (const std::exception& error) {
            session->RecordCallbackFailure(error.what());
            return 0;
        } catch (...) {
            session->RecordCallbackFailure("unknown exception in dependency picture callback");
            return 0;
        }
    }

    static int CUDAAPI HandleDisplay(void*, CUVIDPARSERDISPINFO*) { return 1; }

    int OnSequence(CUVIDEOFORMAT* format) {
        if (!format || format->codec != codec_) {
            valid_ = false;
            return 1;
        }
        return static_cast<int>(std::max(1U, static_cast<unsigned int>(format->min_num_decode_surfaces)));
    }

    int OnPicture(CUVIDPICPARAMS* picture) {
        if (!picture || pending_frame_ids_.empty()) {
            valid_ = false;
            return 1;
        }

        const int64_t frame_id = pending_frame_ids_.front();
        pending_frame_ids_.pop_front();
        const int64_t node_value = frame_id - first_frame_id_;
        if (node_value < 0 || node_value >= gop_length_) {
            valid_ = false;
            return 1;
        }

        const size_t node = static_cast<size_t>(node_value);
        if (seen_nodes_[node]) valid_ = false;

        std::vector<uint32_t> parent_nodes;
        switch (codec_) {
            case cudaVideoCodec_HEVC:
                if (!CollectHevcReferences(*picture, frame_id, node_value, parent_nodes)) {
                    valid_ = false;
                    return 1;
                }
                break;
            case cudaVideoCodec_H264:
            case cudaVideoCodec_AV1:
            default:
                // Codec-specific reference extraction is added here. The public
                // builder currently filters these codecs before parsing.
                valid_ = false;
                return 1;
        }

        parents_[node] = std::move(parent_nodes);
        coded_positions_[node] = next_coded_position_++;
        seen_nodes_[node] = 1;
        return 1;
    }

    bool CollectHevcReferences(CUVIDPICPARAMS& picture, int64_t frame_id, int64_t node_value,
                               std::vector<uint32_t>& parent_nodes) {
        const CUVIDHEVCPICPARAMS& hevc = picture.CodecSpecific.hevc;
        if (picture.field_pic_flag || hevc.NumPocLtCurr != 0) return false;

        std::unordered_set<uint32_t> unique_parents;
        auto collect_references = [&](const unsigned char* slots, int count) {
            for (int index = 0; index < count; ++index) {
                const unsigned int slot = slots[index];
                if (slot >= 16 || hevc.RefPicIdx[slot] < 0) {
                    valid_ = false;
                    continue;
                }

                const auto parent = hevc_poc_to_frame_id_.find(hevc.PicOrderCntVal[slot]);
                if (parent == hevc_poc_to_frame_id_.end()) {
                    valid_ = false;
                    continue;
                }

                const int64_t parent_node = parent->second - first_frame_id_;
                if (parent_node < 0 || parent_node >= gop_length_ || parent_node == node_value) {
                    valid_ = false;
                    continue;
                }
                unique_parents.insert(static_cast<uint32_t>(parent_node));
            }
        };

        collect_references(hevc.RefPicSetStCurrBefore, hevc.NumPocStCurrBefore);
        collect_references(hevc.RefPicSetStCurrAfter, hevc.NumPocStCurrAfter);

        parent_nodes.assign(unique_parents.begin(), unique_parents.end());
        std::sort(parent_nodes.begin(), parent_nodes.end());
        hevc_poc_to_frame_id_[hevc.CurrPicOrderCntVal] = frame_id;
        return true;
    }

    void ParsePacket(CUVIDSOURCEDATAPACKET& packet) {
        const CUresult status = parse_(parser_, &packet);
        if (!callback_error_.empty()) throw std::runtime_error(callback_error_);
        if (status != CUDA_SUCCESS) {
            throw ParserError("cuvidParseVideoData", status);
        }
    }

    void LoadParserApi() {
        library_ = cuvid_dlopen(ParserLibraryName());
        if (!library_) {
            const char* detail = cuvid_dlerror();
            throw std::runtime_error(std::string("Failed to load ") + ParserLibraryName() +
                                     (detail ? std::string(": ") + detail : std::string()));
        }

        auto load = [this](auto& function, const char* name) {
            using Function = std::remove_reference_t<decltype(function)>;
            function = reinterpret_cast<Function>(cuvid_dlsym(library_, name));
            if (!function) {
                throw std::runtime_error(std::string("Failed to load CUVID parser function ") + name);
            }
        };
        load(create_, "cuvidCreateVideoParser");
        load(parse_, "cuvidParseVideoData");
        load(destroy_, "cuvidDestroyVideoParser");
    }

    void CloseParserApi() noexcept {
        if (parser_ && destroy_) {
            destroy_(parser_);
            parser_ = nullptr;
        }
        create_ = nullptr;
        parse_ = nullptr;
        destroy_ = nullptr;
        if (library_) {
            cuvid_dlclose(library_);
            library_ = nullptr;
        }
    }

    void RecordCallbackFailure(const char* message) noexcept {
        valid_ = false;
        try {
            if (callback_error_.empty()) callback_error_ = message;
        } catch (...) {
        }
    }

    using CreateFunction = decltype(&cuvidCreateVideoParser);
    using ParseFunction = decltype(&cuvidParseVideoData);
    using DestroyFunction = decltype(&cuvidDestroyVideoParser);

    cuvid_lib library_ = nullptr;
    CreateFunction create_ = nullptr;
    ParseFunction parse_ = nullptr;
    DestroyFunction destroy_ = nullptr;
    CUvideoparser parser_ = nullptr;
    cudaVideoCodec codec_;
    int first_frame_id_ = 0;
    int gop_length_ = 0;
    uint32_t next_coded_position_ = 0;
    bool valid_ = true;
    bool finished_ = false;
    std::string callback_error_;
    std::deque<int64_t> pending_frame_ids_;
    std::unordered_map<int, int64_t> hevc_poc_to_frame_id_;
    std::vector<std::vector<uint32_t>> parents_;
    std::vector<uint32_t> coded_positions_;
    std::vector<uint8_t> seen_nodes_;
};

std::optional<GopDependencyGraph> BuildGraphWithCuvidParser(cudaVideoCodec codec, int first_frame_id,
                                                            int gop_length,
                                                            const std::vector<int>& packet_sizes,
                                                            const std::vector<int>& frame_ids,
                                                            const std::vector<uint8_t>& packet_data) {
    if (packet_sizes.size() != frame_ids.size()) {
        throw std::invalid_argument("dependency parser packet sizes and frame IDs have different lengths");
    }

    CuvidDependencyParserSession session(codec, first_frame_id, gop_length);
    const int64_t gop_end = static_cast<int64_t>(first_frame_id) + gop_length;
    size_t packet_offset = 0;

    for (size_t packet = 0; packet < packet_sizes.size(); ++packet) {
        const int packet_size = packet_sizes[packet];
        // The demux path uses non-positive sizes for control entries without payload.
        if (packet_size <= 0) continue;
        if (packet_offset > packet_data.size() ||
            static_cast<size_t>(packet_size) > packet_data.size() - packet_offset) {
            throw std::invalid_argument("serialized GOP packet payload is truncated");
        }

        const int frame_id = frame_ids[packet];
        if (frame_id >= first_frame_id && frame_id < gop_end) {
            session.ParseAccessUnit(packet_data.data() + packet_offset, static_cast<size_t>(packet_size),
                                    frame_id);
        }
        packet_offset += static_cast<size_t>(packet_size);
    }

    if (packet_offset != packet_data.size()) {
        throw std::invalid_argument("serialized GOP contains unindexed packet data");
    }
    return session.Finish();
}

}  // namespace

std::optional<GopDependencyGraph> BuildGopDependencyGraph(cudaVideoCodec codec, int first_frame_id,
                                                          int gop_length,
                                                          const std::vector<int>& packet_sizes,
                                                          const std::vector<int>& frame_ids,
                                                          const std::vector<uint8_t>& packet_data) {
    switch (codec) {
        case cudaVideoCodec_HEVC:
            return BuildGraphWithCuvidParser(codec, first_frame_id, gop_length, packet_sizes, frame_ids,
                                             packet_data);
        case cudaVideoCodec_H264:
        case cudaVideoCodec_AV1:
        default:
            return std::nullopt;
    }
}

std::vector<uint8_t> SerializeGopDependencyGraph(const GopDependencyGraph& graph) {
    if (graph.coded_positions.size() > std::numeric_limits<uint32_t>::max() ||
        graph.parent_nodes.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("[ERROR] dependency CSR graph is too large");
    }
    const uint32_t node_count = static_cast<uint32_t>(graph.coded_positions.size());
    ValidateGopDependencyGraph(graph, node_count);

    std::vector<uint8_t> bytes;
    bytes.reserve(kGraphHeaderSize +
                  sizeof(uint32_t) *
                      (graph.row_offsets.size() + graph.parent_nodes.size() + graph.coded_positions.size()));
    bytes.insert(bytes.end(), kGraphMagic, kGraphMagic + 4);
    AppendU16Le(bytes, kGraphVersion);
    AppendU16Le(bytes, 0);  // flags
    AppendU32Le(bytes, node_count);
    AppendU32Le(bytes, static_cast<uint32_t>(graph.parent_nodes.size()));
    for (uint32_t value : graph.row_offsets) AppendU32Le(bytes, value);
    for (uint32_t value : graph.parent_nodes) AppendU32Le(bytes, value);
    for (uint32_t value : graph.coded_positions) AppendU32Le(bytes, value);
    return bytes;
}

std::optional<GopDependencyGraph> ParseGopDependencyGraph(const uint8_t* bytes, size_t size,
                                                          uint32_t expected_node_count) {
    if (size < 4 || std::memcmp(bytes, kGraphMagic, 4) != 0) return std::nullopt;
    if (size < kGraphHeaderSize) {
        throw std::invalid_argument("[ERROR] dependency CSR header is truncated");
    }

    const uint16_t version = ReadU16Le(bytes + 4);
    const uint16_t flags = ReadU16Le(bytes + 6);
    const uint32_t node_count = ReadU32Le(bytes + 8);
    const uint32_t edge_count = ReadU32Le(bytes + 12);
    if (version != kGraphVersion || flags != 0) {
        throw std::invalid_argument("[ERROR] unsupported dependency CSR trailer");
    }
    if (node_count != expected_node_count) {
        throw std::invalid_argument("[ERROR] dependency CSR node count does not match GOP length");
    }

    const uint64_t word_count = static_cast<uint64_t>(node_count) + 1 + edge_count + node_count;
    const uint64_t expected_size = kGraphHeaderSize + word_count * sizeof(uint32_t);
    if (expected_size != size) {
        throw std::invalid_argument("[ERROR] dependency CSR trailer size is invalid");
    }

    GopDependencyGraph graph;
    const uint8_t* cursor = bytes + kGraphHeaderSize;
    graph.row_offsets.resize(static_cast<size_t>(node_count) + 1);
    for (uint32_t& value : graph.row_offsets) {
        value = ReadU32Le(cursor);
        cursor += sizeof(uint32_t);
    }
    graph.parent_nodes.resize(edge_count);
    for (uint32_t& value : graph.parent_nodes) {
        value = ReadU32Le(cursor);
        cursor += sizeof(uint32_t);
    }
    graph.coded_positions.resize(node_count);
    for (uint32_t& value : graph.coded_positions) {
        value = ReadU32Le(cursor);
        cursor += sizeof(uint32_t);
    }

    ValidateGopDependencyGraph(graph, expected_node_count);
    return graph;
}

std::vector<uint8_t> BuildGopDependencyDecodeMask(const GopDependencyGraph& graph,
                                                  const std::vector<uint32_t>& target_nodes) {
    if (graph.coded_positions.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("[ERROR] dependency CSR graph is too large");
    }
    const uint32_t node_count = static_cast<uint32_t>(graph.coded_positions.size());
    ValidateGopDependencyGraph(graph, node_count);

    std::vector<uint8_t> needed_nodes(node_count, 0);
    std::vector<uint32_t> stack;
    stack.reserve(target_nodes.size());
    for (uint32_t node : target_nodes) {
        if (node >= node_count) {
            throw std::invalid_argument("[ERROR] dependency graph target is outside the GOP");
        }
        if (!needed_nodes[node]) {
            needed_nodes[node] = 1;
            stack.push_back(node);
        }
    }

    while (!stack.empty()) {
        const uint32_t node = stack.back();
        stack.pop_back();
        for (uint32_t edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
            const uint32_t parent = graph.parent_nodes[edge];
            if (!needed_nodes[parent]) {
                needed_nodes[parent] = 1;
                stack.push_back(parent);
            }
        }
    }
    return needed_nodes;
}

}  // namespace accvlab::on_demand_video_decoder::internal
