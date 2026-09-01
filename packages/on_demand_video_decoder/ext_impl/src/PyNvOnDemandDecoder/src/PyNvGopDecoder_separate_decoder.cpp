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

#include "PyNvGopDecoder.hpp"
#include "FrameOutput.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

#include "ColorConvertKernels.cuh"

namespace frame_output = accvlab::on_demand_video_decoder::internal::frame_output;

void PyNvGopDecoder::get_gop_internal(
    const std::vector<std::string>& filepaths, const std::vector<int> frame_ids,
    const FastStreamInfo* fastStreamInfos, std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
    std::vector<std::vector<std::unique_ptr<uint8_t[]>>>& vpacket_array,
    std::vector<std::vector<int>>& all_gop_lens, std::vector<std::vector<int>>& all_first_frame_ids) {
    int st = 0;
    if (filepaths.size() != frame_ids.size()) {
        throw std::invalid_argument("[ERROR] filepaths and frame_ids must have the same length");
    }
    if (filepaths.size() > max_num_files) {
        throw std::invalid_argument("[ERROR] filepaths size is greater than max_num_files");
    }

    const size_t total_frames = frame_ids.size();

    // Initialize packet queues and arrays
    vpacket_queue.reserve(total_frames);
    vpacket_array.reserve(total_frames);
    for (size_t i = 0; i < total_frames; ++i) {
        vpacket_queue.emplace_back(std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>());
        vpacket_queue[i]->setSize(MAX_SIZE);
        vpacket_array.emplace_back();
    }

    // Initialize demuxers
    st = InitializeDemuxers(filepaths, demuxers, fastStreamInfos);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitializeDemuxers failed.");
    }

    // Initialize GOP metadata containers
    all_gop_lens.resize(total_frames);
    all_first_frame_ids.resize(total_frames);
    std::vector<std::vector<int>> all_sorted_frame_ids(total_frames);

    // Extract packets for each video
    nvtxRangePushA("Packet extraction");
    for (int i = 0; i < total_frames; ++i) {
        try {
            std::vector<int> sorted_frame_ids = {frame_ids[i]};
            // Only process GOP info for VFR videos
            if (demuxers[i]->IsVFRV2()) {
                st = ExtractAndProcessGopInfo(demuxers[i], sorted_frame_ids, all_first_frame_ids[i],
                                              all_gop_lens[i]);
                if (st != 0) {
                    throw std::runtime_error("[ERROR] extract and process gop info failed for file: " +
                                             filepaths[i]);
                }
            }
            all_sorted_frame_ids[i] = std::move(sorted_frame_ids);
        } catch (const std::exception& e) {
            this->force_join_all();
            LOG(ERROR) << "Packet extraction failed: " << e.what();
            throw std::runtime_error(e.what());
        }
    }
    try {
        demux_pool.submit_indexed(total_frames, [&](size_t index) {
            DemuxGopProc(demuxers[index].get(), vpacket_queue[index].get(), all_sorted_frame_ids[index],
                         all_first_frame_ids[index], all_gop_lens[index], vpacket_array[index], true);
        });
    } catch (const std::exception& e) {
        this->force_join_all();
        LOG(ERROR) << "Packet extraction failed: " << e.what();
        throw std::runtime_error(e.what());
    }
    nvtxRangePop();  // Packet extraction

    // Wait for all demux threads to complete
    nvtxRangePushA("Demux thread join");
    try {
        demux_pool.wait_all();
    } catch (const std::exception& e) {
        this->force_join_all();
        throw std::runtime_error(e.what());
    }

    // Validate extracted packets
    for (int i = 0; i < total_frames; i++) {
        if (vpacket_array.at(i).size() == 0) {
            this->force_join_all();
            LOG(ERROR) << "vpacket_array[" << i << "] is empty";
            throw std::runtime_error("[ERROR] vpacket_array is empty");
        }
    }
    nvtxRangePop();  // Demux thread join
}

std::vector<SerializedPacketBundle> PyNvGopDecoder::get_gop_list(const std::vector<std::string>& filepaths,
                                                                 const std::vector<int> frame_ids,
                                                                 const FastStreamInfo* fastStreamInfos) {
    nvtxRangePushA("GetGOPList");
    const size_t total_videos = frame_ids.size();

    // Use internal implementation to extract GOP data
    std::vector<std::unique_ptr<PyNvGopDemuxer>> demuxers;
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    std::vector<std::vector<std::unique_ptr<uint8_t[]>>> vpacket_array;
    std::vector<std::vector<int>> all_gop_lens;
    std::vector<std::vector<int>> all_first_frame_ids;

    get_gop_internal(filepaths, frame_ids, fastStreamInfos, demuxers, vpacket_queue, vpacket_array,
                     all_gop_lens, all_first_frame_ids);

    // Create separate bundle for each video
    nvtxRangePushA("CreateSerializedPacketBundles");
    std::vector<SerializedPacketBundle> results;
    results.reserve(total_videos);

    for (size_t i = 0; i < total_videos; ++i) {
        // Create temporary vectors containing only data for this video
        std::vector<std::unique_ptr<PyNvGopDemuxer>> single_demuxer;
        single_demuxer.push_back(std::move(demuxers[i]));

        std::vector<std::vector<int>> single_gop_lens = {all_gop_lens[i]};
        std::vector<std::vector<int>> single_first_frame_ids = {all_first_frame_ids[i]};

        std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> single_queue;
        single_queue.push_back(std::move(vpacket_queue[i]));

        std::vector<std::vector<std::unique_ptr<uint8_t[]>>> single_array;
        single_array.push_back(std::move(vpacket_array[i]));

        // Create bundle for this single video
        SerializedPacketBundle bundle = createSerializedPacketBundle(
            1,  // Only one frame per video
            single_demuxer, single_gop_lens, single_first_frame_ids, single_queue, single_array);

        // Restore demuxer for proper cleanup (moved it back from single_demuxer)
        demuxers[i] = std::move(single_demuxer[0]);

        results.push_back(std::move(bundle));
    }
    nvtxRangePop();  // CreateSerializedPacketBundles

    nvtxRangePop();  // GetGOPList
    return results;
}

void PyNvGopDecoder::decode_from_gop(const uint8_t* data, size_t size,
                                     const std::vector<std::string>& filepaths,
                                     const std::vector<int> frame_ids, bool convert_to_rgb, bool as_bgr,
                                     std::vector<DecodedFrameExt>* out_if_no_color_conversion,
                                     std::vector<RGBFrame>* out_if_color_converted) {
    assert(!convert_to_rgb || (out_if_color_converted != nullptr && out_if_no_color_conversion == nullptr) &&
                                  "If color conversion is used, out_if_color_converted has to point "
                                  "to vector where to write the results, and "
                                  "out_if_no_color_conversion has to be nullptr");
    assert(convert_to_rgb || (out_if_color_converted == nullptr && out_if_no_color_conversion != nullptr) &&
                                 "If color conversion is not used, out_if_color_converted has to "
                                 "be nullptr, and out_if_no_color_conversion has to point to "
                                 "vector where to write the results");
    if (filepaths.size() != frame_ids.size()) {
        throw std::invalid_argument("[ERROR] filepaths and frame_ids must have the same length");
    }
    if (filepaths.size() > max_num_files) {
        throw std::invalid_argument("[ERROR] filepaths size is greater than max_num_files");
    }

    nvtxRangePushA("DecodeFromGOP");

    // Parse serialized packet data using embedded offset table
    std::vector<int> color_ranges;
    std::vector<int> codec_ids;
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> frame_sizes;
    std::vector<int> gop_lens;
    std::vector<int> first_frame_ids;
    std::vector<std::vector<int>> packets_bytes;
    std::vector<std::vector<int>> decode_idxs;
    std::vector<const uint8_t*> packet_binary_data_ptrs;
    std::vector<size_t> packet_binary_data_sizes;

    nvtxRangePushA("ParseSerializedPacketData");
    const uint32_t total_frames = parseSerializedPacketData(
        data, size, color_ranges, codec_ids, widths, heights, frame_sizes, gop_lens, first_frame_ids,
        packets_bytes, decode_idxs, packet_binary_data_ptrs, packet_binary_data_sizes);
    nvtxRangePop();  // ParseSerializedPacketData

    if (total_frames != frame_ids.size()) {
        throw std::invalid_argument("[ERROR] frame_ids size " + std::to_string(frame_ids.size()) +
                                    " does not match serialized data frame count " +
                                    std::to_string(total_frames));
    }

    // Validate frame_id ranges
    for (uint32_t i = 0; i < total_frames; ++i) {
        if (frame_ids[i] < first_frame_ids[i] || frame_ids[i] >= first_frame_ids[i] + gop_lens[i]) {
            throw std::invalid_argument("[ERROR] GOP range from " + std::to_string(first_frame_ids[i]) +
                                        " to " + std::to_string(first_frame_ids[i] + gop_lens[i]) +
                                        " does not contain frame_id " + std::to_string(frame_ids[i]));
        }
    }

    // create packet queue and reconstruct from serialized data
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    vpacket_queue.resize(total_frames);

    for (int i = 0; i < total_frames; ++i) {
        // find #of skip packet for reuse
        int skip_packets = 0;
        const int last_frame_id = this->last_decoded_frame_infos[i].frame_id;
        if (this->last_decoded_frame_infos[i].filename != filepaths[i]) {
            skip_packets = 0;
        } else if (last_frame_id < first_frame_ids[i] || last_frame_id >= first_frame_ids[i] + gop_lens[i]) {
            skip_packets = 0;
        } else if (last_frame_id >= frame_ids[i]) {
            skip_packets = 0;
        } else {
            skip_packets = this->last_decoded_frame_infos[i].packet_id;
        }
        if (skip_packets == 0) {
            reset_last_decoded_frame_info(this->last_decoded_frame_infos[i]);
        }

        vpacket_queue[i] = std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>();
        vpacket_queue[i]->setSize(MAX_SIZE);

        // reconstruct packet queue from serialized data
        size_t offset = 0;
        const int num_packets = packets_bytes[i].size();
        for (int j = 0; j < num_packets; ++j) {
            int packet_bytes = packets_bytes.at(i).at(j);
            int decode_idx = decode_idxs.at(i).at(j);

            if (skip_packets > 0) {
                skip_packets--;
                if (packet_bytes > 0) {
                    offset += packet_bytes;
                }
                continue;
            }

            if (packet_bytes == -1) {
                // End of stream marker
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, -1, 0));
            } else if (packet_bytes == 0) {
                // GOP end marker
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, 0, 0));
            } else {
                // Regular packet - use pointer from numpy data
                uint8_t* pVideo = const_cast<uint8_t*>(packet_binary_data_ptrs[i] + offset);
                offset += packet_bytes;

                if (decode_idx == frame_ids[i]) {
                    decode_idx = decode_idx * 2;
                } else {
                    decode_idx = decode_idx * 2;
                }
                vpacket_queue[i]->push_back(std::make_tuple(pVideo, packet_bytes, decode_idx));
            }
        }
    }

    // start decoding process via encapsulated function
    int st = main_decode(color_ranges, codec_ids, widths, heights, frame_sizes, filepaths, frame_ids,
                         convert_to_rgb, as_bgr, vpacket_queue, out_if_no_color_conversion,
                         out_if_color_converted);
    if (st != 0) {
        throw std::runtime_error("[ERROR] main_decode failed.");
    }

    nvtxRangePop();
}

void PyNvGopDecoder::decode_from_packet_list(std::vector<std::vector<int>> packets_bytes,
                                             std::vector<std::vector<int>> decode_idxs,
                                             std::vector<int> widths, std::vector<int> heights,
                                             std::vector<std::vector<const uint8_t*>> packet_binary_data_ptrs,
                                             std::vector<int> frame_ids, bool as_bgr,
                                             std::vector<RGBFrame>* dst) {
    nvtxRangePushA("DecodeFromGOP");

    int st = 0;
    const int total_frames = frame_ids.size();

    std::vector<int> dummp;

    ensureCudaContextInitialized();

    st = InitGpuMemPool(heights, widths, dummp, true);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitGpuMemPool failed.");
    }

    std::vector<std::vector<uint8_t*>> per_file_frame_buffers;
    st = GetFileFrameBuffers(&widths, &heights, nullptr, true, per_file_frame_buffers);
    if (st != 0) {
        throw std::runtime_error("[ERROR] GetFileFrameBuffers failed.");
    }

    std::vector<std::vector<RGBFrame>> rgb_frames;
    rgb_frames.resize(total_frames);
    dst->resize(total_frames);

    // create packet queue and reconstruct from serialized data
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    vpacket_queue.resize(total_frames);

    for (int i = 0; i < total_frames; ++i) {
        // find #of skip packet for reuse
        int skip_packets = 0;
        // const int last_frame_id = this->last_decoded_frame_infos[i].frame_id;
        // if (this->last_decoded_frame_infos[i].filename != filepaths[i]) {
        //     skip_packets = 0;
        // } else if (last_frame_id < first_frame_ids[i] || last_frame_id >= first_frame_ids[i] + gop_lens[i]) {
        //     skip_packets = 0;
        // } else if (last_frame_id >= frame_ids[i]) {
        //     skip_packets = 0;
        // } else {
        //     skip_packets = this->last_decoded_frame_infos[i].packet_id;
        // }
        if (skip_packets == 0) {
            reset_last_decoded_frame_info(this->last_decoded_frame_infos[i]);
        }

        vpacket_queue[i] = std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>();
        vpacket_queue[i]->setSize(MAX_SIZE);

        // reconstruct packet queue from serialized data
        const int num_packets = packets_bytes[i].size();
        for (int j = 0; j < num_packets; ++j) {
            int packet_bytes = packets_bytes.at(i).at(j);
            int decode_idx = decode_idxs.at(i).at(j);

            if (skip_packets > 0) {
                skip_packets--;
                continue;
            }

            if (packet_bytes == -1) {
                // End of stream marker
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, -1, 0));
            } else if (packet_bytes == 0) {
                // GOP end marker
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, 0, 0));
            } else {
                // Regular packet - use direct pointer from packet_binary_data_ptrs
                uint8_t* pVideo = const_cast<uint8_t*>(packet_binary_data_ptrs[i][j]);

                if (decode_idx == frame_ids[i]) {
                    decode_idx = decode_idx * 2;
                } else {
                    decode_idx = decode_idx * 2;
                }
                vpacket_queue[i]->push_back(std::make_tuple(pVideo, packet_bytes, decode_idx));
            }
        }
    }

    // Prepare minimal metadata required by main_decode
    std::vector<int> color_ranges(total_frames, 0);        // Unknown -> default to limited/unspecified
    std::vector<int> codec_ids(total_frames, 0);           // Ignored if decoders already initialized
    std::vector<int> frame_sizes;                          // Unused in RGB path
    std::vector<std::string> filepaths(total_frames, "");  // Empty filenames as in v2 path

    // Call unified decode path (RGB conversion always true for v2)
    st = main_decode(color_ranges, codec_ids, widths, heights, frame_sizes, filepaths, frame_ids,
                     /*convert_to_rgb=*/true, as_bgr, vpacket_queue, /*out_if_no_color_conversion=*/nullptr,
                     /*out_if_color_converted=*/dst);
    if (st != 0) {
        nvtxRangePop();
        throw std::runtime_error("[ERROR] main_decode failed.");
    }

    nvtxRangePop();
}

void PyNvGopDecoder::decode_from_gop_list(const std::vector<const uint8_t*>& datas,
                                          const std::vector<size_t>& sizes,
                                          const std::vector<std::string>& filepaths,
                                          const std::vector<int>& frame_ids, bool convert_to_rgb, bool as_bgr,
                                          std::vector<DecodedFrameExt>* out_if_no_color_conversion,
                                          std::vector<RGBFrame>* out_if_color_converted,
                                          bool skip_final_sync) {
    if (convert_to_rgb) {
        if (out_if_color_converted == nullptr || out_if_no_color_conversion != nullptr) {
            throw std::invalid_argument(
                "[ERROR] RGB decode requires out_if_color_converted and a null "
                "out_if_no_color_conversion");
        }
    } else {
        if (out_if_color_converted != nullptr || out_if_no_color_conversion == nullptr) {
            throw std::invalid_argument(
                "[ERROR] raw decode requires out_if_no_color_conversion and a null "
                "out_if_color_converted");
        }
    }
    nvtxRangePushA("DecodeFromPacketList");

    if (datas.size() != sizes.size()) {
        nvtxRangePop();
        throw std::invalid_argument("[ERROR] datas and sizes must have the same length");
    }

    // Aggregated containers across all input bundles
    std::vector<int> color_ranges_all;
    std::vector<int> codec_ids_all;
    std::vector<int> widths_all;
    std::vector<int> heights_all;
    std::vector<int> frame_sizes_all;
    std::vector<int> gop_lens_all;
    std::vector<int> first_frame_ids_all;
    std::vector<std::vector<int>> packets_bytes_all;
    std::vector<std::vector<int>> decode_idxs_all;
    std::vector<const uint8_t*> packet_binary_data_ptrs_all;
    std::vector<size_t> packet_binary_data_sizes_all;

    // First pass: parse each bundle and aggregate
    uint32_t aggregated_frames = 0;
    for (size_t b = 0; b < datas.size(); ++b) {
        const uint8_t* data_ptr = datas[b];
        size_t data_size = sizes[b];

        std::vector<int> color_ranges;
        std::vector<int> codec_ids;
        std::vector<int> widths;
        std::vector<int> heights;
        std::vector<int> frame_sizes;
        std::vector<int> gop_lens;
        std::vector<int> first_frame_ids;
        std::vector<std::vector<int>> packets_bytes;
        std::vector<std::vector<int>> decode_idxs;
        std::vector<const uint8_t*> packet_binary_data_ptrs;
        std::vector<size_t> packet_binary_data_sizes;

        uint32_t frames_in_bundle = parseSerializedPacketData(
            data_ptr, data_size, color_ranges, codec_ids, widths, heights, frame_sizes, gop_lens,
            first_frame_ids, packets_bytes, decode_idxs, packet_binary_data_ptrs, packet_binary_data_sizes);

        // Append to aggregated containers
        for (uint32_t i = 0; i < frames_in_bundle; ++i) {
            color_ranges_all.push_back(color_ranges[i]);
            codec_ids_all.push_back(codec_ids[i]);
            widths_all.push_back(widths[i]);
            heights_all.push_back(heights[i]);
            frame_sizes_all.push_back(frame_sizes[i]);
            gop_lens_all.push_back(gop_lens[i]);
            first_frame_ids_all.push_back(first_frame_ids[i]);
            packets_bytes_all.push_back(std::move(packets_bytes[i]));
            decode_idxs_all.push_back(std::move(decode_idxs[i]));
            packet_binary_data_ptrs_all.push_back(packet_binary_data_ptrs[i]);
            packet_binary_data_sizes_all.push_back(packet_binary_data_sizes[i]);
        }
        aggregated_frames += frames_in_bundle;
    }

    if (aggregated_frames != frame_ids.size()) {
        nvtxRangePop();
        throw std::invalid_argument(
            "[ERROR] frame_ids size does not match aggregated frames from input bundles");
    }
    if (frame_ids.size() != filepaths.size()) {
        nvtxRangePop();
        throw std::invalid_argument("[ERROR] filepaths and frame_ids must have the same length");
    }

    for (uint32_t i = 0; i < aggregated_frames; ++i) {
        if (frame_ids[i] < first_frame_ids_all[i] ||
            frame_ids[i] >= first_frame_ids_all[i] + gop_lens_all[i]) {
            nvtxRangePop();
            throw std::invalid_argument("[ERROR] GOP range from " + std::to_string(first_frame_ids_all[i]) +
                                        " to " + std::to_string(first_frame_ids_all[i] + gop_lens_all[i]) +
                                        " does not contain frame_id " + std::to_string(frame_ids[i]) +
                                        " for file " + filepaths[i]);
        }
    }

    const int total_frames = static_cast<int>(aggregated_frames);

    if (total_frames > this->max_num_files) {
        nvtxRangePop();
        throw std::invalid_argument("[ERROR] total frames exceed max_num_files");
    }

    // Reconstruct packet queues per frame
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    vpacket_queue.resize(total_frames);

    for (int i = 0; i < total_frames; ++i) {
        int skip_packets = 0;
        const int last_frame_id = this->last_decoded_frame_infos[i].frame_id;
        if (this->last_decoded_frame_infos[i].filename != filepaths[i]) {
            skip_packets = 0;
        } else if (last_frame_id < first_frame_ids_all[i] ||
                   last_frame_id >= first_frame_ids_all[i] + gop_lens_all[i]) {
            skip_packets = 0;
        } else if (last_frame_id >= frame_ids[i]) {
            skip_packets = 0;
        } else {
            skip_packets = this->last_decoded_frame_infos[i].packet_id;
        }
        if (skip_packets == 0) {
            reset_last_decoded_frame_info(this->last_decoded_frame_infos[i]);
        }

        vpacket_queue[i] = std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>();
        vpacket_queue[i]->setSize(MAX_SIZE);

        size_t offset = 0;
        const int num_packets = static_cast<int>(packets_bytes_all[i].size());
        for (int j = 0; j < num_packets; ++j) {
            const int packet_bytes = packets_bytes_all[i][j];
            int decode_idx = decode_idxs_all[i][j];

            if (skip_packets > 0) {
                skip_packets--;
                if (packet_bytes > 0) {
                    offset += packet_bytes;
                }
                continue;
            }

            if (packet_bytes == -1) {
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, -1, 0));
            } else if (packet_bytes == 0) {
                vpacket_queue[i]->push_back(std::make_tuple(nullptr, 0, 0));
            } else {
                uint8_t* pVideo = const_cast<uint8_t*>(packet_binary_data_ptrs_all[i] + offset);
                offset += packet_bytes;

                // Timestamp encoding: keep consistent with existing logic
                decode_idx = decode_idx * 2;
                vpacket_queue[i]->push_back(std::make_tuple(pVideo, packet_bytes, decode_idx));
            }
        }
    }

    int st = main_decode(color_ranges_all, codec_ids_all, widths_all, heights_all, frame_sizes_all, filepaths,
                         frame_ids, convert_to_rgb, as_bgr, vpacket_queue, out_if_no_color_conversion,
                         out_if_color_converted, skip_final_sync);
    if (st != 0) {
        nvtxRangePop();
        throw std::runtime_error("[ERROR] main_decode failed.");
    }

    nvtxRangePop();
}

void PyNvGopDecoder::decode_from_gop_groups(const std::vector<const uint8_t*>& datas,
                                            const std::vector<size_t>& sizes,
                                            const std::vector<std::string>& source_names,
                                            const std::vector<std::vector<int>>& frame_id_groups, bool as_bgr,
                                            std::vector<RGBFrame>& output) {
    nvtxRangePushA("DecodeFromGOPGroups");

    const size_t num_groups = datas.size();
    if (sizes.size() != num_groups || source_names.size() != num_groups ||
        frame_id_groups.size() != num_groups) {
        nvtxRangePop();
        throw std::invalid_argument(
            "[ERROR] gop_datas, source_names, and frame_id_groups must have the same length");
    }
    if (num_groups > static_cast<size_t>(max_num_files)) {
        nvtxRangePop();
        throw std::invalid_argument("[ERROR] number of GOP groups exceeds max_num_files");
    }

    if (num_groups == 0) {
        output.clear();
        nvtxRangePop();
        return;
    }

    std::vector<int> color_ranges;
    std::vector<GroupedDecoderConfig> decoder_configs;
    std::vector<int> gop_lens;
    std::vector<int> first_frame_ids;
    std::vector<std::vector<int>> packets_bytes;
    std::vector<std::vector<int>> decode_idxs;
    std::vector<const uint8_t*> packet_binary_data_ptrs;
    std::vector<size_t> packet_binary_data_sizes;

    color_ranges.reserve(num_groups);
    decoder_configs.reserve(num_groups);
    gop_lens.reserve(num_groups);
    first_frame_ids.reserve(num_groups);
    packets_bytes.reserve(num_groups);
    decode_idxs.reserve(num_groups);
    packet_binary_data_ptrs.reserve(num_groups);
    packet_binary_data_sizes.reserve(num_groups);

    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        const auto& target_ids = frame_id_groups[group_idx];
        if (target_ids.empty()) {
            nvtxRangePop();
            throw std::invalid_argument("[ERROR] frame_id_groups must not contain an empty group");
        }
        if (!std::is_sorted(target_ids.begin(), target_ids.end()) ||
            std::adjacent_find(target_ids.begin(), target_ids.end()) != target_ids.end()) {
            nvtxRangePop();
            throw std::invalid_argument(
                "[ERROR] frame IDs in each group must be strictly increasing and unique");
        }
        if (target_ids.front() < 0) {
            nvtxRangePop();
            throw std::invalid_argument("[ERROR] frame IDs must be non-negative");
        }

        std::vector<int> bundle_color_ranges;
        std::vector<int> bundle_codec_ids;
        std::vector<int> bundle_widths;
        std::vector<int> bundle_heights;
        std::vector<int> bundle_frame_sizes;
        std::vector<int> bundle_gop_lens;
        std::vector<int> bundle_first_frame_ids;
        std::vector<std::vector<int>> bundle_packets_bytes;
        std::vector<std::vector<int>> bundle_decode_idxs;
        std::vector<const uint8_t*> bundle_packet_binary_data_ptrs;
        std::vector<size_t> bundle_packet_binary_data_sizes;

        const uint32_t frames_in_bundle = parseSerializedPacketData(
            datas[group_idx], sizes[group_idx], bundle_color_ranges, bundle_codec_ids, bundle_widths,
            bundle_heights, bundle_frame_sizes, bundle_gop_lens, bundle_first_frame_ids, bundle_packets_bytes,
            bundle_decode_idxs, bundle_packet_binary_data_ptrs, bundle_packet_binary_data_sizes);

        if (frames_in_bundle != 1) {
            nvtxRangePop();
            throw std::invalid_argument("[ERROR] grouped GOP payload must contain exactly one block");
        }
        const int64_t first = bundle_first_frame_ids[0];
        const int64_t end = first + static_cast<int64_t>(bundle_gop_lens[0]);
        if (target_ids.front() < first || target_ids.back() >= end) {
            nvtxRangePop();
            throw std::invalid_argument("[ERROR] grouped GOP payload does not contain every target frame");
        }

        color_ranges.push_back(bundle_color_ranges[0]);
        decoder_configs.push_back(
            {bundle_codec_ids[0], bundle_widths[0], bundle_heights[0], bundle_frame_sizes[0]});
        gop_lens.push_back(bundle_gop_lens[0]);
        first_frame_ids.push_back(bundle_first_frame_ids[0]);
        packets_bytes.push_back(std::move(bundle_packets_bytes[0]));
        decode_idxs.push_back(std::move(bundle_decode_idxs[0]));
        packet_binary_data_ptrs.push_back(bundle_packet_binary_data_ptrs[0]);
        packet_binary_data_sizes.push_back(bundle_packet_binary_data_sizes[0]);
    }

    std::vector<size_t> decoder_slots;
    int status = AssignGroupedDecoderSlots(decoder_configs, decoder_slots);
    if (status != 0) {
        nvtxRangePop();
        throw std::runtime_error("[ERROR] AssignGroupedDecoderSlots failed");
    }

    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue(num_groups);
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        const size_t slot_idx = decoder_slots[group_idx];
        int skip_packets = 0;
        const int last_frame_id = last_decoded_frame_infos[slot_idx].frame_id;
        if (last_decoded_frame_infos[slot_idx].filename != source_names[group_idx] ||
            last_frame_id < first_frame_ids[group_idx] ||
            last_frame_id >= first_frame_ids[group_idx] + gop_lens[group_idx] ||
            last_frame_id >= frame_id_groups[group_idx].front()) {
            skip_packets = 0;
        } else {
            skip_packets = last_decoded_frame_infos[slot_idx].packet_id;
        }
        if (skip_packets == 0) {
            reset_last_decoded_frame_info(last_decoded_frame_infos[slot_idx]);
        }

        vpacket_queue[group_idx] = std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>();
        vpacket_queue[group_idx]->setSize(MAX_SIZE);

        size_t offset = 0;
        for (size_t packet_idx = 0; packet_idx < packets_bytes[group_idx].size(); ++packet_idx) {
            const int packet_bytes = packets_bytes[group_idx][packet_idx];
            int decode_idx = decode_idxs[group_idx][packet_idx];

            if (skip_packets > 0) {
                --skip_packets;
                if (packet_bytes > 0) {
                    offset += static_cast<size_t>(packet_bytes);
                }
                continue;
            }

            if (packet_bytes == -1) {
                vpacket_queue[group_idx]->push_back(std::make_tuple(nullptr, -1, 0));
            } else if (packet_bytes == 0) {
                vpacket_queue[group_idx]->push_back(std::make_tuple(nullptr, 0, 0));
            } else if (packet_bytes > 0) {
                if (offset + static_cast<size_t>(packet_bytes) > packet_binary_data_sizes[group_idx]) {
                    nvtxRangePop();
                    throw std::invalid_argument("[ERROR] GOP packet data is truncated");
                }
                uint8_t* video_data = const_cast<uint8_t*>(packet_binary_data_ptrs[group_idx] + offset);
                offset += static_cast<size_t>(packet_bytes);
                vpacket_queue[group_idx]->push_back(
                    std::make_tuple(video_data, packet_bytes, decode_idx * 2));
            } else {
                nvtxRangePop();
                throw std::invalid_argument("[ERROR] invalid negative GOP packet size");
            }
        }
    }

    status = main_decode_groups(color_ranges, decoder_configs, source_names, frame_id_groups, decoder_slots,
                                as_bgr, vpacket_queue, output);
    if (status != 0) {
        nvtxRangePop();
        throw std::runtime_error("[ERROR] main_decode_groups failed");
    }

    nvtxRangePop();
}

int PyNvGopDecoder::main_decode_groups(
    const std::vector<int>& color_ranges, const std::vector<GroupedDecoderConfig>& decoder_configs,
    const std::vector<std::string>& source_names, const std::vector<std::vector<int>>& frame_id_groups,
    const std::vector<size_t>& decoder_slots, bool as_bgr,
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
    std::vector<RGBFrame>& output) {
    ensureCudaContextInitialized();

    const size_t num_groups = frame_id_groups.size();
    if (decoder_configs.size() != num_groups || decoder_slots.size() != num_groups) {
        LOG(ERROR) << "Grouped decoder configuration or slot mapping size does not match group count";
        return -1;
    }
    std::unordered_set<size_t> unique_slots;
    for (const size_t slot_idx : decoder_slots) {
        if (slot_idx >= vdec.size() || !unique_slots.insert(slot_idx).second) {
            LOG(ERROR) << "Grouped decoder slot mapping contains an invalid or duplicate slot";
            return -1;
        }
    }
    size_t total_output_frames = 0;
    std::vector<int> output_widths;
    std::vector<int> output_heights;
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        total_output_frames += frame_id_groups[group_idx].size();
        output_widths.insert(output_widths.end(), frame_id_groups[group_idx].size(),
                             decoder_configs[group_idx].width);
        output_heights.insert(output_heights.end(), frame_id_groups[group_idx].size(),
                              decoder_configs[group_idx].height);
    }

    const std::vector<int> no_frame_sizes;
    int status = InitGpuMemPool(output_heights, output_widths, no_frame_sizes, true);
    if (status != 0) {
        LOG(ERROR) << "InitGpuMemPool failed for grouped GOP decode";
        return status;
    }
    std::vector<std::vector<uint8_t*>> flat_frame_buffers;
    status = GetFileFrameBuffers(&output_widths, &output_heights, &no_frame_sizes, true, flat_frame_buffers);
    if (status != 0) {
        LOG(ERROR) << "GetFileFrameBuffers failed for grouped GOP decode";
        return status;
    }

    std::vector<std::vector<uint8_t*>> group_frame_buffers(num_groups);
    size_t flat_buffer_idx = 0;
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        group_frame_buffers[group_idx].reserve(frame_id_groups[group_idx].size());
        for (size_t target_idx = 0; target_idx < frame_id_groups[group_idx].size(); ++target_idx) {
            group_frame_buffers[group_idx].push_back(flat_frame_buffers[flat_buffer_idx++][0]);
        }
    }

    std::vector<std::vector<RGBFrame>> rgb_frames(num_groups);

    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        try {
            const AVColorRange color_range = static_cast<AVColorRange>(color_ranges[group_idx]);
            WarnIfColorRangeUnspecified(color_range, source_names[group_idx]);
            rgb_frames[group_idx].reserve(frame_id_groups[group_idx].size());
        } catch (const std::exception& error) {
            force_join_all();
            LOG(ERROR) << "Grouped DecProc failed: " << error.what();
            return -1;
        }
    }

    try {
        decode_pool.run_indexed(num_groups, [&](size_t group_idx) {
            const size_t slot_idx = decoder_slots[group_idx];
            const AVColorRange color_range = static_cast<AVColorRange>(color_ranges[group_idx]);
            DecProc<RGBFrame>(color_range, vdec[slot_idx].get(), rgb_frames[group_idx],
                              group_frame_buffers[group_idx], vpacket_queue[group_idx].get(),
                              frame_id_groups[group_idx], as_bgr, source_names[group_idx],
                              last_decoded_frame_infos[slot_idx]);
        });
    } catch (const std::exception& error) {
        force_join_all();
        LOG(ERROR) << "Grouped DecProc failed: " << error.what();
        return -1;
    }

    output.clear();
    output.reserve(total_output_frames);
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        if (rgb_frames[group_idx].size() != frame_id_groups[group_idx].size()) {
            force_join_all();
            LOG(ERROR) << "Grouped RGB decode produced " << rgb_frames[group_idx].size()
                       << " frames, expected " << frame_id_groups[group_idx].size() << " for group "
                       << group_idx;
            return -1;
        }
        for (auto& frame : rgb_frames[group_idx]) {
            output.push_back(std::move(frame));
        }
    }

    // DecProc queues color conversion on cu_stream. Waiting for its CPU worker only
    // guarantees that the work was submitted; the output frames are ready when
    // this public synchronous API returns only after the stream is synchronized.
    CUDA_DRVAPI_CALL(cuStreamSynchronize(this->cu_stream));
    return 0;
}

int PyNvGopDecoder::main_decode(
    const std::vector<int>& color_ranges, const std::vector<int>& codec_ids, std::vector<int>& widths,
    std::vector<int>& heights, std::vector<int>& frame_sizes, const std::vector<std::string>& filepaths,
    const std::vector<int>& frame_ids, bool convert_to_rgb, bool as_bgr,
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
    std::vector<DecodedFrameExt>* out_if_no_color_conversion, std::vector<RGBFrame>* out_if_color_converted,
    bool skip_final_sync) {
    // start decoding process
    int st = 0;

    // lazy loading
    ensureCudaContextInitialized();

    st = InitGpuMemPool(heights, widths, frame_sizes, convert_to_rgb);
    if (st != 0) {
        LOG(ERROR) << "InitGpuMemPool failed.";
        return st;
    }

    st = InitializeDecoders(codec_ids);
    if (st != 0) {
        LOG(ERROR) << "InitializeDecoders failed.";
        return st;
    }

    const int total_frames = static_cast<int>(frame_ids.size());

    std::vector<std::vector<uint8_t*>> per_file_frame_buffers;
    st = GetFileFrameBuffers(&widths, &heights, &frame_sizes, convert_to_rgb, per_file_frame_buffers);
    if (st != 0) {
        LOG(ERROR) << "GetFileFrameBuffers failed.";
        return st;
    }

    std::vector<std::vector<DecodedFrameExt>> decodedFrames;
    std::vector<std::vector<RGBFrame>> rgb_frames;
    if (convert_to_rgb) {
        rgb_frames.resize(total_frames);
        out_if_color_converted->resize(total_frames);
    } else {
        decodedFrames.resize(total_frames);
        out_if_no_color_conversion->resize(total_frames);
    }

    // decoding process
    for (int i = 0; i < total_frames; ++i) {
        try {
            std::vector<int> sorted_frame_ids = {frame_ids[i]};
            if (convert_to_rgb) {
                WarnIfColorRangeUnspecified(static_cast<AVColorRange>(color_ranges[i]), filepaths[i]);
                rgb_frames[i].reserve(sorted_frame_ids.size());
            } else {
                decodedFrames[i].reserve(sorted_frame_ids.size());
            }
        } catch (const std::exception& e) {
            this->force_join_all();
            LOG(ERROR) << "DecProc failed: " << e.what();
            return -1;
        }
    }

    try {
        decode_pool.run_indexed(total_frames, [&](size_t index) {
            std::vector<int> sorted_frame_ids = {frame_ids[index]};
            AVColorRange color_range = static_cast<AVColorRange>(color_ranges[index]);
            if (convert_to_rgb) {
                DecProc<RGBFrame>(color_range, this->vdec[index].get(), rgb_frames[index],
                                  per_file_frame_buffers[index], vpacket_queue[index].get(), sorted_frame_ids,
                                  as_bgr, filepaths[index], this->last_decoded_frame_infos[index]);
            } else {
                DecProc<DecodedFrameExt>(color_range, this->vdec[index].get(), decodedFrames[index],
                                         per_file_frame_buffers[index], vpacket_queue[index].get(),
                                         sorted_frame_ids, false, filepaths[index],
                                         this->last_decoded_frame_infos[index]);
            }
        });
    } catch (const std::exception& e) {
        this->force_join_all();
        LOG(ERROR) << "DecProc failed: " << e.what();
        return -1;
    }

    for (int i = 0; i < total_frames; ++i) {
        if (convert_to_rgb) {
            if (rgb_frames[i].empty()) {
                this->force_join_all();
                LOG(ERROR) << "rgb_frames[i] is empty";
                return -1;
            }
            (*out_if_color_converted)[i] = std::move(rgb_frames[i][0]);
        } else {
            if (decodedFrames[i].empty()) {
                this->force_join_all();
                LOG(ERROR) << "decodedFrames[i] is empty";
                return -1;
            }
            (*out_if_no_color_conversion)[i] = std::move(decodedFrames[i][0]);
        }
    }

    // Single sync on this->cu_stream covers all NvDecoders because every NvDecoder in vdec[]
    // is constructed with this->cu_stream (see PyNvGopDecoder_common.cpp).  If any NvDecoder
    // were ever constructed with a different stream, this sync would not cover its GPU writes
    // and callers could observe incomplete frames.
    if (!skip_final_sync) {
        CUDA_DRVAPI_CALL(cuStreamSynchronize(this->cu_stream));
    }
    return 0;
}

SerializedPacketBundle PyNvGopDecoder::createSerializedPacketBundle(
    size_t total_frames, const std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
    const std::vector<std::vector<int>>& all_gop_lens,
    const std::vector<std::vector<int>>& all_first_frame_ids,
    const std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
    const std::vector<std::vector<std::unique_ptr<uint8_t[]>>>& vpacket_array) {
    nvtxRangePushA("Direct Serialization");

    /*
     * Binary Data Structure Layout:
     * =============================
     * 
     * Header Section:
     * - uint32_t: total_frames (number of frames)
     * - size_t[total_frames]: frame_offsets (offset table for direct frame access)
     * 
     * Frame Data Section (repeated total_frames times):
     * 
     * 1. Metadata (7 * int32_t = 28 bytes):
     *    - int32_t: color_range (video color range)
     *    - int32_t: codec_id (NVIDIA codec identifier)
     *    - int32_t: width (frame width in pixels)
     *    - int32_t: height (frame height in pixels)
     *    - int32_t: frame_size (size of decoded frame in bytes)
     *    - int32_t: gop_len (number of frames in this GOP)
     *    - int32_t: first_frame_id (first frame ID in this GOP)
     * 
     * 2. Packets bytes array:
     *    - uint32_t: packets_bytes_count (number of packet sizes)
     *    - int32_t[]: packet_sizes (size of each packet in bytes)
     * 
     * 3. Decode indices array:
     *    - uint32_t: decode_idxs_count (number of decode indices)
     *    - int32_t[]: decode_indices (decode index for each packet)
     * 
     * 4. Binary packet data:
     *    - uint64_t: binary_data_size (total size of binary data)
     *    - uint8_t[]: packet_binary_data (concatenated packet data)
     * 
     * Total size calculation:
     * - Header: sizeof(uint32_t) + total_frames * sizeof(size_t)
     * - Per frame: 28 bytes + packets_array + decode_array + binary_data
     * 
     * Frame Access Pattern:
     * ====================
     * 1. Read total_frames from offset 0
     * 2. Read frame_offsets[i] from offset (sizeof(uint32_t) + i * sizeof(size_t))
     * 3. Jump to frame_offsets[i] to access frame i's data directly
     * 
     * This design enables:
     * - O(1) random access to any frame
     * - Parallel processing without data copying
     * - Self-contained binary format (no external metadata needed)
     */

    // Calculate total size: header + offset table + frame data
    size_t total_size = sizeof(uint32_t) + total_frames * sizeof(size_t);  // Header + offset table

    std::vector<std::vector<int>> packets_bytes_temp(total_frames);
    std::vector<std::vector<int>> decode_idxs_temp(total_frames);
    std::vector<std::vector<uint8_t>> packet_binary_data_temp(total_frames);

    // Step 1: Extract packet information and prepare temporary data
    for (int i = 0; i < total_frames; ++i) {
        // Extract packet sizes and decode indices from packet queue
        while (!vpacket_queue[i]->empty()) {
            auto packet = vpacket_queue[i]->pop_front();
            packets_bytes_temp[i].push_back(std::get<1>(packet));    // packet size
            decode_idxs_temp[i].push_back(std::get<2>(packet) / 2);  // decode index (divided by 2)
        }

        // Calculate total binary data size for this frame
        size_t binary_data_size = 0;
        int num_packets = vpacket_array.at(i).size();
        for (int j = 0; j < num_packets; ++j) {
            binary_data_size += packets_bytes_temp.at(i).at(j);
        }

        // Copy all packet binary data into a contiguous buffer
        packet_binary_data_temp[i].resize(binary_data_size);
        size_t offset = 0;
        for (size_t j = 0; j < num_packets; ++j) {
            size_t packet_size = packets_bytes_temp.at(i).at(j);
            std::memcpy(packet_binary_data_temp.at(i).data() + offset, vpacket_array.at(i).at(j).get(),
                        packet_size);
            offset += packet_size;
        }

        // Calculate the size contribution of this frame to the total serialized data
        total_size += 7 * sizeof(int32_t);  // metadata (7 fields)
        total_size +=
            sizeof(uint32_t) + packets_bytes_temp[i].size() * sizeof(int32_t);          // packets_bytes array
        total_size += sizeof(uint32_t) + decode_idxs_temp[i].size() * sizeof(int32_t);  // decode_idxs array
        total_size += sizeof(uint64_t) + packet_binary_data_temp[i].size();             // binary_data block
    }

    // Step 2: Allocate memory buffer and serialize data
    SerializedPacketBundle result;
    result.data = std::make_unique<uint8_t[]>(total_size);
    result.size = total_size;

    uint8_t* ptr = result.data.get();  // Current write position in the buffer

    // Phase 1: Calculate frame offsets
    std::vector<size_t> frame_offsets_temp;
    frame_offsets_temp.reserve(total_frames);

    // Calculate where each frame's data will start
    size_t current_offset =
        sizeof(uint32_t) + total_frames * sizeof(size_t);  // After header and offset table
    for (int i = 0; i < total_frames; ++i) {
        frame_offsets_temp.push_back(current_offset);

        // Calculate this frame's data size
        size_t frame_data_size = 7 * sizeof(int32_t);  // metadata
        frame_data_size +=
            sizeof(uint32_t) + packets_bytes_temp[i].size() * sizeof(int32_t);               // packets_bytes
        frame_data_size += sizeof(uint32_t) + decode_idxs_temp[i].size() * sizeof(int32_t);  // decode_idxs
        frame_data_size += sizeof(uint64_t) + packet_binary_data_temp[i].size();             // binary_data

        current_offset += frame_data_size;
    }

    // Phase 2: Write binary data
    // Write header: total number of frames
    *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(total_frames);
    ptr += sizeof(uint32_t);

    // Write offset table
    for (int i = 0; i < total_frames; ++i) {
        *reinterpret_cast<size_t*>(ptr) = frame_offsets_temp[i];
        ptr += sizeof(size_t);
    }

    // Write frame data for each frame
    for (int i = 0; i < total_frames; ++i) {
        // 1. Write metadata (7 fields, 28 bytes total)
        *reinterpret_cast<int32_t*>(ptr) = static_cast<int>(demuxers[i]->GetColorRange());
        ptr += sizeof(int32_t);  // color_range
        *reinterpret_cast<int32_t*>(ptr) = demuxers[i]->GetNvCodecId();
        ptr += sizeof(int32_t);  // codec_id
        *reinterpret_cast<int32_t*>(ptr) = demuxers[i]->GetWidth();
        ptr += sizeof(int32_t);  // width
        *reinterpret_cast<int32_t*>(ptr) = demuxers[i]->GetHeight();
        ptr += sizeof(int32_t);  // height
        *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(frame_output::frame_bytes(
            frame_output::output_format_from_av_pixel_format(demuxers[i]->GetPixelFormat()),
            static_cast<size_t>(demuxers[i]->GetHeight()), static_cast<size_t>(demuxers[i]->GetWidth())));
        ptr += sizeof(int32_t);  // frame_size
        *reinterpret_cast<int32_t*>(ptr) = all_gop_lens[i][0];
        ptr += sizeof(int32_t);  // gop_len
        *reinterpret_cast<int32_t*>(ptr) = all_first_frame_ids[i][0];
        ptr += sizeof(int32_t);  // first_frame_id

        // 2. Write packets_bytes array (packet sizes)
        const auto& packets_bytes = packets_bytes_temp[i];
        *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(packets_bytes.size());  // array count
        ptr += sizeof(uint32_t);
        std::memcpy(ptr, packets_bytes.data(), packets_bytes.size() * sizeof(int32_t));  // array data
        ptr += packets_bytes.size() * sizeof(int32_t);

        // 3. Write decode_idxs array (decode indices)
        const auto& decode_idxs = decode_idxs_temp[i];
        *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(decode_idxs.size());  // array count
        ptr += sizeof(uint32_t);
        std::memcpy(ptr, decode_idxs.data(), decode_idxs.size() * sizeof(int32_t));  // array data
        ptr += decode_idxs.size() * sizeof(int32_t);

        // 4. Write packet binary data (concatenated packet contents)
        const auto& binary_data = packet_binary_data_temp[i];
        *reinterpret_cast<uint64_t*>(ptr) = static_cast<uint64_t>(binary_data.size());  // data size
        ptr += sizeof(uint64_t);
        if (!binary_data.empty()) {
            std::memcpy(ptr, binary_data.data(), binary_data.size());  // binary data
            ptr += binary_data.size();
        }
    }

#ifdef IS_DEBUG_BUILD
    // Print debug information about the serialized data
    printf("\n=== createSerializedPacketBundle Debug Info ===\n");
    printf("Total frames: %zu\n", total_frames);
    printf("Total serialized size: %zu bytes\n", total_size);
    printf("Header size: %zu bytes\n", sizeof(uint32_t) + total_frames * sizeof(size_t));
    printf("\n");

    for (int i = 0; i < total_frames; ++i) {
        printf("--- Frame %d ---\n", i);

        // Print metadata
        printf("Metadata (7 fields):\n");
        printf("  color_range: %d\n", static_cast<int>(demuxers[i]->GetColorRange()));
        printf("  codec_id: %d\n", demuxers[i]->GetNvCodecId());
        printf("  width: %d\n", demuxers[i]->GetWidth());
        printf("  height: %d\n", demuxers[i]->GetHeight());
        printf(
            "  frame_size: %zu\n",
            frame_output::frame_bytes(
                frame_output::output_format_from_av_pixel_format(demuxers[i]->GetPixelFormat()),
                static_cast<size_t>(demuxers[i]->GetHeight()), static_cast<size_t>(demuxers[i]->GetWidth())));
        printf("  gop_len: %d\n", all_gop_lens[i][0]);
        printf("  first_frame_id: %d\n", all_first_frame_ids[i][0]);

        // Print packets_bytes array (first 100 elements)
        printf("packets_bytes array (size: %zu):\n", packets_bytes_temp[i].size());
        printf("  First 100 elements: ");
        for (size_t j = 0; j < std::min(packets_bytes_temp[i].size(), size_t(100)); ++j) {
            printf("%d ", packets_bytes_temp[i][j]);
        }
        if (packets_bytes_temp[i].size() > 100) {
            printf("... (total: %zu elements)", packets_bytes_temp[i].size());
        }
        printf("\n");

        // Print decode_idxs array (first 100 elements)
        printf("decode_idxs array (size: %zu):\n", decode_idxs_temp[i].size());
        printf("  First 100 elements: ");
        for (size_t j = 0; j < std::min(decode_idxs_temp[i].size(), size_t(100)); ++j) {
            printf("%d ", decode_idxs_temp[i][j]);
        }
        if (decode_idxs_temp[i].size() > 100) {
            printf("... (total: %zu elements)", decode_idxs_temp[i].size());
        }
        printf("\n");

        // Print packet_binary_data (each packet's first 20 and last 20 bytes)
        printf("packet_binary_data (size: %zu bytes):\n", packet_binary_data_temp[i].size());

        size_t offset = 0;
        for (size_t packet_idx = 0; packet_idx < packets_bytes_temp[i].size(); ++packet_idx) {
            int packet_size = packets_bytes_temp[i][packet_idx];

            if (packet_size > 0) {  // Skip special markers (-1, 0)
                printf("  Packet %zu (size: %d bytes):\n", packet_idx, packet_size);

                // Save packet to binary file
                std::string packet_filename =
                    "frame_" + std::to_string(i) + "_packet_" + std::to_string(packet_idx) + "_ref.bin";
                std::ofstream packet_file(packet_filename, std::ios::binary);
                if (packet_file.is_open()) {
                    packet_file.write(
                        reinterpret_cast<const char*>(packet_binary_data_temp[i].data() + offset),
                        packet_size);
                    packet_file.close();
                    printf("    Saved to: %s\n", packet_filename.c_str());
                } else {
                    printf("    Failed to save packet to file\n");
                }

                // Print first 20 bytes of this packet
                printf("    First 20 bytes: ");
                size_t first_bytes = std::min(static_cast<size_t>(packet_size), size_t(20));
                for (size_t j = 0; j < first_bytes; ++j) {
                    printf("%02X ", packet_binary_data_temp[i][offset + j]);
                }
                printf("\n");

                // Print last 20 bytes of this packet (if packet is larger than 20 bytes)
                if (packet_size > 20) {
                    printf("    Last 20 bytes:  ");
                    size_t last_start = offset + packet_size - 20;
                    for (size_t j = 0; j < 20; ++j) {
                        printf("%02X ", packet_binary_data_temp[i][last_start + j]);
                    }
                    printf("\n");
                }

                offset += packet_size;
            } else {
                printf("  Packet %zu: Special marker (%d)\n", packet_idx, packet_size);
            }
        }
        printf("\n");
    }

    printf("=== Serialization Complete ===\n");
    printf("Total serialized size: %zu bytes\n", total_size);
    printf("================================\n\n");
#endif  // DEBUG

    nvtxRangePop();  // Direct Serialization

    // Step 3: Fill metadata vectors in the result structure for easy access
    // Note: frame offsets are now embedded in the binary data header
    result.gop_lens.reserve(total_frames);
    result.first_frame_ids.reserve(total_frames);
    for (int i = 0; i < total_frames; ++i) {
        result.gop_lens.push_back(all_gop_lens[i][0]);                // GOP length for each frame
        result.first_frame_ids.push_back(all_first_frame_ids[i][0]);  // First frame ID for each GOP
    }

    return result;
}

uint32_t PyNvGopDecoder::parseSerializedPacketData(
    const uint8_t* data, size_t size, std::vector<int>& color_ranges, std::vector<int>& codec_ids,
    std::vector<int>& widths, std::vector<int>& heights, std::vector<int>& frame_sizes,
    std::vector<int>& gop_lens, std::vector<int>& first_frame_ids,
    std::vector<std::vector<int>>& packets_bytes, std::vector<std::vector<int>>& decode_idxs,
    std::vector<const uint8_t*>& packet_binary_data_ptrs, std::vector<size_t>& packet_binary_data_sizes) {
    /*
     * Parse self-contained binary data using embedded offset table.
     * This is the counterpart to createSerializedPacketBundle() and handles
     * the new format with embedded offsets for efficient random access.
     * 
     * Binary format being parsed:
     * - Header: total_frames + frame_offsets[]
     * - Frame data blocks at the specified offsets
     */

    if (size < sizeof(uint32_t)) {
        throw std::invalid_argument("[ERROR] Binary data too small to contain header");
    }

    const uint8_t* ptr = data;

    // Read total_frames from header
    uint32_t total_frames = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);

    if (total_frames == 0) {
        return 0;
    }

    // Validate header size
    size_t expected_header_size = sizeof(uint32_t) + total_frames * sizeof(size_t);
    if (size < expected_header_size) {
        throw std::invalid_argument("[ERROR] Binary data too small to contain offset table");
    }

    // Read frame offsets from embedded offset table
    std::vector<size_t> frame_offsets(total_frames);
    for (uint32_t i = 0; i < total_frames; ++i) {
        frame_offsets[i] = *reinterpret_cast<const size_t*>(ptr);
        ptr += sizeof(size_t);
    }

    // Resize output vectors
    color_ranges.resize(total_frames);
    codec_ids.resize(total_frames);
    widths.resize(total_frames);
    heights.resize(total_frames);
    frame_sizes.resize(total_frames);
    gop_lens.resize(total_frames);
    first_frame_ids.resize(total_frames);
    packets_bytes.resize(total_frames);
    decode_idxs.resize(total_frames);
    packet_binary_data_ptrs.resize(total_frames);
    packet_binary_data_sizes.resize(total_frames);

    // Parse each frame using its offset (enables parallel processing)
    for (uint32_t i = 0; i < total_frames; ++i) {
        // Validate frame offset
        if (frame_offsets[i] >= size) {
            throw std::out_of_range("[ERROR] Frame " + std::to_string(i) + " offset " +
                                    std::to_string(frame_offsets[i]) + " exceeds data size " +
                                    std::to_string(size));
        }

        // Jump to frame data using offset
        const uint8_t* frame_ptr = data + frame_offsets[i];

        // Parse frame metadata (7 fields, 28 bytes total)
        color_ranges[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        codec_ids[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        widths[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        heights[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        frame_sizes[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        gop_lens[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);
        first_frame_ids[i] = *reinterpret_cast<const int32_t*>(frame_ptr);
        frame_ptr += sizeof(int32_t);

        // Parse packets_bytes array
        const uint32_t packets_bytes_size = *reinterpret_cast<const uint32_t*>(frame_ptr);
        frame_ptr += sizeof(uint32_t);
        packets_bytes[i].resize(packets_bytes_size);
        if (packets_bytes_size > 0) {
            std::memcpy(packets_bytes[i].data(), frame_ptr, packets_bytes_size * sizeof(int32_t));
            frame_ptr += packets_bytes_size * sizeof(int32_t);
        }

        // Parse decode_idxs array
        const uint32_t decode_idxs_size = *reinterpret_cast<const uint32_t*>(frame_ptr);
        frame_ptr += sizeof(uint32_t);
        decode_idxs[i].resize(decode_idxs_size);
        if (decode_idxs_size > 0) {
            std::memcpy(decode_idxs[i].data(), frame_ptr, decode_idxs_size * sizeof(int32_t));
            frame_ptr += decode_idxs_size * sizeof(int32_t);
        }

        // Parse packet binary data information (store pointer and size)
        const uint64_t binary_data_size = *reinterpret_cast<const uint64_t*>(frame_ptr);
        frame_ptr += sizeof(uint64_t);
        packet_binary_data_ptrs[i] = frame_ptr;
        packet_binary_data_sizes[i] = binary_data_size;
        // Note: frame_ptr is not advanced here as we only store the pointer
    }

    return total_frames;
}

void SaveBinaryDataToFile(const uint8_t* data, size_t size, const std::string& dst_filepath) {
    nvtxRangePushA("SaveBinaryDataToFile");

    try {
        if (data == nullptr) {
            throw std::invalid_argument("[ERROR] data is null");
        }

        if (size == 0) {
            throw std::invalid_argument("[ERROR] data is empty");
        }

        // Create output directory if it doesn't exist
        std::filesystem::path output_path(dst_filepath);
        std::filesystem::path parent_dir = output_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
            if (!std::filesystem::create_directories(parent_dir)) {
                throw std::runtime_error("[ERROR] Failed to create output directory: " + parent_dir.string());
            }
        }

        // Open file for binary writing
        std::ofstream file(dst_filepath, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("[ERROR] Failed to open file for writing: " + dst_filepath);
        }

        // Write data to file
        file.write(reinterpret_cast<const char*>(data), size);
        if (file.fail()) {
            throw std::runtime_error("[ERROR] Failed to write data to file: " + dst_filepath);
        }

        file.close();

        // Verify file was written correctly
        if (!std::filesystem::exists(dst_filepath)) {
            throw std::runtime_error("[ERROR] File was not created: " + dst_filepath);
        }

        // Verify file size matches expected size
        std::error_code ec;
        size_t file_size = std::filesystem::file_size(dst_filepath, ec);
        if (ec) {
            throw std::runtime_error("[ERROR] Failed to get file size: " + dst_filepath);
        }

        if (file_size != size) {
            throw std::runtime_error("[ERROR] File size mismatch. Expected: " + std::to_string(size) +
                                     ", Got: " + std::to_string(file_size));
        }

    } catch (const std::exception& e) {
        std::string error_msg = "[ERROR] SaveBinaryDataToFile failed: ";
        error_msg += e.what();
        throw std::runtime_error(error_msg);
    }

    nvtxRangePop();
}

void PyNvGopDecoder::LoadGOPFromFiles(const std::vector<std::string>& file_paths,
                                      std::vector<std::vector<uint8_t>>& file_data_buffers) {
    nvtxRangePushA("LoadGOPFromFiles");

    if (file_paths.empty()) {
        throw std::invalid_argument("[ERROR] file_paths is empty");
    }

    // Read all binary files in parallel
    file_data_buffers.resize(file_paths.size());

    parallel_pool.run_indexed(file_paths.size(), [&](size_t file_idx) {
        const auto& file_path = file_paths[file_idx];

        // Check if file exists
        if (!std::filesystem::exists(file_path)) {
            throw std::runtime_error("[ERROR] File does not exist: " + file_path);
        }

        // Read entire file into memory
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("[ERROR] Failed to open file: " + file_path);
        }

        const std::streampos end_position = file.tellg();
        if (end_position < 0) {
            throw std::runtime_error("[ERROR] Failed to determine file size: " + file_path);
        }
        const size_t file_size = static_cast<size_t>(end_position);
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> file_buffer(file_size);
        file.read(reinterpret_cast<char*>(file_buffer.data()), file_size);
        if (file.fail()) {
            throw std::runtime_error("[ERROR] Failed to read file: " + file_path);
        }
        file.close();

        // Validate the complete bundle before exposing it to Python.
        std::vector<int> color_ranges;
        std::vector<int> codec_ids;
        std::vector<int> widths;
        std::vector<int> heights;
        std::vector<int> frame_sizes;
        std::vector<int> gop_lens;
        std::vector<int> first_frame_ids;
        std::vector<std::vector<int>> packets_bytes;
        std::vector<std::vector<int>> decode_idxs;
        std::vector<const uint8_t*> packet_binary_data_ptrs;
        std::vector<size_t> packet_binary_data_sizes;
        const uint32_t frame_count =
            parseSerializedPacketData(file_buffer.data(), file_buffer.size(), color_ranges, codec_ids, widths,
                                      heights, frame_sizes, gop_lens, first_frame_ids, packets_bytes,
                                      decode_idxs, packet_binary_data_ptrs, packet_binary_data_sizes);
        if (frame_count == 0) {
            throw std::invalid_argument("[ERROR] File contains no frames: " + file_path);
        }

        // Store file data
        file_data_buffers[file_idx] = std::move(file_buffer);
    });

    nvtxRangePop();
}
