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

    // lazy loading
    ensureDemuxRunnersInitialized();

    // Initialize demuxers
    st = InitializeDemuxers(filepaths, demuxers, fastStreamInfos);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitializeDemuxers failed.");
    }

    // Initialize GOP metadata containers
    all_gop_lens.resize(total_frames);
    all_first_frame_ids.resize(total_frames);

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
#ifdef PROCESS_SYNC
            DemuxGopProc(demuxers[i].get(), vpacket_queue[i].get(), sorted_frame_ids, all_first_frame_ids[i],
                         all_gop_lens[i], vpacket_array[i], true);
#else
            demux_runners[i].join();
            demux_runners[i].start(PyNvGopDecoder::DemuxGopProc, demuxers[i].get(), vpacket_queue[i].get(),
                                   sorted_frame_ids, std::ref(all_first_frame_ids[i]),
                                   std::ref(all_gop_lens[i]), std::ref(vpacket_array[i]), true);
#endif
        } catch (const std::exception& e) {
            this->force_join_all();
            LOG(ERROR) << "Packet extraction failed: " << e.what();
            throw std::runtime_error(e.what());
        }
    }
    nvtxRangePop();  // Packet extraction

    // Wait for all demux threads to complete
    nvtxRangePushA("Demux thread join");
    try {
        for (int i = 0; i < total_frames; ++i) {
#ifndef PROCESS_SYNC
            demux_runners[i].join();
#endif
        }
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

SerializedPacketBundle PyNvGopDecoder::get_gop(const std::vector<std::string>& filepaths,
                                               const std::vector<int> frame_ids,
                                               const FastStreamInfo* fastStreamInfos) {
    nvtxRangePushA("GetGOP");
    const size_t total_frames = frame_ids.size();

    // Use internal implementation to extract GOP data
    std::vector<std::unique_ptr<PyNvGopDemuxer>> demuxers;
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    std::vector<std::vector<std::unique_ptr<uint8_t[]>>> vpacket_array;
    std::vector<std::vector<int>> all_gop_lens;
    std::vector<std::vector<int>> all_first_frame_ids;

    get_gop_internal(filepaths, frame_ids, fastStreamInfos, demuxers, vpacket_queue, vpacket_array,
                     all_gop_lens, all_first_frame_ids);

    // Merge all video data into a single bundle
    nvtxRangePushA("CreateSerializedPacketBundle");
    SerializedPacketBundle result = createSerializedPacketBundle(
        total_frames, demuxers, all_gop_lens, all_first_frame_ids, vpacket_queue, vpacket_array);
    nvtxRangePop();  // CreateSerializedPacketBundle

    nvtxRangePop();  // GetGOP
    return result;
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
    // ensureDemuxRunnersInitialized();
    ensureDecodeRunnersInitialized();

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
                                          const std::vector<int> frame_ids, bool as_bgr,
                                          std::vector<RGBFrame>* dst) {
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

    // Decode to RGB using unified path
    std::vector<RGBFrame> output_frames;
    output_frames.resize(total_frames);

    int st = main_decode(color_ranges_all, codec_ids_all, widths_all, heights_all, frame_sizes_all, filepaths,
                         frame_ids, /*convert_to_rgb=*/true, as_bgr, vpacket_queue,
                         /*out_if_no_color_conversion=*/nullptr,
                         /*out_if_color_converted=*/dst);
    if (st != 0) {
        nvtxRangePop();
        throw std::runtime_error("[ERROR] main_decode failed.");
    }

    nvtxRangePop();
}

int PyNvGopDecoder::main_decode(
    const std::vector<int>& color_ranges, const std::vector<int>& codec_ids, std::vector<int>& widths,
    std::vector<int>& heights, std::vector<int>& frame_sizes, const std::vector<std::string>& filepaths,
    const std::vector<int>& frame_ids, bool convert_to_rgb, bool as_bgr,
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
    std::vector<DecodedFrameExt>* out_if_no_color_conversion, std::vector<RGBFrame>* out_if_color_converted) {
    // start decoding process
    int st = 0;

    // lazy loading
    ensureCudaContextInitialized();
    ensureDecodeRunnersInitialized();

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
                rgb_frames[i].reserve(sorted_frame_ids.size());
            } else {
                decodedFrames[i].reserve(sorted_frame_ids.size());
            }
            auto& packet_queue = vpacket_queue[i];
            auto& frame_buffers = per_file_frame_buffers[i];
            AVColorRange color_range = static_cast<AVColorRange>(color_ranges[i]);
#ifdef PROCESS_SYNC
            if (convert_to_rgb) {
                DecProc<RGBFrame>(color_range, this->vdec[i].get(), rgb_frames[i], frame_buffers,
                                  packet_queue.get(), sorted_frame_ids, as_bgr, filepaths[i],
                                  this->last_decoded_frame_infos[i]);
            } else {
                DecProc<DecodedFrameExt>(color_range, this->vdec[i].get(), decodedFrames[i], frame_buffers,
                                         packet_queue.get(), sorted_frame_ids, false, filepaths[i],
                                         this->last_decoded_frame_infos[i]);
            }
#else
            if (convert_to_rgb) {
                decode_runners[i].join();
                decode_runners[i].start(PyNvGopDecoder::DecProc<RGBFrame>, color_range, this->vdec[i].get(),
                                        std::ref(rgb_frames[i]), frame_buffers, packet_queue.get(),
                                        sorted_frame_ids, as_bgr, filepaths[i],
                                        std::ref(this->last_decoded_frame_infos[i]));
            } else {
                decode_runners[i].join();
                decode_runners[i].start(PyNvGopDecoder::DecProc<DecodedFrameExt>, color_range,
                                        this->vdec[i].get(), std::ref(decodedFrames[i]), frame_buffers,
                                        packet_queue.get(), sorted_frame_ids, false, filepaths[i],
                                        std::ref(this->last_decoded_frame_infos[i]));
            }
#endif
        } catch (const std::exception& e) {
            this->force_join_all();
            LOG(ERROR) << "DecProc failed: " << e.what();
            return -1;
        }
    }

    for (int i = 0; i < total_frames; ++i) {
#ifndef PROCESS_SYNC
        // sync all decoders
        for (int i = 0; i < total_frames; ++i) {
            decode_runners[i].join();
        }
#endif
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
        *reinterpret_cast<int32_t*>(ptr) = demuxers[i]->GetFrameSize();
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
        printf("  frame_size: %d\n", demuxers[i]->GetFrameSize());
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

void PyNvGopDecoder::MergePacketDataToOne(const std::vector<uint8_t*>& packet_data_buffers,
                                          const std::vector<size_t>& buffer_sizes,
                                          std::unique_ptr<uint8_t[]>& merged_data, size_t& merged_size) {
    nvtxRangePushA("MergePacketDataToOne");

    try {
        if (packet_data_buffers.empty()) {
            throw std::invalid_argument("[ERROR] packet_data_buffers is empty");
        }

        if (buffer_sizes.size() != packet_data_buffers.size()) {
            throw std::invalid_argument("[ERROR] buffer_sizes size does not match packet_data_buffers size");
        }

        // Ensure merge thread pool is initialized
        ensureMergeRunnersInitialized();

        // Calculate number of threads to use
        size_t num_threads = std::min(packet_data_buffers.size(), merge_runners.size());

        // Helper lambda for parallel execution with exception handling
        auto executeInParallel = [&](const std::function<void(size_t)>& task_func) {
            std::vector<std::exception_ptr> exceptions(num_threads);

            // Start parallel tasks
            for (size_t i = 0; i < num_threads; ++i) {
                merge_runners[i].start([&, i]() {
                    try {
                        task_func(i);
                    } catch (...) {
                        exceptions[i] = std::current_exception();
                    }
                });
            }

            // Wait for all tasks to complete
            for (size_t i = 0; i < num_threads; ++i) {
                merge_runners[i].join();
            }

            // Check for exceptions
            for (auto& ex : exceptions) {
                if (ex) {
                    std::rethrow_exception(ex);
                }
            }
        };

        // Step 1: Parse all binary data buffers in parallel and collect frame data
        std::vector<uint32_t> buffer_frame_counts(packet_data_buffers.size());
        std::vector<std::vector<size_t>> buffer_frame_offsets(packet_data_buffers.size());
        uint32_t total_merged_frames = 0;

        executeInParallel([&](size_t thread_id) {
            // Process buffers assigned to this thread
            for (size_t buffer_idx = thread_id; buffer_idx < packet_data_buffers.size();
                 buffer_idx += num_threads) {
                const uint8_t* buffer = packet_data_buffers[buffer_idx];
                size_t buffer_size = buffer_sizes[buffer_idx];

                // Parse buffer header to get frame count and offsets
                if (buffer_size < sizeof(uint32_t)) {
                    throw std::invalid_argument("[ERROR] Buffer too small at index: " +
                                                std::to_string(buffer_idx));
                }

                const uint8_t* data_ptr = buffer;
                uint32_t frame_count = *reinterpret_cast<const uint32_t*>(data_ptr);
                data_ptr += sizeof(uint32_t);

                if (frame_count == 0) {
                    throw std::invalid_argument("[ERROR] Buffer contains no frames at index: " +
                                                std::to_string(buffer_idx));
                }

                // Validate header size
                size_t expected_header_size = sizeof(uint32_t) + frame_count * sizeof(size_t);
                if (buffer_size < expected_header_size) {
                    throw std::invalid_argument("[ERROR] Buffer header invalid at index: " +
                                                std::to_string(buffer_idx));
                }

                // Read frame offsets
                std::vector<size_t> frame_offsets(frame_count);
                for (uint32_t j = 0; j < frame_count; ++j) {
                    frame_offsets[j] = *reinterpret_cast<const size_t*>(data_ptr);
                    data_ptr += sizeof(size_t);
                }

                // Store buffer data
                buffer_frame_counts[buffer_idx] = frame_count;
                buffer_frame_offsets[buffer_idx] = std::move(frame_offsets);
            }
        });

        // Calculate total frames
        for (auto frame_count : buffer_frame_counts) {
            total_merged_frames += frame_count;
        }

        // Step 2: Calculate merged data size and frame offsets
        size_t merged_header_size = sizeof(uint32_t) + total_merged_frames * sizeof(size_t);
        size_t current_data_offset = merged_header_size;
        std::vector<size_t> merged_frame_offsets;
        merged_frame_offsets.reserve(total_merged_frames);

        // Calculate total size and frame offsets in merged data
        for (size_t buffer_idx = 0; buffer_idx < packet_data_buffers.size(); ++buffer_idx) {
            const uint8_t* buffer = packet_data_buffers[buffer_idx];
            size_t buffer_size = buffer_sizes[buffer_idx];
            const auto& frame_offsets = buffer_frame_offsets[buffer_idx];
            uint32_t frame_count = buffer_frame_counts[buffer_idx];

            for (uint32_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
                // Record new offset for this frame in merged data
                merged_frame_offsets.push_back(current_data_offset);

                // Calculate frame data size
                size_t frame_start = frame_offsets[frame_idx];
                size_t frame_end;
                if (frame_idx + 1 < frame_count) {
                    frame_end = frame_offsets[frame_idx + 1];
                } else {
                    frame_end = buffer_size;
                }
                size_t frame_size = frame_end - frame_start;

                current_data_offset += frame_size;
            }
        }

        merged_size = current_data_offset;

        // Step 3: Allocate and build merged data
        merged_data = std::make_unique<uint8_t[]>(merged_size);
        uint8_t* write_ptr = merged_data.get();

        // Write merged header
        *reinterpret_cast<uint32_t*>(write_ptr) = total_merged_frames;
        write_ptr += sizeof(uint32_t);

        // Write merged frame offsets
        for (size_t offset : merged_frame_offsets) {
            *reinterpret_cast<size_t*>(write_ptr) = offset;
            write_ptr += sizeof(size_t);
        }

        // Step 4: Copy frame data from all buffers - direct buffer assignment per thread
        executeInParallel([&](size_t thread_id) {
            // Each thread processes buffers assigned to it
            for (size_t buffer_idx = thread_id; buffer_idx < packet_data_buffers.size();
                 buffer_idx += num_threads) {
                const uint8_t* buffer = packet_data_buffers[buffer_idx];
                size_t buffer_size = buffer_sizes[buffer_idx];
                const auto& frame_offsets = buffer_frame_offsets[buffer_idx];
                uint32_t frame_count = buffer_frame_counts[buffer_idx];

                // Calculate the starting frame index for this buffer in merged data
                size_t base_frame_idx = 0;
                for (size_t prev_buffer_idx = 0; prev_buffer_idx < buffer_idx; ++prev_buffer_idx) {
                    base_frame_idx += buffer_frame_counts[prev_buffer_idx];
                }

                // Copy all frames from this buffer
                for (uint32_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
                    size_t global_frame_idx = base_frame_idx + frame_idx;
                    size_t expected_offset = merged_frame_offsets[global_frame_idx];

                    // Calculate source frame data boundaries
                    size_t frame_start = frame_offsets[frame_idx];
                    size_t frame_end;
                    if (frame_idx + 1 < frame_count) {
                        frame_end = frame_offsets[frame_idx + 1];
                    } else {
                        frame_end = buffer_size;
                    }
                    size_t frame_size = frame_end - frame_start;

                    // Copy frame data to the correct position in merged data
                    uint8_t* dest_ptr = merged_data.get() + expected_offset;
                    std::memcpy(dest_ptr, buffer + frame_start, frame_size);
                }
            }
        });
    } catch (const std::exception& e) {
        merged_data.reset();
        merged_size = 0;
        std::string error_msg = "[ERROR] MergePacketDataToOne failed: ";
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

    // Ensure merge thread pool is initialized
    ensureMergeRunnersInitialized();

    // Calculate number of threads to use
    size_t num_threads = std::min(file_paths.size(), merge_runners.size());

    // Helper lambda for parallel execution with exception handling
    auto executeInParallel = [&](const std::function<void(size_t)>& task_func) {
        std::vector<std::exception_ptr> exceptions(num_threads);

        // Start parallel tasks
        for (size_t i = 0; i < num_threads; ++i) {
            merge_runners[i].start([&, i]() {
                try {
                    task_func(i);
                } catch (...) {
                    exceptions[i] = std::current_exception();
                }
            });
        }

        // Wait for all tasks to complete
        for (size_t i = 0; i < num_threads; ++i) {
            merge_runners[i].join();
        }

        // Check for exceptions
        for (auto& ex : exceptions) {
            if (ex) {
                std::rethrow_exception(ex);
            }
        }
    };

    // Read all binary files in parallel
    file_data_buffers.resize(file_paths.size());

    executeInParallel([&](size_t thread_id) {
        // Process files assigned to this thread
        for (size_t file_idx = thread_id; file_idx < file_paths.size(); file_idx += num_threads) {
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

            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> file_buffer(file_size);
            file.read(reinterpret_cast<char*>(file_buffer.data()), file_size);
            if (file.fail()) {
                throw std::runtime_error("[ERROR] Failed to read file: " + file_path);
            }
            file.close();

            // Validate file header
            if (file_size < sizeof(uint32_t)) {
                throw std::invalid_argument("[ERROR] File too small: " + file_path);
            }

            const uint8_t* data_ptr = file_buffer.data();
            uint32_t frame_count = *reinterpret_cast<const uint32_t*>(data_ptr);

            if (frame_count == 0) {
                throw std::invalid_argument("[ERROR] File contains no frames: " + file_path);
            }

            // Validate header size
            size_t expected_header_size = sizeof(uint32_t) + frame_count * sizeof(size_t);
            if (file_size < expected_header_size) {
                throw std::invalid_argument("[ERROR] File header invalid: " + file_path);
            }

            // Store file data
            file_data_buffers[file_idx] = std::move(file_buffer);
        }
    });

    nvtxRangePop();
}

void PyNvGopDecoder::MergeBinaryFilesToPacketData(const std::vector<std::string>& file_paths,
                                                  std::unique_ptr<uint8_t[]>& merged_data,
                                                  size_t& merged_size) {
    nvtxRangePushA("MergeBinaryFilesToPacketData");

    try {
        // Step 1: Load all GOP files using the extracted helper function
        std::vector<std::vector<uint8_t>> file_data_buffers;
        LoadGOPFromFiles(file_paths, file_data_buffers);

        // Step 2: Create pointer arrays for MergePacketDataToOne
        std::vector<uint8_t*> buffer_pointers(file_data_buffers.size());
        std::vector<size_t> buffer_sizes(file_data_buffers.size());

        for (size_t i = 0; i < file_data_buffers.size(); ++i) {
            buffer_pointers[i] = file_data_buffers[i].data();
            buffer_sizes[i] = file_data_buffers[i].size();
        }

        // Step 3: Call MergePacketDataToOne to merge the data buffers
        MergePacketDataToOne(buffer_pointers, buffer_sizes, merged_data, merged_size);

    } catch (const std::exception& e) {
        merged_data.reset();
        merged_size = 0;
        std::string error_msg = "[ERROR] MergeBinaryFilesToPacketData failed: ";
        error_msg += e.what();
        throw std::runtime_error(error_msg);
    }

    nvtxRangePop();
}
