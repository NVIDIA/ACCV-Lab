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
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

#include "ColorConvertKernels.cuh"

#define MAX_SIZE 2000

namespace fs = std::filesystem;
namespace frame_output = accvlab::on_demand_video_decoder::internal::frame_output;

void PyNvGopDecoder::decode_from_video(const std::vector<std::string>& filepaths,
                                       const std::vector<int> frame_ids, bool convert_to_rgb, bool as_bgr,
                                       std::vector<DecodedFrameExt>* out_if_no_color_conversion,
                                       std::vector<RGBFrame>* out_if_color_converted,
                                       const FastStreamInfo* fastStreamInfos) {
    int st = 0;

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

    nvtxRangePushA("Decode");
    const size_t total_frames = frame_ids.size();

    // Create local packet queues and arrays for this decode operation
    std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>> vpacket_queue;
    std::vector<std::vector<std::unique_ptr<uint8_t[]>>> vpacket_array;
    vpacket_queue.reserve(total_frames);
    vpacket_array.reserve(total_frames);
    for (size_t i = 0; i < total_frames; ++i) {
        vpacket_queue.emplace_back(std::make_unique<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>());
        vpacket_queue[i]->setSize(MAX_SIZE);
        vpacket_array.emplace_back();
    }

    // lazy loading
    ensureCudaContextInitialized();

    // reset last decoded frame infos
    reset_last_decoded_frame_infos(this->last_decoded_frame_infos);

    // Ensure all threads are in clean state before starting new tasks
    // Use force_join_all to handle any inconsistent thread states
    this->force_join_all();

    // Initialize demuxers and calculate memory requirements
    std::vector<std::unique_ptr<PyNvGopDemuxer>> demuxers;
    st = InitializeDemuxers(filepaths, demuxers, fastStreamInfos);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitializeDemuxers failed.");
    }

    std::vector<int> codec_ids(total_frames), widths(total_frames), heights(total_frames),
        frame_sizes(total_frames);
    for (int i = 0; i < total_frames; ++i) {
        codec_ids[i] = demuxers[i]->GetNvCodecId();
        widths[i] = demuxers[i]->GetWidth();
        heights[i] = demuxers[i]->GetHeight();
        frame_sizes[i] = static_cast<int>(frame_output::frame_bytes(
            frame_output::output_format_from_av_pixel_format(demuxers[i]->GetPixelFormat()),
            static_cast<size_t>(heights[i]), static_cast<size_t>(widths[i])));
    }

    st = InitGpuMemPool(heights, widths, frame_sizes, convert_to_rgb);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitGpuMemPool failed.");
    }

    st = InitializeDecoders(codec_ids);
    if (st != 0) {
        throw std::runtime_error("[ERROR] InitializeDecoders failed.");
    }

    std::vector<std::vector<uint8_t*>> per_file_frame_buffers;
    st = GetFileFrameBuffers(&widths, &heights, &frame_sizes, convert_to_rgb, per_file_frame_buffers);
    if (st != 0) {
        throw std::runtime_error("[ERROR] GetFileFrameBuffers failed.");
    }

    std::vector<std::vector<DecodedFrameExt>> decodedFrames;
    std::vector<std::vector<RGBFrame>> rgb_frames;

    if (convert_to_rgb) {
        rgb_frames.resize(total_frames);
    } else {
        decodedFrames.resize(total_frames);
    }

    std::vector<std::string> file_list_missing_range;
    std::mutex mutex_missing_range;
    std::vector<std::vector<int>> all_gop_lens;
    std::vector<std::vector<int>> all_first_frame_ids;
    all_gop_lens.resize(total_frames);
    all_first_frame_ids.resize(total_frames);
    std::vector<std::vector<int>> all_sorted_frame_ids(total_frames);
    std::vector<char> process_frame(total_frames, true);

    nvtxRangePushA("Frame processing");
    for (int i = 0; i < total_frames; ++i) {
        try {
            std::vector<int> sorted_frame_ids = {frame_ids[i]};
            std::vector<int>& first_frame_ids = all_first_frame_ids[i];
            std::vector<int>& gop_length = all_gop_lens[i];

            // Only process GOP info for VFR videos
            if (demuxers[i]->IsVFRV2()) {
                st = ExtractAndProcessGopInfo(demuxers[i], sorted_frame_ids, first_frame_ids, gop_length);
                if (st != 0) {
                    throw std::runtime_error("[ERROR] extract and process gop info failed for file: " +
                                             filepaths[i]);
                }
            }
            if (convert_to_rgb) {
                WarnIfColorRangeUnspecified(demuxers[i]->GetColorRange(), filepaths[i]);
                rgb_frames[i].reserve(1);
            } else {
                decodedFrames[i].reserve(1);
            }

            all_sorted_frame_ids[i] = std::move(sorted_frame_ids);
        } catch (const std::exception& e) {
            process_frame[i] = false;
            this->force_join_all();
            std::cerr << "[ERROR] " << e.what() << std::endl;
        }
    }
    try {
        demux_pool.submit_indexed(total_frames, [&](size_t index) {
            if (!process_frame[index]) {
                return;
            }
            DemuxGopProc(demuxers[index].get(), vpacket_queue[index].get(), all_sorted_frame_ids[index],
                         all_first_frame_ids[index], all_gop_lens[index], vpacket_array[index], false);
        });
        decode_pool.submit_indexed(total_frames, [&](size_t index) {
            if (!process_frame[index]) {
                return;
            }
            if (convert_to_rgb) {
                DecProc<RGBFrame>(demuxers[index]->GetColorRange(), this->vdec[index].get(),
                                  rgb_frames[index], per_file_frame_buffers[index],
                                  vpacket_queue[index].get(), all_sorted_frame_ids[index], as_bgr,
                                  filepaths[index], this->last_decoded_frame_infos[index]);
            } else {
                DecProc<DecodedFrameExt>(demuxers[index]->GetColorRange(), this->vdec[index].get(),
                                         decodedFrames[index], per_file_frame_buffers[index],
                                         vpacket_queue[index].get(), all_sorted_frame_ids[index], false,
                                         filepaths[index], this->last_decoded_frame_infos[index]);
            }
        });
    } catch (...) {
        this->force_join_all();
        throw;
    }
    nvtxRangePop();  //Frame processing

    nvtxRangePushA("Demux & decode thread join");
    if (convert_to_rgb) {
        out_if_color_converted->resize(total_frames);
    } else {
        out_if_no_color_conversion->resize(total_frames);
    }
    try {
        wait_all(demux_pool, decode_pool);
        for (int i = 0; i < total_frames; ++i) {
            if (convert_to_rgb) {
                if (rgb_frames[i].empty()) {
                    this->force_join_all();
                    throw std::runtime_error("[ERROR] rgb_frames[i] is empty for frame " + std::to_string(i));
                }
                (*out_if_color_converted)[i] = std::move(rgb_frames[i][0]);
            } else {
                if (decodedFrames[i].empty()) {
                    this->force_join_all();
                    throw std::runtime_error("[ERROR] decodedFrames[i] is empty for frame " +
                                             std::to_string(i));
                }
                (*out_if_no_color_conversion)[i] = std::move(decodedFrames[i][0]);
            }
        }

    } catch (const std::exception& e) {
        this->force_join_all();
        throw std::runtime_error(e.what());
    }
    nvtxRangePop();
    nvtxRangePop();
}
