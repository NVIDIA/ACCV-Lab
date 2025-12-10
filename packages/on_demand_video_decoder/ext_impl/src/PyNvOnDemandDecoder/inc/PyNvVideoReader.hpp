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

#include "FFmpegDemuxer.h"
#include "GPUMemoryPool.hpp"
#include "GopDecoderUtils.hpp"
#include "NvCodecUtils.h"
#include "NvDecoder/NvDecoder.h"
#include "PyCAIMemoryView.hpp"
#include "PyDecodedFrameExt.hpp"
#include "PyNvGopDemuxer.hpp"
#include "PyRGBFrame.hpp"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvVideoReader {
#else
class PyNvVideoReader {
#endif
   public:
    // Currently, we assume that for each file, the frame_ids to be extracted have
    // no duplication.
    PyNvVideoReader(std::string filename, int iGpu, CUcontext cu_context = NULL, CUstream cu_stream = NULL,
                    bool bSuppressNoColorRangeWarning = false);
    PyNvVideoReader(std::string filename, int iGpu, bool bSuppressNoColorRangeWarning = false)
        : PyNvVideoReader(filename, iGpu, NULL, NULL, bSuppressNoColorRangeWarning) {}
    ~PyNvVideoReader();
    void ReplaceWithFile(const std::string filename);

    /**
     * Release GPU device memory pool to free up GPU memory.
     * 
     * This method only releases the GPU memory pool, preserving decoder state
     * (cur_frame_, packet_queue, etc.) for efficient forward decoding.
     * 
     * Behavior after calling this method:
     * - Requesting frame_id > cur_frame_: efficient forward decode (no re-seek)
     * - Requesting frame_id <= cur_frame_: triggers GOP re-seek and re-decode
     * 
     * The memory pool will be re-allocated automatically on the next decode.
     */
    void ReleaseMemPools();

    static Pixel_Format GetNativeFormat(const cudaVideoSurfaceFormat inputFormat);

    std::vector<DecodedFrameExt> run(const std::vector<int> frame_ids);

    DecodedFrameExt run_single(const int frame_id);

    std::vector<RGBFrame> run_rgb_out(const std::vector<int> frame_ids, bool as_bgr);

    RGBFrame run_single_rgb_out(const int frame_ids, bool as_bgr);

    void parse_keyframe_idx(std::vector<int>& key_frames, std::map<int, int64_t>& frame2pts,
                            std::map<int64_t, int>& pts2frame);

    static void DemuxGopProc(PyNvGopDemuxer* demuxer,
                             ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                             const int key_frame_id, const int gop_len, std::vector<uint8_t*>& packet_array,
                             bool seeking);

   protected:
    void releasePacketArray();
    void startNewGop();
    void startNextGop();
    DecodedFrameExt returnYUVFrame(void* pFrame_buffer, void* pFrame);
    RGBFrame returnRGBFrame(void* pFrame_buffer, void* pFrame, bool use_bgr_format,
                            bool& file_added_for_warning);
    void run_single_frame_internal(const int frame_ids, bool convert_to_rgb, bool as_bgr,
                                   DecodedFrameExt* out_if_no_color_conversion,
                                   RGBFrame* out_if_color_converted);
    void fetchNextGop();
    void fetchNewGop(int cur_keyframe);
    void decodeNextPacket();
    bool processDecodedFrames(const int frame_id, uint8_t* pFrame, uint8_t* pReturnFrame, bool convert_to_rgb,
                              bool use_bgr_format, bool& file_added_for_warning,
                              RGBFrame* out_if_color_converted, DecodedFrameExt* out_if_no_color_conversion);
    static void demuxGopProcZeroLen(PyNvGopDemuxer* demuxer,
                                    ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                                    const int key_frame_ids, std::vector<uint8_t*>& packet_array,
                                    bool seeking);

   private:
    std::string filename = {};

    bool suppress_no_color_range_given_warning = false;
    bool destroy_context = false;
    CUcontext cu_context = NULL;
    bool owner_stream = false;
    CUstream cu_stream = NULL;
    int gpu_id = 0;
    int cur_frame_ = 0;
    int next_keyframe_ = 0;
    int return_frames_ = 0;

    std::unique_ptr<PyNvGopDemuxer> demuxer;
    std::unique_ptr<NvDecoder> decoder;
    std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>> packet_queue;
    std::vector<uint8_t*> packet_array;

    GPUMemoryPool gpu_mem_pool;
};
