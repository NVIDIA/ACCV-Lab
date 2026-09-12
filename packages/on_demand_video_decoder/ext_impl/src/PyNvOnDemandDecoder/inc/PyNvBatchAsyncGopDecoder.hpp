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

#include "GPUMemoryPool.hpp"
#include "NvCodecUtils.h"
#include "PyCAIMemoryView.hpp"
#include "PyDecodedFrameExt.hpp"
#include "PyNvGopDecoder.hpp"  // also transitively includes ThreadPool.hpp
#include "PyRGBFrame.hpp"

#include <cuda.h>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvBatchAsyncGopDecoder {
#else
class PyNvBatchAsyncGopDecoder {
#endif
   public:
    /**
     * Construct a 2D async GOP batch decoder.
     *
     * Args:
     *   maxfiles: maximum number of videos per decode call (V upper bound)
     *   max_frames_per_decode_call: maximum number of frames per video per call (F upper bound)
     *   iGpu: target GPU device id
     *   suppressNoColorRangeWarning: suppress warning when no color range can be extracted
     */
    PyNvBatchAsyncGopDecoder(int maxfiles, int max_frames_per_decode_call, int iGpu = 0,
                             bool suppressNoColorRangeWarning = false);

    ~PyNvBatchAsyncGopDecoder();

    /**
     * Submit an async 2D RGB GOP decode task.  Returns immediately.
     *
     * numpy_datas[v][g] is the serialized GOP bundle (from GetGOPList) for video v, GOP index g.
     * All bundles for video v together must cover every frame_id in frame_ids_2d[v].
     * frame_ids_2d[v] need not be sorted; output order matches input order.
     *
     * At most one task (RGB or YUV) may be in flight at a time.  Submitting
     * while a previous task is pending joins it first (with a stderr warning).
     */
    void DecodeFromGOPListRGB(std::vector<std::vector<std::vector<uint8_t>>> numpy_datas,
                              const std::vector<std::string>& filepaths,
                              const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr,
                              bool enable_gop_dependency_graph_optimization = false);

    /**
     * Block until the pending RGB task completes and return decoded frames.
     *
     * Returns List[List[RGBFrame]] indexed [v][f], matching frame_ids_2d shape.
     * Returned frames reference internal aggregator pool memory — clone before
     * the next DecodeFromGOPListRGB() call.
     *
     * Args must match those passed to the preceding DecodeFromGOPListRGB().
     */
    std::vector<std::vector<RGBFrame>> DecodeFromGOPListRGBGetBuffer(
        const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
        bool as_bgr);

    /**
     * Submit an async 2D YUV GOP decode task.  Returns immediately.
     *
     * Same numpy_datas/filepath/frame_ids semantics as DecodeFromGOPListRGB.
     */
    void DecodeFromGOPList(std::vector<std::vector<std::vector<uint8_t>>> numpy_datas,
                           const std::vector<std::string>& filepaths,
                           const std::vector<std::vector<int>>& frame_ids_2d,
                           bool enable_gop_dependency_graph_optimization = false);

    /**
     * Block until the pending YUV task completes and return decoded frames.
     *
     * Returns List[List[DecodedFrameExt]] indexed [v][f].
     * Returned frames reference internal aggregator pool memory — clone before
     * the next DecodeFromGOPList() call.
     */
    std::vector<std::vector<DecodedFrameExt>> DecodeFromGOPListGetBuffer(
        const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d);

    /**
     * Release aggregator GPU memory pools (RGB and YUV).
     * Decoder state is preserved.
     */
    void release_device_memory();

    /**
     * Release the internal GOP decoder instance.
     * It is re-created lazily on the next decode call.
     */
    void release_decoder();

   private:
    // Unified result type for both RGB and YUV async tasks.
    struct DecodeResultGOP {
        std::vector<std::string> file_path_list;
        std::vector<std::vector<int>> frame_id_list_2d;
        bool as_bgr = false;
        bool is_rgb = true;
        std::vector<std::vector<RGBFrame>> decoded_rgb_frames;
        std::vector<std::vector<DecodedFrameExt>> decoded_yuv_frames;
        std::exception_ptr exception;
        bool is_ready = false;

        DecodeResultGOP() = default;
    };

    std::string generate_request_key(const std::vector<std::string>& filepaths,
                                     const std::vector<std::vector<int>>& frame_ids_2d);

    bool validate_request(const DecodeResultGOP& result, const std::vector<std::string>& filepaths,
                          const std::vector<std::vector<int>>& frame_ids_2d);

    void validate_decode_input(const std::vector<std::string>& filepaths,
                               const std::vector<std::vector<int>>& frame_ids_2d,
                               const std::vector<std::vector<std::vector<uint8_t>>>& numpy_datas);

    // Common submission path shared by RGB and YUV decode calls.
    void submit_work(std::vector<std::vector<std::vector<uint8_t>>> numpy_datas,
                     std::vector<std::string> filepaths, std::vector<std::vector<int>> frame_ids_2d,
                     bool as_bgr, bool is_rgb, bool enable_gop_dependency_graph_optimization);

   private:
    bool suppress_no_color_range_warning_ = false;
    bool destroy_context_ = false;
    CUcontext cu_context_ = nullptr;
    CUstream cu_stream_ = nullptr;
    int gpu_id_ = 0;
    int maxfiles_ = 0;
    int max_frames_per_decode_call_ = 0;

    // Single GOP decoder sized to maxfiles (V) and sharing this object's stream.
    // The worker issues F transposed decode_from_gop_list calls (frame-slot f of all
    // V videos at once), decoding the V videos in parallel; continuous-decode across
    // the F calls makes each video's GOP decode once instead of once per frame.
    std::unique_ptr<PyNvGopDecoder> gop_dec_;

    // Per-video aggregator pools (RGB and YUV paths share separate pools).
    std::vector<GPUMemoryPool> rgb_agg_pools_;
    std::vector<GPUMemoryPool> yuv_agg_pools_;

    // Async machinery.  Single-slot result buffer protected by async_mutex_.
    std::deque<DecodeResultGOP> result_queue_;
    std::condition_variable result_cv_;
    ThreadRunner decode_worker_;
    std::mutex async_mutex_;
    bool has_pending_task_ = false;
};
