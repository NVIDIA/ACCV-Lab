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

#include "FixedSizeVideoReaderMap.hpp"
#include "GPUMemoryPool.hpp"
#include "NvCodecUtils.h"
#include "PyNvVideoReader.hpp"
#include "PyRGBFrame.hpp"
#include "ThreadPool.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvBatchAsyncStreamReader {
#else
class PyNvBatchAsyncStreamReader {
#endif
   public:
    /**
     * Construct an async 2D batch stream decoder.
     *
     * Args:
     *   num_of_set: number of decoder slots per file (same meaning as in PyNvSampleReader)
     *   num_of_file: maximum number of videos per decode call (V upper bound)
     *   max_frames_per_decode_call: maximum number of frames per video per decode call (F upper bound)
     *   iGpu: target GPU device id
     *   bSuppressNoColorRangeWarning: suppress warning if no color range can be extracted
     */
    PyNvBatchAsyncStreamReader(int num_of_set, int num_of_file, int max_frames_per_decode_call, int iGpu,
                               bool bSuppressNoColorRangeWarning = false);

    ~PyNvBatchAsyncStreamReader();

    /**
     * Clear all underlying video readers (also waits for any pending async task).
     */
    void clearAllReaders();

    /**
     * Release per-reader memory pools AND the 2D aggregator pool.
     * Decoder state is preserved for efficient forward decoding.
     */
    void ReleaseMemPools();

    /**
     * Release all decoder instances. After this, readers will be lazily re-created
     * on the next decode call.
     */
    void ReleaseDecoder();

    /**
     * Submit an async 2D decode task. Returns immediately.
     *
     * Contract: at most one in-flight task. Submitting while a previous task is
     * pending causes a warning, joins the previous task, and discards its result.
     *
     * Args:
     *   filepaths: list of video file paths (len == V_call, V_call <= num_of_file)
     *   frame_ids_2d: 2D frame id list (outer V_call, inner F_call). All inner
     *                 lengths must be equal. F_call <= max_frames_per_decode_call.
     *                 frame_ids_2d[v][f] = the f-th frame requested for video v.
     *   as_bgr: output BGR if true, RGB if false
     */
    void Decode(const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
                bool as_bgr);

    /**
     * Block until the pending async task completes and return its decoded frames.
     *
     * Contract:
     *   - GPU side: all decode + D2D copies complete (worker did cuStreamSynchronize).
     *               Downstream torch / CUDA ops can read without further sync.
     *   - Buffer:   the returned RGBFrame objects reference an internal aggregator
     *               pool. They are invalidated on the next Decode() call. Users
     *               must consume / clone the data before calling Decode() again.
     *
     * Args: must match the request previously submitted via Decode().
     * Returns: List[List[RGBFrame]], outer V_call, inner F_call, indexed [v][f]
     *          to match the input frame_ids_2d shape.
     */
    std::vector<std::vector<RGBFrame>> GetBuffer(const std::vector<std::string>& filepaths,
                                                 const std::vector<std::vector<int>>& frame_ids_2d,
                                                 bool as_bgr);

    /**
     * Wait for any pending async decode task to complete. No-op if no pending task.
     */
    void waitForPendingAsyncTask();

    /**
     * Clear any cached async result (does NOT clear in-flight task).
     */
    void clearDecodeResultBuffer();

   private:
    struct DecodeResult2D {
        std::vector<std::string> file_path_list;
        std::vector<std::vector<int>> frame_id_list_2d;
        bool as_bgr;
        std::vector<std::vector<RGBFrame>> decoded_frames;
        std::exception_ptr exception;
        bool is_ready;

        DecodeResult2D() : as_bgr(false), is_ready(false) {}
    };

    // Build a key string for diagnostic reporting on request mismatch.
    std::string generate_request_key(const std::vector<std::string>& filepaths,
                                     const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr);

    // Field-wise compare a buffered result against an incoming request.
    bool validate_request(const DecodeResult2D& result, const std::vector<std::string>& filepaths,
                          const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr);

    // Throw if Decode() input violates declared invariants.
    void validate_decode_input(const std::vector<std::string>& filepaths,
                               const std::vector<std::vector<int>>& frame_ids_2d);

    // Sync 1D decode over the owned VideoReaderMap. The 2D worker calls this
    // F times. Mirrors PyNvSampleReader::run_rgb_out but operates on this
    // class's reader pool so the two classes don't share decoder state.
    std::vector<RGBFrame> run_rgb_out_1d(const std::vector<std::string>& filepaths,
                                         const std::vector<int>& frame_ids, bool as_bgr);

   private:
    bool suppress_no_color_range_given_warning = false;
    bool destroy_context = false;
    CUcontext cu_context = nullptr;
    CUstream cu_stream = nullptr;
    int gpu_id = 0;
    int num_of_file = 0;
    int num_of_set = 0;
    int max_frames_per_decode_call = 0;

    std::vector<FixedSizeVideoReaderMap> VideoReaderMap;

    // 2D-specific aggregator pools, one per video slot. Each pool holds the
    // F frames decoded for that slot in a single Decode() call. Per-video
    // sizing means videos in one call can have different resolutions — each
    // pool sizes itself to F * (H_v * W_v * 3) at f==0 and reallocates only
    // when the slot's resolution grows across calls.
    // Only accessed by the single decode worker; no thread-safety required.
    std::vector<GPUMemoryPool> agg_pools;

    // Async machinery (mirrors PyNvSampleReader)
    ConcurrentQueue<DecodeResult2D> decode_result_queue;  // capacity = 1
    ThreadRunner decode_worker;
    std::mutex async_mutex;
    bool has_pending_task = false;
};
