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
#include "PyNvVideoReader.hpp"
#include "ThreadPool.hpp"
#include "NvCodecUtils.h"
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
#include <sstream>
#include <exception>

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvSampleReader {
#else
class PyNvSampleReader {
#endif
   public:
    // Currently, we assume that for each file, the frame_ids to be extracted have
    // no duplication.
    PyNvSampleReader(int num_of_set, int num_of_file, int iGpu, bool bSuppressNoColorRangeWarning = false);

    ~PyNvSampleReader();

    void clearAllReaders();

    /**
     * Release GPU device memory from all video readers to free up GPU memory.
     * 
     * This method releases the GPU memory pools from all PyNvVideoReader
     * instances managed by this SampleReader. Decoder state (current position,
     * packet queue, etc.) is preserved for efficient forward decoding.
     * 
     * Behavior after calling this method:
     * - Requesting frame_id > cur_frame_: efficient forward decode (no re-seek)
     * - Requesting frame_id <= cur_frame_: triggers GOP re-seek and re-decode
     * 
     * The memory pools will be re-allocated automatically on the next decode.
     */
    void ReleaseMemPools();

    /**
     * Release all video decoder instances to free up GPU memory
     * 
     * This method clears all video readers, which releases:
     * - NvDecoder instances and their GPU frame buffers
     * - Each video reader's GPUMemoryPool instances
     * 
     * This is useful for freeing GPU memory occupied by decoder instances.
     * 
     * Note: After calling this method, video readers will need to be
     * re-created on the next decode operation.
     */
    void ReleaseDecoder();

    std::vector<RGBFrame> run_rgb_out(const std::vector<std::string>& filepaths,
                                      const std::vector<int> frame_ids, bool as_bgr);
    std::vector<DecodedFrameExt> run(const std::vector<std::string>& filepaths,
                                     const std::vector<int> frame_ids);

    /**
     * Asynchronously decode video frames to RGB/BGR format.
     * 
     * This method submits a decode task to a background thread and returns immediately.
     * The decoded frames will be stored in an internal buffer and can be retrieved
     * using DecodeN12ToRGBAsyncGetBuffer.
     * 
     * If a previous async decode task is still running, this method will wait for
     * it to complete before starting the new task, and print a warning.
     * 
     * Args:
     *     filepaths: List of video file paths to decode from
     *     frame_ids: List of frame IDs to decode from the video files
     *     as_bgr: Whether to output in BGR format (True) or RGB format (False)
     */
    void DecodeN12ToRGBAsync(const std::vector<std::string>& filepaths, const std::vector<int>& frame_ids,
                             bool as_bgr);

    /**
     * Get decoded frames from the async decode buffer.
     * 
     * This method retrieves decoded frames from the internal buffer that were
     * previously submitted via DecodeN12ToRGBAsync. It validates that the
     * requested filepaths and frame_ids match the buffered result.
     * 
     * Args:
     *     filepaths: List of video file paths (must match the async request)
     *     frame_ids: List of frame IDs (must match the async request)
     *     as_bgr: BGR format flag (must match the async request)
     * 
     * Returns:
     *     List of RGBFrame objects containing the decoded frames
     * 
     * Raises:
     *     RuntimeError: If no matching result is found in buffer, or validation fails
     */
    std::vector<RGBFrame> DecodeN12ToRGBAsyncGetBuffer(const std::vector<std::string>& filepaths,
                                                       const std::vector<int>& frame_ids, bool as_bgr);

    /**
     * Wait for any pending async decode task to complete.
     * 
     * This method should be called at the beginning of any operation that
     * accesses shared resources (VideoReaderMap, GPU memory pool, etc.) to
     * ensure thread safety when mixing sync and async APIs.
     * 
     * If there's no pending task, this method returns immediately with
     * negligible overhead (~100ns).
     */
    void waitForPendingAsyncTask();

    /**
     * Clear the decode result buffer.
     * 
     * This method clears any existing async result from the buffer.
     * Called by sync API to invalidate stale async results.
     */
    void clearDecodeResultBuffer();

   private:
    // Structure to store async decode request and result
    struct DecodeResult {
        std::vector<std::string> file_path_list;
        std::vector<int> frame_id_list;
        bool as_bgr;
        std::vector<RGBFrame> decoded_frames;
        std::exception_ptr exception;
        bool is_ready;

        DecodeResult() : is_ready(false) {}
    };

    // Helper function to generate a unique key for a decode request
    std::string generate_request_key(const std::vector<std::string>& filepaths,
                                     const std::vector<int>& frame_ids, bool as_bgr);

    // Helper function to validate request parameters match
    bool validate_request(const DecodeResult& result, const std::vector<std::string>& filepaths,
                          const std::vector<int>& frame_ids, bool as_bgr);

   private:
    bool suppress_no_color_range_given_warning = false;
    bool destroy_context = false;
    CUcontext cu_context = NULL;
    CUstream cu_stream = NULL;
    int gpu_id = 0;
    int num_of_file = 0;
    int num_of_set = 0;

    std::vector<FixedSizeVideoReaderMap> VideoReaderMap;

    // Async decode related members
    ConcurrentQueue<DecodeResult> decode_result_queue;  // Buffer size = 1
    ThreadRunner decode_worker;                         // Worker thread for async decoding
    std::mutex async_mutex;                             // Mutex for async operations
    bool has_pending_task;                              // Flag to track if there's a pending task
};
