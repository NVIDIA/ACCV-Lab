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
#include "ThreadPool.hpp"
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

#define MAX_SIZE 2000

namespace py = pybind11;

struct SerializedPacketBundle {
    std::unique_ptr<uint8_t[]> data;   // Binary data containing header, offset table, and frame data
    size_t size;                       // Total size of binary data
    std::vector<int> gop_lens;         // GOP lengths for easy access (also embedded in binary data)
    std::vector<int> first_frame_ids;  // First frame IDs for easy access (also embedded in binary data)
};

struct LastDecodedFrameInfo {
    std::string filename;
    int frame_id;
    int packet_id;
};

inline void reset_last_decoded_frame_info(LastDecodedFrameInfo& last_decoded_frame_info) {
    last_decoded_frame_info.filename = "";
    last_decoded_frame_info.frame_id = -1;
    last_decoded_frame_info.packet_id = 0;
}

inline void reset_last_decoded_frame_infos(std::vector<LastDecodedFrameInfo>& last_decoded_frame_infos) {
    for (auto& info : last_decoded_frame_infos) {
        reset_last_decoded_frame_info(info);
    }
}

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvGopDecoder {
#else
class PyNvGopDecoder {
#endif
   public:
    PyNvGopDecoder(int iMaxFileNum = 100, int iGpu = 0, bool bSuppressNoColorRangeWarning = false);

    ~PyNvGopDecoder();

    /**
     * Force all thread runners to synchronously terminate their tasks
     * 
     * This method provides a way to forcefully stop all running threads in case of
     * exceptions or when immediate termination is needed. It clears all pending tasks
     * and waits for current tasks to complete, then resets all thread states.
     * 
     * Use this method when:
     * - Threads are in an inconsistent state due to exceptions
     * - You need to ensure clean state before starting new operations
     * - Emergency cleanup is required
     * 
     * @note This method will block until all threads are in a clean state
     */
    void force_join_all();

    void decode_from_video(const std::vector<std::string>& filepaths, const std::vector<int> frame_ids,
                           bool convert_to_rgb, bool as_bgr,
                           std::vector<DecodedFrameExt>* out_if_no_color_conversion,
                           std::vector<RGBFrame>* out_if_color_converted,
                           const FastStreamInfo* fastStreamInfos = nullptr);

    SerializedPacketBundle get_gop(const std::vector<std::string>& filepaths,
                                   const std::vector<int> frame_ids,
                                   const FastStreamInfo* fastStreamInfos = nullptr);

    /**
     * Extract GOP data for multiple videos and return them as separate bundles
     * 
     * Similar to get_gop(), but instead of merging all video data into one bundle,
     * this method returns a separate SerializedPacketBundle for each video file.
     * This is useful when you want to cache or process each video's data independently.
     * 
     * @param filepaths Vector of video file paths
     * @param frame_ids Vector of frame IDs corresponding to each filepath
     * @param fastStreamInfos Optional array of FastStreamInfo for performance optimization
     * @return Vector of SerializedPacketBundle, one for each video file
     */
    std::vector<SerializedPacketBundle> get_gop_list(const std::vector<std::string>& filepaths,
                                                     const std::vector<int> frame_ids,
                                                     const FastStreamInfo* fastStreamInfos = nullptr);

    void decode_from_gop(const uint8_t* data, size_t size, const std::vector<std::string>& filepaths,
                         const std::vector<int> frame_ids, bool convert_to_rgb, bool as_bgr,
                         std::vector<DecodedFrameExt>* out_if_no_color_conversion,
                         std::vector<RGBFrame>* out_if_color_converted);

    void decode_from_packet_list(std::vector<std::vector<int>> packets_bytes,
                                 std::vector<std::vector<int>> decode_idxs, std::vector<int> widths,
                                 std::vector<int> heights,
                                 std::vector<std::vector<const uint8_t*>> packet_binary_data_ptrs,
                                 std::vector<int> frame_ids, bool as_bgr, std::vector<RGBFrame>* dst);

    /**
     * Decode frames from a list of serialized packet bundles (each as a contiguous byte array)
     * The method parses each bundle via parseSerializedPacketData, rebuilds packet queues, and
     * calls main_decode once for all aggregated frames with RGB output.
     * @param datas Vector of pointers to serialized packet data buffers
     * @param sizes Vector of sizes for each serialized packet data buffer
     * @param filepaths Vector of source filepaths corresponding to each target frame (aggregated)
     * @param frame_ids Vector of target frame IDs (aggregated across all bundles)
     * @param as_bgr Whether to output BGR (true) or RGB (false)
     * @param dst Output vector of RGBFrame with size equal to frame_ids.size()
     */
    void decode_from_gop_list(const std::vector<const uint8_t*>& datas, const std::vector<size_t>& sizes,
                              const std::vector<std::string>& filepaths, const std::vector<int> frame_ids,
                              bool as_bgr, std::vector<RGBFrame>* dst);

    /**
     * Merge multiple packet data buffers containing SerializedPacketBundle data into a single data buffer (multithreaded)
     * @param packet_data_buffers Vector of binary data buffer pointers (already read from files)
     * @param buffer_sizes Vector of buffer sizes corresponding to each buffer pointer
     * @param merged_data Output pointer to the merged binary data (caller must delete[])
     * @param merged_size Output size of the merged binary data
     */
    void MergePacketDataToOne(const std::vector<uint8_t*>& packet_data_buffers,
                              const std::vector<size_t>& buffer_sizes,
                              std::unique_ptr<uint8_t[]>& merged_data, size_t& merged_size);

    /**
     * Merge multiple GOP binary files containing SerializedPacketBundle data into a single data buffer (multithreaded)
     * @param file_paths Vector of file paths to binary files
     * @param merged_data Output pointer to the merged binary data (caller must delete[])
     * @param merged_size Output size of the merged binary data
     */
    void MergeBinaryFilesToPacketData(const std::vector<std::string>& file_paths,
                                      std::unique_ptr<uint8_t[]>& merged_data, size_t& merged_size);

    /**
     * Load GOP data from multiple binary files in parallel
     * 
     * This function loads serialized GOP data from multiple files concurrently,
     * with parallel file reading, validation, and error handling. This is a reusable
     * component for loading GOP binary files that can be used by multiple higher-level
     * functions like MergeBinaryFilesToPacketData and LoadGopList (Python binding).
     * 
     * Binary File Format:
     * - Files must contain SerializedPacketBundle data (same format as GetGOP output)
     * - Header: uint32_t total_frames + size_t[total_frames] frame_offsets
     * - Frame data blocks follow the header
     * 
     * Performance:
     * - Files are read in parallel using internal thread pool (merge_runners)
     * - Number of threads = min(file_paths.size(), merge_runners.size())
     * - Each file is validated for correct GOP format
     * 
     * @param file_paths Vector of file paths to GOP binary files
     * @param file_data_buffers Output vector where each element contains the binary data of one file
     * 
     * @throws std::runtime_error if any file cannot be read
     * @throws std::invalid_argument if file format validation fails (invalid header, empty frames, etc.)
     * 
     * Thread Safety: Thread-safe with internal thread pool for parallelization
     * 
     * Example:
     *     std::vector<std::vector<uint8_t>> gop_data_list;
     *     decoder.LoadGOPFromFiles(file_paths, gop_data_list);
     *     // Now gop_data_list[i] contains the binary data from file_paths[i]
     */
    void LoadGOPFromFiles(const std::vector<std::string>& file_paths,
                          std::vector<std::vector<uint8_t>>& file_data_buffers);

    /**
     * Parse serialized packet data with embedded offset table
     * 
     * This function parses the self-contained binary data format and extracts
     * all frame metadata and packet information using the embedded offset table.
     * It's the counterpart to createSerializedPacketBundle for deserialization.
     * 
     * @param data Pointer to the serialized binary data (with embedded offset table)
     * @param size Size of the binary data
     * @param color_ranges Output vector for color ranges of all frames
     * @param codec_ids Output vector for codec IDs of all frames
     * @param widths Output vector for frame widths
     * @param heights Output vector for frame heights
     * @param frame_sizes Output vector for frame sizes
     * @param gop_lens Output vector for GOP lengths
     * @param first_frame_ids Output vector for first frame IDs
     * @param packets_bytes Output vector of packet size arrays for all frames
     * @param decode_idxs Output vector of decode index arrays for all frames
     * @param packet_binary_data_ptrs Output vector of pointers to binary data for all frames
     * @param packet_binary_data_sizes Output vector of binary data sizes for all frames
     * @return Number of frames parsed
     */
    static uint32_t parseSerializedPacketData(
        const uint8_t* data, size_t size, std::vector<int>& color_ranges, std::vector<int>& codec_ids,
        std::vector<int>& widths, std::vector<int>& heights, std::vector<int>& frame_sizes,
        std::vector<int>& gop_lens, std::vector<int>& first_frame_ids,
        std::vector<std::vector<int>>& packets_bytes, std::vector<std::vector<int>>& decode_idxs,
        std::vector<const uint8_t*>& packet_binary_data_ptrs, std::vector<size_t>& packet_binary_data_sizes);

    /**
     * Initialize decoders
     * 
     * This method implements several critical optimizations for high-performance video decoding:
     *
     * ## Decoder Resource Optimization:
     * 1. **Decoder Instance Reuse**: Maintains a persistent vector of NvDecoder instances (vdec)
     *    and only creates new decoders when needed (i >= vdec.size()), enabling:
     *    - Reduced initialization overhead for subsequent decode operations
     *    - Preservation of decoder state and optimizations across sessions
     *    - Lower GPU resource consumption through instance pooling
     * 2. **Context-Aware Initialization**: Each decoder is initialized with shared CUDA context
     *    and stream for optimal GPU resource utilization.
     * 
     * @param codec_ids Vector of NVIDIA video codec identifiers (cudaVideoCodec) for each file.
     *                  The size of this vector determines the number of decoders to initialize.
     * 
     * @return 0 on success, non-zero error code on failure
     */
    int InitializeDecoders(const std::vector<int>& codec_ids);

    /**
     * Release GPU device memory pool to free up GPU memory
     * 
     * This method releases the GPU memory pool by calling HardRelease(),
     * which frees all allocated GPU memory and resets the pool state.
     * This is useful for temporarily freeing excessive GPU memory usage.
     * 
     * Note: After calling this method, the memory pool will need to be
     * re-allocated on the next decode operation.
     */
    void ReleaseMemPools();

    /**
     * Release all decoder instances to free up GPU memory
     * 
     * This method clears all decoder instances, which releases:
     * - NvDecoder instances and their GPU frame buffers
     * 
     * This is useful for freeing GPU memory occupied by decoder instances.
     * 
     * Note: After calling this method, decoder instances will need to be
     * re-created on the next decode operation.
     */
    void ReleaseDecoder();

   protected:
    int main_decode(
        const std::vector<int>& color_ranges, const std::vector<int>& codec_ids, std::vector<int>& widths,
        std::vector<int>& heights, std::vector<int>& frame_sizes, const std::vector<std::string>& filepaths,
        const std::vector<int>& frame_ids, bool convert_to_rgb, bool as_bgr,
        std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
        std::vector<DecodedFrameExt>* out_if_no_color_conversion,
        std::vector<RGBFrame>* out_if_color_converted);

    /**
     * Perform GOP-based video demuxing and packet extraction for high-performance parallel decoding
     * 
     * This static method implements the core demuxing pipeline for GOP (Group of Pictures) based
     * video processing. It orchestrates the extraction, organization, and queuing of video packets
     * optimized for parallel decoding workflows. The function is designed to handle both sequential
     * and random-access frame extraction with minimal overhead.
     * 
     * ## Core Functionality:
     * 1. **GOP Structure Analysis**: Analyzes video GOP boundaries and frame dependencies
     * 2. **Selective Packet Extraction**: Extracts only packets required for target frames
     * 3. **Dependency Resolution**: Ensures all dependency frames (I/P/B frames) are included
     * 4. **Concurrent Queue Population**: Populates thread-safe packet queue for parallel decoding
     * 5. **Memory Management**: Manages packet memory allocation and lifetime
     * 
     * ## Processing Pipeline:
     * ```
     * Input: Demuxer + Target Frame IDs
     *   ↓
     * 1. GOP Boundary Detection
     *   ↓
     * 2. Dependency Frame Analysis (I/P/B frame relationships)
     *   ↓
     * 3. Packet Range Calculation (minimal packet set for decode)
     *   ↓
     * 4. Sequential Packet Extraction
     *   ↓
     * 5. Packet Queue Population (thread-safe)
     *   ↓
     * 6. Metadata Collection (GOP lens, first frame IDs)
     *   ↓
     * Output: Populated packet queue + GOP metadata
     * ```
     * 
     * ## Performance Optimizations:
     * 1. **Minimal Packet Extraction**: Only extracts packets absolutely required for target frames,
     *    significantly reducing I/O and memory usage compared to full-stream demuxing
     * 2. **GOP-Aware Processing**: Leverages GOP structure to minimize packet dependencies,
     *    enabling efficient random access without full stream processing
     * 3. **Concurrent Queue Integration**: Uses lock-free concurrent queue for zero-copy packet
     *    handoff to decoder threads, eliminating synchronization bottlenecks
     * 4. **Adaptive Data Extraction**: When `use_dd` is enabled, extracts complete GOP data
     *    for optimal decode context, trading memory usage for decode quality and performance
     * 5. **Batch Processing**: Processes multiple frame requests in a single demux pass,
     *    amortizing FFmpeg initialization and seek costs
     * 
     * ## GOP Structure Handling:
     * - **I-Frame Detection**: Identifies keyframes that serve as GOP boundaries
     * - **P/B Frame Dependencies**: Calculates dependency chains to ensure complete decode contexts
     * - **Seek Optimization**: Minimizes demuxer seeks by processing frames in GOP order
     * - **Cross-GOP Optimization**: Handles cases where target frames span multiple GOPs
     * 
     * ## Memory Management Strategy:
     * - **Packet Array Management**: Maintains `packet_array` with proper lifetime tracking
     * - **Adaptive Extraction Control**: Uses `use_dd` flag to control data extraction scope:
     *   - `true`: Complete GOP extraction for maximum decode context (higher memory usage)
     *   - `false`: Minimal packet extraction to target frames only (optimized memory usage)
     * - **Zero-Copy Design**: Packets are allocated once and passed by pointer to avoid copies
     * 
     * ## Thread Safety & Concurrency:
     * - **Thread-Safe Queue**: Uses `ConcurrentQueue` for lock-free packet handoff
     * - **Read-Only Demuxer**: Demuxer is used in read-only mode during packet extraction
     * - **Atomic Operations**: GOP metadata updates are performed atomically
     * - **Signal Completion**: Queue receives completion signal `(nullptr, -1, 0)` when done
     * 
     * @param demuxer Initialized PyNvGopDemuxer instance with opened video stream.
     *                Must be properly initialized with valid video source.
     *                The demuxer provides access to video metadata, seeking capabilities,
     *                and packet extraction functionality.
     * 
     * @param packet_queue Thread-safe concurrent queue for high-performance packet distribution to decoder threads.
     *                     
     *                     **Data Structure Design:**
     *                     - Type: `ConcurrentQueue<std::tuple<uint8_t*, int, int>>`
     *                     - Tuple format: `(packet_data_ptr, packet_size_bytes, timestamp)`
     *                     - Lock-free implementation for maximum throughput
     *                     - MPMC (Multi-Producer Multi-Consumer) queue supporting concurrent access
     *                     
     *                     **Packet Data Format:**
     *                     - `uint8_t* data_ptr`: Pointer to raw H.264/H.265 packet data in NAL unit format
     *                     - `int size_bytes`: Size of packet data in bytes (including NAL headers)
     *                     - `int timestamp`: Custom timestamp for frame identification and ordering
     *                       - **Target Frame Timestamp**: `frame_idx * 2 + 1` for frames that need to be decoded
     *                       - **Dependency Frame Timestamp**: `frame_idx * 2` for frames needed for decode context
     *                       - This encoding allows easy identification of target vs dependency frames
     *                       - Odd timestamps (target frames) have higher priority in processing
     *                       - Even timestamps (dependency frames) provide necessary decode context
     *                     
     *                     **Queue Lifecycle & Signaling:**
     *                     - **Active Phase**: Demuxer pushes valid packets `(valid_ptr, size > 0, timestamp)`
     *                     - **Completion Signal**: Demuxer pushes termination tuple `(nullptr, -1, 0)`
     *                     - **Consumer Behavior**: Decoder threads poll until receiving termination signal
     *                     - **Memory Ownership**: Queue holds pointers only; actual packet memory managed externally
     *                     
     *                     **Concurrency & Performance:**
     *                     - **Lock-Free Operations**: Enqueue/dequeue operations are atomic and lock-free
     *                     - **Cache-Friendly Design**: Minimizes cache line contention between threads
     *                     - **Backpressure Handling**: Automatically manages flow control for high-throughput scenarios
     *                     - **NUMA Awareness**: Optimized for multi-socket systems with proper memory locality
     *                     
     *                     **Threading Model:**
     *                     ```
     *                     Producer (Demux Thread):
     *                       while(has_packets) {
     *                         packet = extract_packet();
     *                         // Custom timestamp encoding:
     *                         // - Target frames: frame_idx * 2 + 1 (odd numbers)
     *                         // - Dependency frames: frame_idx * 2 (even numbers)
     *                         int custom_timestamp = is_target_frame ? 
     *                                               (frame_idx * 2 + 1) : (frame_idx * 2);
     *                         queue.enqueue({packet.data, packet.size, custom_timestamp});
     *                       }
     *                       queue.enqueue({nullptr, -1, 0}); // Signal completion
     *                     
     *                     Consumer (Decode Thread):
     *                       while(true) {
     *                         auto [data, size, timestamp] = queue.dequeue();
     *                         if(data == nullptr && size == -1) break; // Termination
     *                         
     *                         // Identify frame type from timestamp
     *                         bool is_target_frame = (timestamp % 2 == 1);
     *                         int frame_idx = is_target_frame ? (timestamp - 1) / 2 : timestamp / 2;
     *                         
     *                         decode_packet(data, size, timestamp, is_target_frame);
     *                       }
     *                     ```
     *
     *                     Multiple decoder threads can safely consume from this queue concurrently,
     *                     enabling efficient parallel processing of video packets with minimal
     *                     synchronization overhead.
     * 
     * @param sorted_frame_ids Vector of target frame IDs in ascending order.
     *                         These are the frames that need to be decoded.
     *                         The function will extract minimal packet set required to decode these frames.
     *                         Frame IDs must be valid for the video stream and properly sorted.
     * 
     * @param first_frame_ids Output vector containing the first frame ID of each GOP that contains
     *                        target frames. This metadata enables GOP-aware decoding optimizations
     *                        and helps decoders understand GOP boundaries for proper state management.
     *                        Size will match the number of relevant GOPs.
     * 
     * @param gop_lens Output vector containing the length of each GOP that contains target frames.
     *                 Provides essential GOP structure information for decoder optimization.
     *                 Combined with `first_frame_ids`, enables efficient GOP-based processing.
     *                 Size will match `first_frame_ids.size()`.
     * 
     * @param packet_array Output vector containing raw packet data pointers.
     *                     Each pointer references memory-managed packet data that will be
     *                     consumed by decoder threads via the packet queue.
     *                     Packet lifetime is managed internally and remains valid until
     *                     all decoder operations complete.
     * 
     * @param use_dd Flag controlling packet extraction scope in demuxer-decoder separate mode:
     *               - `true`: Extract complete GOP data (full Group of Pictures)
     *                 - Reads all packets from GOP start to GOP end, regardless of target frames
     *                 - Ensures complete decode context for optimal decoder performance
     *                 - Used in demuxer-decoder separate mode for maximum decode quality
     *                 - Higher memory usage but better decode reliability and performance
     *               - `false`: Extract minimal packet range (up to target frames only)
     *                 - Reads packets only up to the last target frame in each GOP
     *                 - Minimizes memory usage and I/O by avoiding unnecessary packet extraction
     *                 - May result in incomplete decode context for some edge cases
     *                 - Optimized for memory-constrained scenarios with specific frame targets
     * 
     * @see DecProc() for the corresponding decoding pipeline that consumes packet queue output
     * @see CreateDemuxer() for proper demuxer initialization
     */
    static void DemuxGopProc(PyNvGopDemuxer* demuxer,
                             ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                             std::vector<int> sorted_frame_ids, std::vector<int>& first_frame_ids,
                             std::vector<int>& gop_lens,
                             std::vector<std::unique_ptr<uint8_t[]>>& packet_array, bool use_dd);

    /**
     * Decode video packets and convert frames to specified output format
     * 
     * This is a high-performance, templated decoding function that processes video packets
     * from a concurrent queue and outputs frames in either RGB or YUV format. The function
     * is designed for both single-threaded and multi-threaded environments with optimal
     * GPU resource utilization.
     * 
     * ## Key Features:
     * 1. **Template-Based Processing**: Uses C++17 `if constexpr` to compile different code
     *    paths for RGB (`RGBFrame`) vs YUV (`DecodedFrameExt`) output without runtime overhead
     * 2. **Concurrent Queue Processing**: Consumes packets from a thread-safe queue, enabling
     *    parallel demuxing and decoding workflows. If you ues a demuxer-free or demuxer-decode
     *    separate method, you should construct a packet queue and push packets to it by yourself.
     * 3. **Smart Decoder State Management**: Implements intelligent decoder state continuity
     *    optimization through `last_decoded_frame_info` tracking for seamless cross-session decoding
     * 
     * ## Processing Pipeline:
     * 1. **Initialization**: Sets CUDA context and flushes decoder if needed
     * 2. **Packet Processing**: Consumes packets from queue until completion signal
     * 3. **Frame Decoding**: Calls NvDecoder with packet data and retrieval flags
     * 4. **Format Conversion**: 
     *    - RGB: Calls `GetRGBFromFrame()` with color space conversion
     *    - YUV: Calls `GetYUVFromFrame()` with direct GPU memory copy
     * 5. **State Updates**: Maintains decoder state for cross-session optimization
     * 6. **Validation**: Ensures output frame count matches expected frame count
     * 
     * - **Decoder Buffer Continuity**: If `last_decoded_frame_info.filename` matches the current
     *   `filename` parameter and is non-empty, the decoder buffer is NOT flushed, allowing
     *   continuation from the previous decode session
     * - **Automatic Flush Control**: If `last_decoded_frame_info.filename` is empty or differs
     *   from current filename, decoder buffers are flushed to ensure clean decode state
     * - **Manual State Control**: Users can manually manipulate `last_decoded_frame_info` before
     *   calling `DecProc` to control continuous decode behavior:
     *   ```cpp
     *   // Force decoder flush (start fresh)
     *   last_decoded_frame_info.filename = "";
     *   DecProc<RGBFrame>(..., last_decoded_frame_info);
     *   
     *   // Enable continuous decode from previous session
     *   last_decoded_frame_info.filename = "same_video.mp4";
     *   DecProc<RGBFrame>(..., "same_video.mp4", ..., last_decoded_frame_info);
     *   ```
     * - **State Tracking**: The function updates `last_decoded_frame_info` with the last
     *   successfully decoded frame information for subsequent calls
     * 
     * @tparam OutputFrame Either `RGBFrame` for RGB output or `DecodedFrameExt` for YUV output
     * @param color_range AVColorRange specifying the color range (JPEG/full vs MPEG/limited)
     * @param decoder Initialized NvDecoder instance with proper CUDA context
     * @param output_frames Output vector to store decoded frames (resized automatically)
     * @param p_frames Pre-allocated GPU memory buffers for frame data (one per expected frame)
     * @param packet_queue Thread-safe queue containing packets as (data_ptr, size, timestamp) tuples.
     *                     Queue signals completion with (nullptr, -1, 0) tuple
     * @param sorted_frame_ids Expected frame IDs in ascending order for validation and filtering
     * @param use_bgr_format If true, outputs BGR instead of RGB (RGB template only, ignored for YUV)
     * @param filename Source filename for error reporting and logging context
     * @param last_decoded_frame_info Reference to decoder state tracking structure for continuous decode
     *                               optimization. Controls decoder buffer flushing behavior:
     *                               - If `filename` field is empty: Forces decoder flush (clean start)
     *                               - If `filename` field matches current file: Enables continuous decode
     *                               - If `filename` field differs: Forces decoder flush and state reset
     *                               - Updated automatically with last decoded frame info after successful decode
     * 
     * @throws std::runtime_error If frame conversion fails, context setup fails, or frame count mismatch
     * 
     * ## Usage Examples:
     * ```cpp
     * // Basic RGB output with BGR format
     * std::vector<RGBFrame> rgb_frames;
     * DecProc<RGBFrame>(color_range, decoder, rgb_frames, frame_buffers, 
     *                   packet_queue, frame_ids, true, "video.mp4", state_info);
     * 
     * // YUV output (minimal overhead)
     * std::vector<DecodedFrameExt> yuv_frames;
     * DecProc<DecodedFrameExt>(color_range, decoder, yuv_frames, frame_buffers,
     *                          packet_queue, frame_ids, false, "video.mp4", state_info);
     * 
     * // Continuous decode optimization example
     * LastDecodedFrameInfo state_info;
     * 
     * // First decode session - will flush decoder
     * state_info.filename = "";  // Force fresh start
     * DecProc<RGBFrame>(color_range, decoder, frames1, buffers1, queue1, 
     *                   ids1, false, "video.mp4", state_info);
     * 
     * // Second decode session - continuous decode (no flush)
     * // state_info.filename is now "video.mp4" from previous call
     * DecProc<RGBFrame>(color_range, decoder, frames2, buffers2, queue2,
     *                   ids2, false, "video.mp4", state_info);  // No decoder flush
     * 
     * // Third decode session - different file, will flush decoder
     * DecProc<RGBFrame>(color_range, decoder, frames3, buffers3, queue3,
     *                   ids3, false, "other.mp4", state_info);  // Decoder flushed
     * ```
     * 
     * @note This function expects packets to be in decode order and properly formatted
     *       for the target decoder. Packet queue must be properly populated by demuxing
     *       operations before calling this function.
     */
    template <typename OutputFrame>
    static void DecProc(AVColorRange color_range, NvDecoder* decoder, std::vector<OutputFrame>& output_frames,
                        std::vector<uint8_t*> p_frames,
                        ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                        const std::vector<int> sorted_frame_ids, bool use_bgr_format,
                        const std::string& filename, LastDecodedFrameInfo& last_decoded_frame_info);

    /**
     * Create and initialize a video demuxer for GOP-based decoding
     * 
     * This static method creates a PyNvGopDemuxer instance optimized for efficient
     * video demuxing and GOP (Group of Pictures) analysis. It supports both fast
     * initialization using pre-computed stream information and standard initialization.
     * 
     * ## Key Features:
     * 1. **Fast Initialization**: When FastStreamInfo is provided, bypasses expensive
     *    FFmpeg stream analysis for significant performance improvement in batch processing
     * 2. **Automatic Resource Management**: Creates demuxer with proper RAII semantics
     *    using std::unique_ptr for automatic cleanup
     * 
     * @param demuxer Reference to unique_ptr where the created demuxer will be stored.
     *                The previous demuxer (if any) is automatically destroyed.
     * @param filename Path to the video file to demux. Must be a valid video file
     *                 accessible by FFmpeg.
     * @param fastStreamInfo Optional pointer to pre-computed stream information.
     *                       If provided, enables fast initialization mode.
     *                       If nullptr, performs standard FFmpeg initialization.
     * 
     * @return void
     */
    static void CreateDemuxer(std::unique_ptr<PyNvGopDemuxer>& demuxer, const std::string& filename,
                              const FastStreamInfo* fastStreamInfo);

    /**
   * Convert a decoded frame to a CAI memory view format
   * @param frame_in Input decoded frame to be converted
   * @param out_buffer Output buffer to store the converted frame data
   * @return CAIMemoryView containing the converted frame data
   */
    CAIMemoryView ConvertFrame(const DecodedFrameExt& frame_in, uint8_t* out_buffer);

    /**
   * Process file paths and frame IDs to create necessary data structures
   * @param filepaths List of file paths
   * @param frame_ids List of frame IDs
   * @param fileFrameIds Output map of file to frame IDs
   * @param fileInverseFrameIds Output map of file to inverse frame IDs
   * @param fileList Output list of unique files
   * @return 0 if successful, non-zero error code otherwise
   */
    int processFrameIds(const std::vector<std::string>& filepaths, const std::vector<int>& frame_ids,
                        std::map<std::string, std::vector<int>>& fileFrameIds,
                        std::map<std::string, std::vector<int>>& fileInverseFrameIds,
                        std::vector<std::string>& fileList);

    /**
   * Initialize demuxers and allocate memory for video files
   * @param filepaths List of file paths
   * @param demuxers Output vector of initialized demuxers
   * @param fastStreamInfos Pointer to array of FastStreamInfo
   * @return 0 if successful, non-zero error code otherwise
   */
    int InitializeDemuxers(const std::vector<std::string>& filepaths,
                           std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
                           const FastStreamInfo* fastStreamInfos);

    /**
   * Initialize GPU memory pool
   * @param heights Vector of heights
   * @param widths Vector of widths
   * @param frame_sizes Vector of frame sizes
   * @param convert_to_rgb Flag indicating whether to convert to RGB
   * @return 0 if successful, non-zero error code otherwise
   */
    int InitGpuMemPool(const std::vector<int>& heights, const std::vector<int>& widths,
                       const std::vector<int>& frame_sizes, bool convert_to_rgb);

    int GetFileFrameBuffers(const std::vector<int>* widths, const std::vector<int>* heights,
                            const std::vector<int>* frame_sizes, bool convert_to_rgb,
                            std::vector<std::vector<uint8_t*>>& per_file_frame_buffers);

    /**
     * Extract and process GOP information for VFR (Variable Frame Rate) videos
     * 
     * This method is specifically designed for Variable Frame Rate (VFR) videos and provides
     * essential GOP (Group of Pictures) metadata required for accurate random access decoding.
     * 
     * @param demuxer The initialized VFR demuxer instance. Must be valid, successfully opened.
     * @param sorted_frame_ids Input/output vector for sorted frame IDs that need GOP processing
     * @param first_frame_ids Output vector containing the first frame ID of each GOP
     * @param gop_length Output vector containing the length of each GOP
     * 
     * @return 0 if successful, non-zero error code if GOP extraction fails
     */
    int ExtractAndProcessGopInfo(const std::unique_ptr<PyNvGopDemuxer>& demuxer,
                                 std::vector<int>& sorted_frame_ids, std::vector<int>& first_frame_ids,
                                 std::vector<int>& gop_length);

    static Pixel_Format GetNativeFormat(const cudaVideoSurfaceFormat inputFormat);

   private:
    int max_num_files = 0;

    bool suppress_no_color_range_given_warning = false;

    bool destroy_context = false;
    CUcontext cu_context = NULL;
    CUstream cu_stream = NULL;
    int gpu_id = 0;

    std::vector<std::unique_ptr<NvDecoder>> vdec;
    std::vector<LastDecodedFrameInfo> last_decoded_frame_infos;

    GPUMemoryPool gpu_mem_pool;

    // Thread runners for reuse
    std::vector<ThreadRunner> demux_runners;
    std::vector<ThreadRunner> decode_runners;
    std::vector<ThreadRunner> merge_runners;

    // Lazy loading functions
    void ensureCudaContextInitialized();
    void ensureDemuxRunnersInitialized();
    void ensureDecodeRunnersInitialized();
    void ensureMergeRunnersInitialized();

    /**
     * Internal implementation for GOP extraction
     * 
     * This method extracts GOP data from video files and returns the intermediate data structures
     * needed to create SerializedPacketBundles. It is used by both get_gop() and get_gop_list()
     * to avoid code duplication.
     * 
     * @param filepaths Vector of video file paths
     * @param frame_ids Vector of frame IDs corresponding to each filepath
     * @param fastStreamInfos Optional array of FastStreamInfo for performance optimization
     * @param demuxers Output vector of initialized demuxers
     * @param vpacket_queue Output vector of packet queues
     * @param vpacket_array Output vector of packet arrays
     * @param all_gop_lens Output vector of GOP lengths for each video
     * @param all_first_frame_ids Output vector of first frame IDs for each video
     */
    void get_gop_internal(
        const std::vector<std::string>& filepaths, const std::vector<int> frame_ids,
        const FastStreamInfo* fastStreamInfos, std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
        std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
        std::vector<std::vector<std::unique_ptr<uint8_t[]>>>& vpacket_array,
        std::vector<std::vector<int>>& all_gop_lens, std::vector<std::vector<int>>& all_first_frame_ids);

    /**
     * Create a SerializedPacketBundle from extracted packet data
     * 
     * This function serializes video packet data into a self-contained binary format for 
     * efficient storage and transmission. The binary data includes an embedded offset table
     * in the header for O(1) random access to any frame data.
     * 
     * Binary format structure:
     * - Header: uint32_t total_frames + size_t[total_frames] frame_offsets
     * - Frame data: metadata + packet_sizes_array + decode_indices_array + binary_data (repeated)
     * 
     * The embedded offset table enables efficient random access and parallel processing
     * without requiring external metadata. Users can parse the header once to obtain
     * all frame offsets, then jump directly to any frame's data.
     * 
     * @param total_frames Number of frames to process
     * @param demuxers Vector of demuxers containing video metadata (width, height, codec, etc.)
     * @param all_gop_lens Vector of GOP lengths for each frame
     * @param all_first_frame_ids Vector of first frame IDs for each frame
     * @return SerializedPacketBundle containing self-contained binary data with embedded offsets
     */
    SerializedPacketBundle createSerializedPacketBundle(
        size_t total_frames, const std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
        const std::vector<std::vector<int>>& all_gop_lens,
        const std::vector<std::vector<int>>& all_first_frame_ids,
        const std::vector<std::unique_ptr<ConcurrentQueue<std::tuple<uint8_t*, int, int>>>>& vpacket_queue,
        const std::vector<std::vector<std::unique_ptr<uint8_t[]>>>& vpacket_array);

    /**
     * Convert decoded frame to RGB format
     * @param decoder The decoder instance
     * @param pFrame Pointer to the decoded frame data
     * @param pFrame_buffer Pointer to the output RGB buffer
     * @param color_range Color range of the input frame
     * @param use_bgr_format Whether to use BGR format instead of RGB
     * @param rgb_frame Output reference to construct the RGB frame directly
     * @return 0 on success, -1 on error
     */
    static int GetRGBFromFrame(NvDecoder* decoder, const uint8_t* pFrame, uint8_t* pFrame_buffer,
                               AVColorRange color_range, bool use_bgr_format, RGBFrame& rgb_frame);

    /**
     * Create a DecodedFrameExt object from decoded frame data
     * @param decoder The decoder instance
     * @param pFrame Pointer to the decoded frame data
     * @param pFrame_buffer Pointer to the output frame buffer
     * @param color_range Color range of the input frame
     * @param timestamp Timestamp of the frame
     * @param decoded_frame Output reference to construct the DecodedFrameExt object
     * @return 0 on success, -1 on error
     */
    static int GetYUVFromFrame(NvDecoder* decoder, const uint8_t* pFrame, uint8_t* pFrame_buffer,
                               AVColorRange color_range, int64_t timestamp, DecodedFrameExt& decoded_frame);
};

/**
 * Save binary data to a file
 * @param data Pointer to the binary data to save
 * @param size Size of the binary data in bytes
 * @param dst_filepath The destination file path where data will be saved
 */
void SaveBinaryDataToFile(const uint8_t* data, size_t size, const std::string& dst_filepath);
