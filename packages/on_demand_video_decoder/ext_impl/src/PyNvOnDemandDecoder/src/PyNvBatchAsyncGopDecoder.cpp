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

#include "PyNvBatchAsyncGopDecoder.hpp"
#include "FrameOutput.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nvtx3/nvtx3.hpp"

namespace py = pybind11;
namespace frame_output = accvlab::on_demand_video_decoder::internal::frame_output;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PyNvBatchAsyncGopDecoder::PyNvBatchAsyncGopDecoder(int maxfiles, int max_frames_per_decode_call, int iGpu,
                                                   bool suppressNoColorRangeWarning)
    : suppress_no_color_range_warning_(suppressNoColorRangeWarning),
      gpu_id_(iGpu),
      maxfiles_(maxfiles),
      max_frames_per_decode_call_(max_frames_per_decode_call),
      has_pending_task_(false) {
    if (maxfiles <= 0) throw std::invalid_argument("maxfiles must be > 0, got " + std::to_string(maxfiles));
    if (max_frames_per_decode_call <= 0)
        throw std::invalid_argument("max_frames_per_decode_call must be > 0, got " +
                                    std::to_string(max_frames_per_decode_call));

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu)
        throw std::invalid_argument("GPU ordinal " + std::to_string(iGpu) + " out of range [0, " +
                                    std::to_string(nGpu - 1) + "]");

    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, gpu_id_));
    ck(cuDevicePrimaryCtxRetain(&cu_context_, cuDevice));
    destroy_context_ = true;

    if (!cu_context_)
        throw std::domain_error(
            "[ERROR] Failed to create a CUDA context. Create a cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");

    ck(cuCtxPushCurrent(cu_context_));
    ck(cuStreamCreate(&cu_stream_, CU_STREAM_DEFAULT));
    CUcontext popped = nullptr;
    ck(cuCtxPopCurrent(&popped));

    // One internal decoder per video: size the GOP decoder to maxfiles (V) and let
    // it share this object's CUDA stream.  The worker issues F transposed decode calls
    // (frame-slot f of all V videos at once).  Frames within a GOP are inter-dependent
    // and can only be decoded sequentially, so continuous-decode state (kept per slot
    // across the F calls) makes each video's GOP decode exactly once instead of O(F^2).
    // Sharing cu_stream_ keeps decode, color-convert and the aggregator D2D copies on
    // one ordered stream, so the F calls need no intermediate sync — only one at the end.
    gop_dec_ = std::make_unique<PyNvGopDecoder>(maxfiles_, iGpu, suppressNoColorRangeWarning, cu_stream_);

    rgb_agg_pools_.resize(maxfiles_);
    yuv_agg_pools_.resize(maxfiles_);
}

PyNvBatchAsyncGopDecoder::~PyNvBatchAsyncGopDecoder() {
    bool need_join = false;
    {
        std::lock_guard<std::mutex> lk(async_mutex_);
        need_join = has_pending_task_;
    }
    if (need_join) {
        decode_worker_.join();
    }
    result_queue_.clear();

    for (auto& p : rgb_agg_pools_) p.HardRelease();
    for (auto& p : yuv_agg_pools_) p.HardRelease();

    // Destroy the GOP decoder first: it borrows cu_stream_ (external_stream) and its
    // NvDecoder / mempool teardown must run while that stream is still alive.
    gop_dec_.reset();

    if (cu_stream_) {
        cuCtxPushCurrent(cu_context_);
        cuStreamDestroy(cu_stream_);
        cu_stream_ = nullptr;
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
    }

    if (destroy_context_ && cu_context_) {
        CUdevice cuDevice = 0;
        cuDeviceGet(&cuDevice, gpu_id_);
        cuDevicePrimaryCtxRelease(cuDevice);
        cu_context_ = nullptr;
        destroy_context_ = false;
    }
}

// ---------------------------------------------------------------------------
// Public API — release helpers
// ---------------------------------------------------------------------------

void PyNvBatchAsyncGopDecoder::release_device_memory() {
    // Join any pending worker before touching the pools it may still be writing to.
    // Mirrors the pattern in the destructor.
    {
        std::unique_lock<std::mutex> lock(async_mutex_);
        if (has_pending_task_) {
            lock.unlock();
            decode_worker_.join();
            lock.lock();
            has_pending_task_ = false;
        }
        // The queued result's RGBFrame/YUV views point into pool memory that is
        // about to be freed — discard it so GetBuffer does not return dangling pointers.
        result_queue_.clear();
    }
    for (auto& p : rgb_agg_pools_) p.HardRelease();
    for (auto& p : yuv_agg_pools_) p.HardRelease();
    if (gop_dec_) gop_dec_->ReleaseMemPools();
}

void PyNvBatchAsyncGopDecoder::release_decoder() {
    // Join any pending worker before releasing gop_dec_, which the worker borrows.
    // Mirrors the pattern in the destructor.
    bool need_join = false;
    {
        std::lock_guard<std::mutex> lk(async_mutex_);
        need_join = has_pending_task_;
    }
    if (need_join) {
        decode_worker_.join();
        std::lock_guard<std::mutex> lk(async_mutex_);
        has_pending_task_ = false;
    }
    // Do NOT clear result_queue_: the worker's result lives in rgb_agg_pools_ (not in
    // gop_dec_), so already-decoded frames remain valid after the decoder is released.
    if (gop_dec_) gop_dec_->ReleaseDecoder();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::string PyNvBatchAsyncGopDecoder::generate_request_key(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d) {
    std::ostringstream oss;
    for (size_t v = 0; v < filepaths.size(); ++v) {
        oss << filepaths[v] << ":[";
        for (size_t f = 0; f < frame_ids_2d[v].size(); ++f) {
            if (f) oss << ',';
            oss << frame_ids_2d[v][f];
        }
        oss << "] ";
    }
    return oss.str();
}

bool PyNvBatchAsyncGopDecoder::validate_request(const DecodeResultGOP& result,
                                                const std::vector<std::string>& filepaths,
                                                const std::vector<std::vector<int>>& frame_ids_2d) {
    return result.file_path_list == filepaths && result.frame_id_list_2d == frame_ids_2d;
}

void PyNvBatchAsyncGopDecoder::validate_decode_input(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
    const std::vector<std::vector<std::vector<uint8_t>>>& numpy_datas) {
    const int V = static_cast<int>(filepaths.size());
    if (V == 0) throw std::invalid_argument("filepaths must not be empty");
    if (V > maxfiles_)
        throw std::invalid_argument("number of files (" + std::to_string(V) + ") exceeds maxfiles (" +
                                    std::to_string(maxfiles_) + ")");
    if (static_cast<int>(frame_ids_2d.size()) != V)
        throw std::invalid_argument("frame_ids_2d outer length (" + std::to_string(frame_ids_2d.size()) +
                                    ") != filepaths length (" + std::to_string(V) + ")");
    if (static_cast<int>(numpy_datas.size()) != V)
        throw std::invalid_argument("numpy_datas outer length (" + std::to_string(numpy_datas.size()) +
                                    ") != filepaths length (" + std::to_string(V) + ")");

    const int F = static_cast<int>(frame_ids_2d[0].size());
    if (F == 0) throw std::invalid_argument("frame_ids_2d inner lists must not be empty");
    if (F > max_frames_per_decode_call_)
        throw std::invalid_argument("frame count (" + std::to_string(F) +
                                    ") exceeds max_frames_per_decode_call (" +
                                    std::to_string(max_frames_per_decode_call_) + ")");

    for (int v = 1; v < V; ++v) {
        if (static_cast<int>(frame_ids_2d[v].size()) != F)
            throw std::invalid_argument(
                "frame_ids_2d inner lengths are not all equal (jagged input not supported): "
                "frame_ids_2d[0].size()=" +
                std::to_string(F) + " but frame_ids_2d[" + std::to_string(v) +
                "].size()=" + std::to_string(frame_ids_2d[v].size()));
    }

    for (int v = 0; v < V; ++v) {
        if (numpy_datas[v].empty())
            throw std::invalid_argument(
                "numpy_datas[" + std::to_string(v) +
                "] is empty — at least one serialized GOP bundle is required per video");
    }
}

// ---------------------------------------------------------------------------
// Common async submission path
// ---------------------------------------------------------------------------

void PyNvBatchAsyncGopDecoder::submit_work(std::vector<std::vector<std::vector<uint8_t>>> numpy_datas,
                                           std::vector<std::string> filepaths,
                                           std::vector<std::vector<int>> frame_ids_2d, bool as_bgr,
                                           bool is_rgb, bool enable_gop_dependency_graph_optimization) {
    std::unique_lock<std::mutex> lock(async_mutex_);

    if (has_pending_task_) {
        std::cerr << "[WARNING] PyNvBatchAsyncGopDecoder: A previous async decode task is still "
                     "running. Waiting for it to complete before starting the new task."
                  << std::endl;
        lock.unlock();
        decode_worker_.join();
        lock.lock();
        has_pending_task_ = false;
    }
    result_queue_.clear();

    has_pending_task_ = true;
    decode_worker_.start([this, numpy_datas = std::move(numpy_datas), filepaths = std::move(filepaths),
                          frame_ids_2d = std::move(frame_ids_2d), as_bgr, is_rgb,
                          enable_gop_dependency_graph_optimization]() mutable {
        DecodeResultGOP result;
        result.file_path_list = filepaths;
        result.frame_id_list_2d = frame_ids_2d;
        result.as_bgr = as_bgr;
        result.is_rgb = is_rgb;
        result.is_ready = false;

        bool ctx_pushed = false;
        try {
            CUDA_DRVAPI_CALL(cuCtxPushCurrent(cu_context_));
            ctx_pushed = true;
        } catch (...) {
            result.exception = std::current_exception();
            result.is_ready = true;
            {
                std::lock_guard<std::mutex> lk(async_mutex_);
                result_queue_.push_back(std::move(result));
                has_pending_task_ = false;
            }
            result_cv_.notify_all();
            return;
        }

        try {
            nvtxRangePushA("GOP Batch 2D Decode Worker");

            const int V = static_cast<int>(filepaths.size());

            if (is_rgb) {
                result.decoded_rgb_frames.assign(V, std::vector<RGBFrame>{});
            } else {
                result.decoded_yuv_frames.assign(V, std::vector<DecodedFrameExt>{});
            }

            // All videos share one frame count F (jagged input is rejected upstream).
            const int F = static_cast<int>(frame_ids_2d[0].size());

            // ----------------------------------------------------------------
            // Phase 1 — per video: sort frame_ids and resolve each to the GOP
            // bundle covering it.  Frames within a GOP are inter-dependent (P/B
            // frames reference earlier ones) so a video's frames can only decode
            // sequentially — the parallelism unit is the video, not the frame.
            // perm_2d[v][f] maps the f-th sorted frame back to its caller index.
            // ----------------------------------------------------------------
            std::vector<std::vector<int>> perm_2d(V);
            std::vector<std::vector<int>> sorted_fids_2d(V);       // [v][f] ascending frame id
            std::vector<std::vector<const uint8_t*>> datas_2d(V);  // [v][f] covering bundle ptr
            std::vector<std::vector<size_t>> sizes_2d(V);          // [v][f] covering bundle size
            std::vector<std::vector<size_t>> gop_indices_2d(V);    // [v][f] covering GOP index
            std::vector<std::vector<std::vector<int>>> target_fids_by_gop_2d(V);

            struct GopRange {
                int first_frame_id;
                int gop_len;
                const uint8_t* data;
                size_t size;
            };

            for (int v = 0; v < V; ++v) {
                const std::vector<int>& fids_v = frame_ids_2d[v];

                std::vector<int>& perm = perm_2d[v];
                perm.resize(F);
                std::iota(perm.begin(), perm.end(), 0);
                std::sort(perm.begin(), perm.end(),
                          [&fids_v](int a, int b) { return fids_v[a] < fids_v[b]; });

                // Parse this video's bundles into [first_frame_id, +gop_len) ranges.
                std::vector<GopRange> gop_ranges;
                for (const auto& bundle : numpy_datas[v]) {
                    std::vector<int> cr, ci, wi, hi, fs, gl, ffid;
                    std::vector<std::vector<int>> pb, di;
                    std::vector<const uint8_t*> pp;
                    std::vector<size_t> ps;
                    PyNvGopDecoder::parseSerializedPacketData(bundle.data(), bundle.size(), cr, ci, wi, hi,
                                                              fs, gl, ffid, pb, di, pp, ps);
                    for (size_t k = 0; k < ffid.size(); ++k) {
                        gop_ranges.push_back({ffid[k], gl[k], bundle.data(), bundle.size()});
                    }
                }

                sorted_fids_2d[v].resize(F);
                datas_2d[v].resize(F);
                sizes_2d[v].resize(F);
                gop_indices_2d[v].resize(F);
                target_fids_by_gop_2d[v].resize(gop_ranges.size());
                for (int f = 0; f < F; ++f) {
                    const int fid = fids_v[perm[f]];
                    sorted_fids_2d[v][f] = fid;
                    bool found = false;
                    for (size_t g = 0; g < gop_ranges.size(); ++g) {
                        const auto& gr = gop_ranges[g];
                        if (fid >= gr.first_frame_id && fid < gr.first_frame_id + gr.gop_len) {
                            datas_2d[v][f] = gr.data;
                            sizes_2d[v][f] = gr.size;
                            gop_indices_2d[v][f] = g;
                            target_fids_by_gop_2d[v][g].push_back(fid);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        throw std::runtime_error(
                            "PyNvBatchAsyncGopDecoder: no serialized GOP bundle covers frame " +
                            std::to_string(fid) + " for video " + filepaths[v]);
                    }
                }

                if (is_rgb) {
                    result.decoded_rgb_frames[v].resize(F);
                } else {
                    result.decoded_yuv_frames[v].resize(F);
                }
            }

            // ----------------------------------------------------------------
            // Phase 2 — F transposed decode calls.  Call f decodes frame-slot f
            // of every video at once; decode_from_gop_list runs the V videos in
            // parallel (one decoder each).  Across the F calls each video keeps
            // the same decoder slot with ascending frame ids, so continuous-
            // decode resumes from where the previous call stopped — each GOP is
            // decoded once, not once per frame (no O(F^2) re-decode).
            //
            // Every call uses skip_final_sync=true: decode, color-convert and the
            // aggregator D2D copies all run on the shared cu_stream_, so call f's
            // copies are ordered before call f+1 reuses gop_dec_'s pool.  One
            // cuStreamSynchronize after the loop drains all F calls.
            //
            // TODO: drop the D2D copy by decoding straight into the aggregator
            // pools (needs GPUMemoryPool move semantics + PyNvGopDecoder::take_pool()).
            // ----------------------------------------------------------------
            std::vector<const uint8_t*> datas_f(V);
            std::vector<size_t> sizes_f(V);
            std::vector<int> frame_ids_f(V);
            std::vector<std::vector<int>> planned_target_frame_ids_f(V);

            for (int f = 0; f < F; ++f) {
                for (int v = 0; v < V; ++v) {
                    datas_f[v] = datas_2d[v][f];
                    sizes_f[v] = sizes_2d[v][f];
                    frame_ids_f[v] = sorted_fids_2d[v][f];
                    planned_target_frame_ids_f[v] = target_fids_by_gop_2d[v][gop_indices_2d[v][f]];
                }

                const auto* dependency_graph_target_frame_ids =
                    enable_gop_dependency_graph_optimization ? &planned_target_frame_ids_f : nullptr;

                if (is_rgb) {
                    std::vector<RGBFrame> frames_f;
                    gop_dec_->decode_from_gop_list(datas_f, sizes_f, filepaths, frame_ids_f,
                                                   /*convert_to_rgb=*/true, as_bgr, nullptr, &frames_f,
                                                   /*skip_final_sync=*/true,
                                                   enable_gop_dependency_graph_optimization,
                                                   dependency_graph_target_frame_ids);
                    if (static_cast<int>(frames_f.size()) != V) {
                        std::ostringstream oss;
                        oss << "PyNvBatchAsyncGopDecoder: frame-slot " << f
                            << ": decode_from_gop_list returned " << frames_f.size() << " frames, expected "
                            << V;
                        throw std::runtime_error(oss.str());
                    }

                    for (int v = 0; v < V; ++v) {
                        const size_t H = std::get<0>(frames_f[v].shape);
                        const size_t W = std::get<1>(frames_f[v].shape);
                        const size_t frame_bytes =
                            frame_output::frame_bytes(frame_output::FrameOutputFormat::RGB8, H, W);

                        // Size each video's pool once (first frame-slot), then append.
                        if (f == 0)
                            rgb_agg_pools_[v].EnsureSizeAndSoftReset(static_cast<size_t>(F) * frame_bytes,
                                                                     false);

                        void* dst = rgb_agg_pools_[v].AddElement(frame_bytes);
                        // Place at the original (unsorted) output index.
                        result.decoded_rgb_frames[v][perm_2d[v][f]] =
                            frame_output::copy_rgb_frame(frames_f[v], reinterpret_cast<CUdeviceptr>(dst),
                                                         cu_stream_, as_bgr, /*is_async=*/true);
                    }

                } else {
                    std::vector<DecodedFrameExt> frames_f;
                    gop_dec_->decode_from_gop_list(
                        datas_f, sizes_f, filepaths, frame_ids_f,
                        /*convert_to_rgb=*/false, /*as_bgr=*/false, &frames_f, nullptr,
                        /*skip_final_sync=*/true, enable_gop_dependency_graph_optimization,
                        dependency_graph_target_frame_ids);
                    if (static_cast<int>(frames_f.size()) != V) {
                        std::ostringstream oss;
                        oss << "PyNvBatchAsyncGopDecoder: frame-slot " << f
                            << ": decode_from_gop_list returned " << frames_f.size()
                            << " YUV frames, expected " << V;
                        throw std::runtime_error(oss.str());
                    }

                    for (int v = 0; v < V; ++v) {
                        const Pixel_Format fmt = frames_f[v].format;
                        const size_t H = frames_f[v].views[0].shape[0];
                        const size_t W = frames_f[v].views[0].shape[1];
                        const size_t frame_bytes = frame_output::frame_bytes(
                            frame_output::output_format_from_pixel_format(fmt), H, W);

                        if (f == 0)
                            yuv_agg_pools_[v].EnsureSizeAndSoftReset(static_cast<size_t>(F) * frame_bytes,
                                                                     false);

                        void* dst = yuv_agg_pools_[v].AddElement(frame_bytes);
                        // Place at the original (unsorted) output index.
                        result.decoded_yuv_frames[v][perm_2d[v][f]] = frame_output::copy_yuv_frame(
                            frames_f[v], reinterpret_cast<CUdeviceptr>(dst), cu_stream_, /*is_async=*/true);
                    }
                }
            }

            CUDA_DRVAPI_CALL(cuStreamSynchronize(cu_stream_));
            result.is_ready = true;
            nvtxRangePop();

        } catch (...) {
            nvtxRangePop();
            // Drain work already queued on the shared stream (the F calls run with
            // skip_final_sync=true) before releasing the pools it may still touch.
            cuStreamSynchronize(cu_stream_);
            for (auto& p : rgb_agg_pools_) p.SoftRelease();
            for (auto& p : yuv_agg_pools_) p.SoftRelease();
            result.decoded_rgb_frames.clear();
            result.decoded_yuv_frames.clear();
            result.exception = std::current_exception();
            result.is_ready = true;
        }

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
        }

        {
            std::lock_guard<std::mutex> lk(async_mutex_);
            result_queue_.push_back(std::move(result));
            has_pending_task_ = false;
        }
        result_cv_.notify_all();
    });
}

// ---------------------------------------------------------------------------
// Public API — RGB
// ---------------------------------------------------------------------------

void PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGB(
    std::vector<std::vector<std::vector<uint8_t>>> numpy_datas, const std::vector<std::string>& filepaths,
    const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr,
    bool enable_gop_dependency_graph_optimization) {
    validate_decode_input(filepaths, frame_ids_2d, numpy_datas);
    submit_work(std::move(numpy_datas), filepaths, frame_ids_2d, as_bgr, /*is_rgb=*/true,
                enable_gop_dependency_graph_optimization);
}

std::vector<std::vector<RGBFrame>> PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
    bool as_bgr) {
    DecodeResultGOP result;
    {
        std::unique_lock<std::mutex> lock(async_mutex_);
        if (!has_pending_task_ && result_queue_.empty())
            throw std::runtime_error(
                "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: No pending task and "
                "buffer is empty. Call DecodeFromGOPListRGB first.");
        // The check, wait, and pop are all inside this critical section so that a
        // concurrent GetBuffer call cannot slip through the guard and then block
        // forever on an already-consumed result — it wakes up, finds the queue
        // empty, and gets a RuntimeError instead.
        result_cv_.wait(lock, [this] { return !result_queue_.empty() || !has_pending_task_; });
        if (result_queue_.empty())
            throw std::runtime_error(
                "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: result already consumed "
                "by another thread — concurrent GetBuffer calls are not supported.");
        result = std::move(result_queue_.front());
        result_queue_.pop_front();
    }

    if (!result.is_ready)
        throw std::runtime_error(
            "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: internal error — result "
            "not ready when popped.");
    if (result.exception) std::rethrow_exception(result.exception);
    if (!result.is_rgb)
        throw std::runtime_error(
            "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: buffered result is YUV, "
            "not RGB.");
    if (!validate_request(result, filepaths, frame_ids_2d)) {
        std::ostringstream oss;
        oss << "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: request mismatch. "
               "Expected: "
            << generate_request_key(filepaths, frame_ids_2d)
            << "Got: " << generate_request_key(result.file_path_list, result.frame_id_list_2d);
        throw std::runtime_error(oss.str());
    }
    if (result.as_bgr != as_bgr) {
        std::ostringstream oss;
        oss << "PyNvBatchAsyncGopDecoder::DecodeFromGOPListRGBGetBuffer: as_bgr mismatch — "
               "submitted with as_bgr="
            << (result.as_bgr ? "true" : "false")
            << " but GetBuffer called with as_bgr=" << (as_bgr ? "true" : "false");
        throw std::runtime_error(oss.str());
    }

    return result.decoded_rgb_frames;
}

// ---------------------------------------------------------------------------
// Public API — YUV
// ---------------------------------------------------------------------------

void PyNvBatchAsyncGopDecoder::DecodeFromGOPList(std::vector<std::vector<std::vector<uint8_t>>> numpy_datas,
                                                 const std::vector<std::string>& filepaths,
                                                 const std::vector<std::vector<int>>& frame_ids_2d,
                                                 bool enable_gop_dependency_graph_optimization) {
    validate_decode_input(filepaths, frame_ids_2d, numpy_datas);
    submit_work(std::move(numpy_datas), filepaths, frame_ids_2d, /*as_bgr=*/false,
                /*is_rgb=*/false, enable_gop_dependency_graph_optimization);
}

std::vector<std::vector<DecodedFrameExt>> PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d) {
    DecodeResultGOP result;
    {
        std::unique_lock<std::mutex> lock(async_mutex_);
        if (!has_pending_task_ && result_queue_.empty())
            throw std::runtime_error(
                "PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer: No pending task and buffer "
                "is empty. Call DecodeFromGOPList first.");
        // The check, wait, and pop are all inside this critical section so that a
        // concurrent GetBuffer call cannot slip through the guard and then block
        // forever on an already-consumed result — it wakes up, finds the queue
        // empty, and gets a RuntimeError instead.
        result_cv_.wait(lock, [this] { return !result_queue_.empty() || !has_pending_task_; });
        if (result_queue_.empty())
            throw std::runtime_error(
                "PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer: result already consumed "
                "by another thread — concurrent GetBuffer calls are not supported.");
        result = std::move(result_queue_.front());
        result_queue_.pop_front();
    }

    if (!result.is_ready)
        throw std::runtime_error(
            "PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer: internal error — result not "
            "ready when popped.");
    if (result.exception) std::rethrow_exception(result.exception);
    if (result.is_rgb)
        throw std::runtime_error(
            "PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer: buffered result is RGB, "
            "not YUV.");
    if (!validate_request(result, filepaths, frame_ids_2d)) {
        std::ostringstream oss;
        oss << "PyNvBatchAsyncGopDecoder::DecodeFromGOPListGetBuffer: request mismatch. "
               "Expected: "
            << generate_request_key(filepaths, frame_ids_2d)
            << "Got: " << generate_request_key(result.file_path_list, result.frame_id_list_2d);
        throw std::runtime_error(oss.str());
    }

    return result.decoded_yuv_frames;
}

// ---------------------------------------------------------------------------
// pybind11 bindings
// ---------------------------------------------------------------------------

void Init_PyNvBatchAsyncGopDecoder(py::module& m) {
    m.def(
        "CreateBatchAsyncGopDecoder",
        [](int maxfiles, int max_frames_per_decode_call, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvBatchAsyncGopDecoder>(maxfiles, max_frames_per_decode_call, iGpu,
                                                              suppressNoColorRangeWarning);
        },
        py::arg("maxfiles"), py::arg("max_frames_per_decode_call"), py::arg("iGpu") = 0,
        py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
            Create a PyNvBatchAsyncGopDecoder for 2D async GOP-based decoding.

            Accepts pre-serialized GOP bundles (from :func:`GetGOPList`) and decodes
            the requested frames asynchronously.  Both RGB and YUV output paths
            are provided.

            Args:
                maxfiles: Maximum number of videos per decode call (V upper bound).
                max_frames_per_decode_call: Maximum number of frames per video per
                    call (F upper bound).
                iGpu: GPU device id.
                suppressNoColorRangeWarning: Suppress warning when no color range
                    can be extracted.

            Returns:
                :class:`PyNvBatchAsyncGopDecoder` instance.

            Example:
                >>> gop_datas = decoder.GetGOPList(filepaths, frame_ids)  # [v] → SerializedPacketBundle
                >>> gop_dec = CreateBatchAsyncGopDecoder(maxfiles=6, max_frames_per_decode_call=4)
                >>> numpy_datas = [[np.frombuffer(b.data, dtype=np.uint8)] for b in gop_datas]
                >>> gop_dec.DecodeFromGOPListRGB(numpy_datas, filepaths, frame_ids_2d, as_bgr=False)
                >>> out = gop_dec.DecodeFromGOPListRGBGetBuffer(filepaths, frame_ids_2d, as_bgr=False)
        )pbdoc");

    py::class_<PyNvBatchAsyncGopDecoder, std::shared_ptr<PyNvBatchAsyncGopDecoder>>(
        m, "PyNvBatchAsyncGopDecoder", py::module_local(),
        R"pbdoc(
        GPU-accelerated 2D async GOP-based video decoder.

        Decodes V videos × F frames each from pre-serialized GOP bundles.  Submit
        with :meth:`DecodeFromGOPListRGB` / :meth:`DecodeFromGOPList`, retrieve
        with the matching ``GetBuffer`` method.

        Only one task (RGB **or** YUV) may be in flight at a time.  Calling a
        new Decode while the previous one is still running joins it first (a
        warning is printed to stderr).

        Do not instantiate directly — use :func:`CreateBatchAsyncGopDecoder`.

        .. seealso::
            :class:`PyNvBatchAsyncStreamReader` for stream-based (non-GOP) decoding.
        )pbdoc")
        .def(
            "DecodeFromGOPListRGB",
            [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec,
               const std::vector<std::vector<py::array_t<uint8_t>>>& numpy_data_arrays,
               const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
               bool as_bgr, bool enable_gop_dependency_graph_optimization) {
                // Copy bundle data while GIL is held (numpy buffer access requires GIL).
                std::vector<std::vector<std::vector<uint8_t>>> bundle_copies;
                bundle_copies.reserve(numpy_data_arrays.size());
                for (const auto& bundles_v : numpy_data_arrays) {
                    bundle_copies.emplace_back();
                    bundle_copies.back().reserve(bundles_v.size());
                    for (const auto& arr : bundles_v) {
                        auto buf = arr.request();
                        const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
                        bundle_copies.back().emplace_back(ptr, ptr + buf.size);
                    }
                }
                {
                    py::gil_scoped_release release;
                    try {
                        dec->DecodeFromGOPListRGB(std::move(bundle_copies), filepaths, frame_ids_2d, as_bgr,
                                                  enable_gop_dependency_graph_optimization);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(e.what());
                    }
                }
            },
            py::arg("numpy_datas"), py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::arg("enable_gop_dependency_graph_optimization") = false,
            R"pbdoc(
            Submit an async 2D RGB decode from serialized GOP bundles. Returns immediately.

            Args:
                numpy_datas: ``List[List[np.ndarray]]`` shaped ``[V][gop_idx]``.
                    Each element is a 1-D ``uint8`` numpy array containing a
                    serialized GOP bundle (one output element of ``GetGOPList``).
                    All bundles for video ``v`` together must cover every frame in
                    ``frame_ids[v]``.
                filepaths: List of video file paths, ``len == V``.
                frame_ids: 2-D list of frame ids ``[V][F]``.  All inner lists
                    must have the same length.  Order is preserved in the output
                    (output ``[v][f]`` corresponds to ``frame_ids[v][f]``).
                as_bgr: Output BGR (True) or RGB (False).
                enable_gop_dependency_graph_optimization: Use the dependency graph contained
                    in a GOP to reduce the decoding work required to produce requested frames;
                    defaults to False, and GOPs without a graph use normal decoding.

            .. warning::
                **Lifetime contract.** Frames returned by the previous
                ``DecodeFromGOPListRGBGetBuffer()`` are invalidated once this
                method is called again.  Clone before re-submitting.
            )pbdoc")
        .def(
            "DecodeFromGOPListRGBGetBuffer",
            [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr) {
                try {
                    py::gil_scoped_release release;
                    return dec->DecodeFromGOPListRGBGetBuffer(filepaths, frame_ids_2d, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            Block until the pending RGB task completes; return decoded frames.

            Args must exactly match those passed to the preceding
            ``DecodeFromGOPListRGB()`` call.

            Returns:
                ``List[List[RGBFrame]]`` indexed ``[v][f]``, matching the shape
                of ``frame_ids``.  Each ``RGBFrame`` lives in GPU memory and is
                a zero-copy view into the internal aggregator pool — clone before
                calling ``DecodeFromGOPListRGB()`` again.

            Raises:
                RuntimeError: No pending task / empty buffer; result type mismatch
                    (YUV result consumed by RGB getter); request parameter mismatch;
                    or any worker-side decode error.

            .. note::
                This call performs a single ``cuStreamSynchronize`` on the shared
                decode stream, so all decode kernels and D2D copies are GPU-complete
                by the time it returns.
            )pbdoc")
        .def(
            "DecodeFromGOPList",
            [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec,
               const std::vector<std::vector<py::array_t<uint8_t>>>& numpy_data_arrays,
               const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
               bool enable_gop_dependency_graph_optimization) {
                std::vector<std::vector<std::vector<uint8_t>>> bundle_copies;
                bundle_copies.reserve(numpy_data_arrays.size());
                for (const auto& bundles_v : numpy_data_arrays) {
                    bundle_copies.emplace_back();
                    bundle_copies.back().reserve(bundles_v.size());
                    for (const auto& arr : bundles_v) {
                        auto buf = arr.request();
                        const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
                        bundle_copies.back().emplace_back(ptr, ptr + buf.size);
                    }
                }
                {
                    py::gil_scoped_release release;
                    try {
                        dec->DecodeFromGOPList(std::move(bundle_copies), filepaths, frame_ids_2d,
                                               enable_gop_dependency_graph_optimization);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(e.what());
                    }
                }
            },
            py::arg("numpy_datas"), py::arg("filepaths"), py::arg("frame_ids"),
            py::arg("enable_gop_dependency_graph_optimization") = false,
            R"pbdoc(
            Submit an async 2D YUV decode from serialized GOP bundles. Returns immediately.

            Same numpy_datas/filepath/frame_ids semantics as ``DecodeFromGOPListRGB``.
            Output is :class:`DecodedFrameExt` (NV12 / P016 / YUV444).
            ``enable_gop_dependency_graph_optimization`` uses the dependency graph
            contained in a GOP to reduce the decoding work required to produce
            requested frames; it defaults to False, and GOPs without a graph use
            normal decoding.
            )pbdoc")
        .def(
            "DecodeFromGOPListGetBuffer",
            [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<std::vector<int>>& frame_ids_2d) {
                try {
                    py::gil_scoped_release release;
                    return dec->DecodeFromGOPListGetBuffer(filepaths, frame_ids_2d);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"),
            R"pbdoc(
            Block until the pending YUV task completes; return decoded frames.

            Returns:
                ``List[List[DecodedFrameExt]]`` indexed ``[v][f]``.  Each frame
                references the internal aggregator pool — clone before calling
                ``DecodeFromGOPList()`` again.
            )pbdoc")
        .def(
            "release_device_memory",
            [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec) { dec->release_device_memory(); },
            R"pbdoc(
            Release aggregator GPU memory pools (RGB and YUV) and the internal
            GOP decoder memory pool.  Decoder state is preserved.
            )pbdoc")
        .def(
            "release_decoder", [](std::shared_ptr<PyNvBatchAsyncGopDecoder>& dec) { dec->release_decoder(); },
            R"pbdoc(
            Release the internal GOP decoder instance.  It is re-created lazily
            on the next decode call.
            )pbdoc");
}
