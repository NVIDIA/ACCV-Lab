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

#include "PyNvBatchAsyncStreamReader.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nvtx3/nvtx3.hpp"

namespace py = pybind11;

namespace {
// Parallel per-file fanout, mirroring PyNvSampleReader.cpp's local helper.
// Each filepath/frame_id pair is processed in its own thread; first exception
// captured rethrows after join.
template <typename T, typename Func>
std::vector<T> process_frames_in_parallel(const std::vector<std::string>& filepaths,
                                          const std::vector<int>& frame_ids,
                                          const std::vector<PyNvVideoReader*>& video_readers,
                                          Func process_frame) {
    nvtxRangePushA("Process Frames in Parallel (2D worker)");
    std::vector<T> res(filepaths.size());
    std::exception_ptr eptr = nullptr;
    std::mutex mutex;

    std::vector<std::thread> threads;
    threads.reserve(filepaths.size());

    for (size_t i = 0; i < filepaths.size(); ++i) {
        threads.emplace_back([&, i]() {
            try {
                res[i] = process_frame(video_readers[i], frame_ids[i]);
            } catch (const std::exception&) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!eptr) eptr = std::current_exception();
            }
        });
    }
    for (auto& t : threads) t.join();

    if (eptr) {
        nvtxRangePop();
        std::rethrow_exception(eptr);
    }
    nvtxRangePop();
    return res;
}
}  // namespace

PyNvBatchAsyncStreamReader::PyNvBatchAsyncStreamReader(int num_of_set, int num_of_file,
                                                       int max_frames_per_decode_call, int iGpu,
                                                       bool bSuppressNoColorRangeWarning)
    : suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning),
      gpu_id(iGpu),
      num_of_file(num_of_file),
      num_of_set(num_of_set),
      max_frames_per_decode_call(max_frames_per_decode_call),
      decode_result_queue(1),  // Buffer size = 1
      has_pending_task(false) {
    if (num_of_set <= 0) {
        throw std::invalid_argument("num_of_set must be > 0, got " + std::to_string(num_of_set));
    }
    if (num_of_file <= 0) {
        throw std::invalid_argument("num_of_file must be > 0, got " + std::to_string(num_of_file));
    }
    if (max_frames_per_decode_call <= 0) {
        throw std::invalid_argument("max_frames_per_decode_call must be > 0, got " +
                                    std::to_string(max_frames_per_decode_call));
    }

#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvBatchAsyncStreamReader object" << std::endl;
#endif

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [0, " << nGpu - 1 << "]" << std::endl;
    }

    this->destroy_context = false;
    this->cu_context = nullptr;

    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, this->gpu_id));
    ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
    this->destroy_context = true;

    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as named argument 'cudacontext = app_ctx'");
    }

    // Push context temporarily for stream creation; pop immediately so the
    // destructor can run on any thread without context-leak issues.
    ck(cuCtxPushCurrent(this->cu_context));
    ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
    ck(cuCtxPopCurrent(NULL));

    VideoReaderMap.reserve(this->num_of_file);
    for (int i = 0; i < this->num_of_file; i++) {
        VideoReaderMap.emplace_back(this->num_of_set);
    }

    // One aggregator pool per video slot. Each pool starts empty (data_=nullptr,
    // allocated_size_=0) and lazy-sizes itself on the first Decode() call that
    // populates that slot.
    agg_pools.resize(this->num_of_file);
}

PyNvBatchAsyncStreamReader::~PyNvBatchAsyncStreamReader() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvBatchAsyncStreamReader object" << std::endl;
#endif

    bool need_join = false;
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        need_join = has_pending_task;
    }
    if (need_join) {
        decode_worker.join();
    }

    decode_result_queue.clear();

    this->clearAllReaders();

    // Temporarily push context for GPU resource cleanup so the destructor
    // works correctly when called from any thread. Both agg_pool.HardRelease()
    // (cuMemFree on the pool storage) and cuStreamDestroy require an active
    // CUDA context — without this, the cuMemFree fails silently and leaks
    // V * F * frame_bytes of device memory per destroyed reader.
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));

        for (auto& p : agg_pools) {
            p.HardRelease();
        }

        if (this->cu_stream) {
            ck(cuStreamDestroy(this->cu_stream));
        }

        ck(cuCtxPopCurrent(NULL));
    }
    if (this->destroy_context) {
        ck(cuDevicePrimaryCtxRelease(this->gpu_id));
    }
}

void PyNvBatchAsyncStreamReader::waitForPendingAsyncTask() {
    bool need_join = false;
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        need_join = has_pending_task;
    }
    if (need_join) {
        decode_worker.join();
    }
}

void PyNvBatchAsyncStreamReader::clearDecodeResultBuffer() {
    while (!decode_result_queue.empty()) {
        decode_result_queue.pop_front();
    }
}

void PyNvBatchAsyncStreamReader::clearAllReaders() {
    waitForPendingAsyncTask();
    for (auto& reader_map : VideoReaderMap) {
        reader_map.clearAllReaders();
    }
}

void PyNvBatchAsyncStreamReader::ReleaseMemPools() {
    waitForPendingAsyncTask();
    for (auto& reader_map : VideoReaderMap) {
        reader_map.releaseAllMemPools();
    }
    // cuMemFree needs context active; per-reader releaseAllMemPools handles
    // its own push/pop, but agg_pools are owned by this class and must be
    // freed under our own context push.
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));
        for (auto& p : agg_pools) {
            p.HardRelease();
        }
        ck(cuCtxPopCurrent(NULL));
    }
}

void PyNvBatchAsyncStreamReader::ReleaseDecoder() {
    waitForPendingAsyncTask();
    clearAllReaders();
}

std::string PyNvBatchAsyncStreamReader::generate_request_key(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
    bool as_bgr) {
    std::ostringstream oss;
    oss << as_bgr << ":";
    for (size_t v = 0; v < filepaths.size(); ++v) {
        oss << filepaths[v] << "[";
        if (v < frame_ids_2d.size()) {
            for (size_t f = 0; f < frame_ids_2d[v].size(); ++f) {
                oss << frame_ids_2d[v][f];
                if (f + 1 < frame_ids_2d[v].size()) oss << ",";
            }
        }
        oss << "]";
        if (v + 1 < filepaths.size()) oss << ";";
    }
    return oss.str();
}

bool PyNvBatchAsyncStreamReader::validate_request(const DecodeResult2D& result,
                                                  const std::vector<std::string>& filepaths,
                                                  const std::vector<std::vector<int>>& frame_ids_2d,
                                                  bool as_bgr) {
    if (result.as_bgr != as_bgr) return false;
    if (result.file_path_list.size() != filepaths.size()) return false;
    if (result.frame_id_list_2d.size() != frame_ids_2d.size()) return false;
    for (size_t v = 0; v < filepaths.size(); ++v) {
        if (result.file_path_list[v] != filepaths[v]) return false;
        if (result.frame_id_list_2d[v].size() != frame_ids_2d[v].size()) return false;
        for (size_t f = 0; f < frame_ids_2d[v].size(); ++f) {
            if (result.frame_id_list_2d[v][f] != frame_ids_2d[v][f]) return false;
        }
    }
    return true;
}

void PyNvBatchAsyncStreamReader::validate_decode_input(const std::vector<std::string>& filepaths,
                                                       const std::vector<std::vector<int>>& frame_ids_2d) {
    if (filepaths.size() != frame_ids_2d.size()) {
        throw std::invalid_argument("filepaths.size() (" + std::to_string(filepaths.size()) +
                                    ") must equal frame_ids_2d.size() (" +
                                    std::to_string(frame_ids_2d.size()) + ")");
    }
    if (filepaths.empty()) {
        throw std::invalid_argument("filepaths must not be empty");
    }
    if (filepaths.size() > static_cast<size_t>(this->num_of_file)) {
        throw std::invalid_argument("Number of files (" + std::to_string(filepaths.size()) +
                                    ") exceeds num_of_file (" + std::to_string(this->num_of_file) +
                                    ") specified at construction.");
    }

    const size_t expected_F = frame_ids_2d[0].size();
    if (expected_F == 0) {
        throw std::invalid_argument("frame_ids_2d[0] must not be empty");
    }
    if (expected_F > static_cast<size_t>(this->max_frames_per_decode_call)) {
        throw std::invalid_argument(
            "frames per video (" + std::to_string(expected_F) + ") exceeds max_frames_per_decode_call (" +
            std::to_string(this->max_frames_per_decode_call) + ") specified at construction.");
    }
    for (size_t v = 0; v < frame_ids_2d.size(); ++v) {
        if (frame_ids_2d[v].size() != expected_F) {
            throw std::invalid_argument("frame_ids_2d[" + std::to_string(v) + "].size() (" +
                                        std::to_string(frame_ids_2d[v].size()) +
                                        ") must equal frame_ids_2d[0].size() (" + std::to_string(expected_F) +
                                        "); jagged inner lengths are not supported");
        }
    }
}

std::vector<RGBFrame> PyNvBatchAsyncStreamReader::run_rgb_out_1d(const std::vector<std::string>& filepaths,
                                                                 const std::vector<int>& frame_ids,
                                                                 bool as_bgr) {
    // Caller (the worker) has already validated outer/inner sizes via
    // validate_decode_input. Here we only resolve readers and dispatch in parallel.
    std::vector<PyNvVideoReader*> video_readers(filepaths.size());

    nvtxRangePushA("Get Video Readers (2D worker)");
    for (size_t i = 0; i < filepaths.size(); ++i) {
        FixedSizeVideoReaderMap& reader_map = this->VideoReaderMap[i];
        PyNvVideoReader* video_reader = nullptr;
        // Only allocate a new reader when there's room AND the file isn't already
        // cached, matching PyNvSampleReader::run_rgb_out's memory-leak guard.
        if (reader_map.notFull() && !reader_map.contains(filepaths[i])) {
            video_reader = new PyNvVideoReader(filepaths[i], this->gpu_id, this->cu_context, this->cu_stream);
        }
        auto cur_video_reader = reader_map.find(filepaths[i], video_reader);
        video_readers[i] = cur_video_reader;
    }
    nvtxRangePop();

    return process_frames_in_parallel<RGBFrame>(filepaths, frame_ids, video_readers,
                                                [as_bgr](PyNvVideoReader* reader, int frame_id) {
                                                    return reader->run_single_rgb_out(frame_id, as_bgr);
                                                });
}

void PyNvBatchAsyncStreamReader::Decode(const std::vector<std::string>& filepaths,
                                        const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr) {
    validate_decode_input(filepaths, frame_ids_2d);

    std::unique_lock<std::mutex> lock(async_mutex);

    // Drain any prior in-flight task before starting a new one. Matches the
    // 1D async behavior (warn + join + discard) but additionally clears stale
    // results unconditionally, so the queue can never carry over a previous
    // submission's frames into a fresh Decode() call.
    if (has_pending_task) {
        std::cerr << "[WARNING] PyNvBatchAsyncStreamReader::Decode: A previous async decode task is "
                     "still running. Waiting for it to complete before starting the new task."
                  << std::endl;
        lock.unlock();
        decode_worker.join();
        lock.lock();
        has_pending_task = false;
    }
    while (!decode_result_queue.empty()) {
        decode_result_queue.pop_front();
    }

    // Snapshot inputs for the worker closure.
    auto filepaths_cap = filepaths;
    auto frame_ids_cap = frame_ids_2d;
    bool as_bgr_cap = as_bgr;

    has_pending_task = true;
    decode_worker.start([this, filepaths_cap, frame_ids_cap, as_bgr_cap]() {
        DecodeResult2D result;
        result.file_path_list = filepaths_cap;
        result.frame_id_list_2d = frame_ids_cap;
        result.as_bgr = as_bgr_cap;
        result.is_ready = false;

        // Worker runs on a fresh std::thread that has no current CUDA context.
        // Push our context for the duration of the worker so all cuMemcpyDtoDAsync /
        // cuStreamSynchronize calls below have something on the stack. The inner
        // PyNvVideoReader path also pushes/pops its own copy of cu_context, which
        // nests cleanly with this outer push.
        bool ctx_pushed = false;
        try {
            CUDA_DRVAPI_CALL(cuCtxPushCurrent(this->cu_context));
            ctx_pushed = true;
        } catch (...) {
            // If we can't even push the context, surface the error to GetBuffer.
            result.exception = std::current_exception();
            result.is_ready = true;
            decode_result_queue.push_back(result);
            std::lock_guard<std::mutex> lk(async_mutex);
            has_pending_task = false;
            return;
        }

        try {
            nvtxRangePushA("Batch 2D Decode Worker");
            const int V = static_cast<int>(filepaths_cap.size());
            const int F = static_cast<int>(frame_ids_cap[0].size());

            result.decoded_frames.assign(V, std::vector<RGBFrame>{});
            for (int v = 0; v < V; ++v) result.decoded_frames[v].reserve(F);

            // Per-video shape snapshot (recorded at f==0, asserted at f>=1).
            // Videos in a single Decode() call may differ in resolution; each
            // video's own F frames must be uniform (always true for one mp4).
            std::vector<std::tuple<size_t, size_t, size_t>> ref_shape(V);
            std::vector<std::tuple<size_t, size_t, size_t>> ref_stride(V);
            std::vector<std::string> ref_typestr(V);
            std::vector<size_t> v_bytes(V, 0);

            for (int f = 0; f < F; ++f) {
                std::vector<int> fids_at_f(V);
                for (int v = 0; v < V; ++v) fids_at_f[v] = frame_ids_cap[v][f];

                auto frames = this->run_rgb_out_1d(filepaths_cap, fids_at_f, as_bgr_cap);

                for (int v = 0; v < V; ++v) {
                    if (f == 0) {
                        // First frame for this video in this call — snapshot shape
                        // and (re)size that video's aggregator pool. EnsureSize
                        // realloc's only when the existing capacity is smaller
                        // than what we need, so resolution growth across calls
                        // triggers a re-alloc automatically; same-or-smaller
                        // resolutions reuse the existing allocation.
                        ref_shape[v] = frames[v].shape;
                        ref_stride[v] = frames[v].stride;
                        ref_typestr[v] = frames[v].typestr;
                        const size_t H = std::get<0>(ref_shape[v]);
                        const size_t W = std::get<1>(ref_shape[v]);
                        v_bytes[v] = H * W * 3;

                        agg_pools[v].EnsureSizeAndSoftReset(static_cast<size_t>(F) * v_bytes[v], false);
                    } else {
                        // Subsequent frames from the same video must keep the
                        // same shape. Files don't change resolution mid-stream
                        // in practice; if they did, AddElement would over-run
                        // the pool below. Defensive check.
                        if (frames[v].shape != ref_shape[v]) {
                            std::ostringstream oss;
                            oss << "PyNvBatchAsyncStreamReader: video " << v
                                << " changed resolution mid-call: f=0 was " << std::get<0>(ref_shape[v])
                                << "x" << std::get<1>(ref_shape[v]) << ", f=" << f << " is "
                                << std::get<0>(frames[v].shape) << "x" << std::get<1>(frames[v].shape) << ".";
                            throw std::runtime_error(oss.str());
                        }
                    }

                    void* dst = agg_pools[v].AddElement(v_bytes[v]);
                    CUDA_DRVAPI_CALL(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst), frames[v].data,
                                                       v_bytes[v], cu_stream));

                    const std::vector<size_t> shape_vec = {std::get<0>(ref_shape[v]),
                                                           std::get<1>(ref_shape[v]), 3};
                    const std::vector<size_t> stride_vec = {
                        std::get<0>(ref_stride[v]), std::get<1>(ref_stride[v]), std::get<2>(ref_stride[v])};
                    result.decoded_frames[v].emplace_back(shape_vec, stride_vec, ref_typestr[v],
                                                          reinterpret_cast<size_t>(cu_stream),
                                                          reinterpret_cast<CUdeviceptr>(dst),
                                                          /*readOnly=*/false, /*isBGR=*/as_bgr_cap);
                }
            }

            // Single terminal sync: all decode kernels + V*F D2D copies on cu_stream
            // are FIFO; one sync drains the whole pipeline so the result becomes
            // GPU-visible to any consumer stream by the time GetBuffer returns.
            CUDA_DRVAPI_CALL(cuStreamSynchronize(cu_stream));

            result.is_ready = true;
            decode_result_queue.push_back(result);

            nvtxRangePop();
        } catch (...) {
            // On failure, soft-reset every pool so the next Decode reuses the
            // same allocations. The buffered RGBFrame views become invalid —
            // clear them and stash the exception for GetBuffer to rethrow.
            for (auto& p : agg_pools) {
                p.SoftRelease();
            }
            result.decoded_frames.clear();
            result.exception = std::current_exception();
            result.is_ready = true;
            decode_result_queue.push_back(result);
        }

        if (ctx_pushed) {
            // Best-effort pop. If this throws, the worker is exiting anyway —
            // swallow so we always release async_mutex below.
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
        }

        {
            std::lock_guard<std::mutex> lk(async_mutex);
            has_pending_task = false;
        }
    });
}

std::vector<std::vector<RGBFrame>> PyNvBatchAsyncStreamReader::GetBuffer(
    const std::vector<std::string>& filepaths, const std::vector<std::vector<int>>& frame_ids_2d,
    bool as_bgr) {
    {
        std::lock_guard<std::mutex> lock(async_mutex);
        if (!has_pending_task && decode_result_queue.empty()) {
            throw std::runtime_error(
                "PyNvBatchAsyncStreamReader::GetBuffer: No pending decode task and buffer is empty. "
                "Call Decode first before calling GetBuffer.");
        }
    }

    // Blocks until worker pushes (worker may still be running).
    DecodeResult2D result = decode_result_queue.pop_front();

    if (!result.is_ready) {
        throw std::runtime_error(
            "PyNvBatchAsyncStreamReader::GetBuffer: Internal error — result not ready when popped.");
    }
    if (result.exception) {
        std::rethrow_exception(result.exception);
    }
    if (!validate_request(result, filepaths, frame_ids_2d, as_bgr)) {
        std::ostringstream oss;
        oss << "PyNvBatchAsyncStreamReader::GetBuffer: Request parameters do not match buffered "
               "result. Expected: "
            << generate_request_key(filepaths, frame_ids_2d, as_bgr) << ", Got: "
            << generate_request_key(result.file_path_list, result.frame_id_list_2d, result.as_bgr);
        throw std::runtime_error(oss.str());
    }

    return result.decoded_frames;
}

void Init_PyNvBatchAsyncStreamReader(py::module& m) {
    m.def(
        "CreateBatchAsyncStreamReader",
        [](int num_of_set, int num_of_file, int max_frames_per_decode_call, int iGpu,
           bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvBatchAsyncStreamReader>(
                num_of_set, num_of_file, max_frames_per_decode_call, iGpu, suppressNoColorRangeWarning);
        },
        py::arg("num_of_set"), py::arg("num_of_file"), py::arg("max_frames_per_decode_call"),
        py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
            Create a PyNvBatchAsyncStreamReader for 2D async stream decoding.

            This reader is **async-only** and **2D-only**: it accepts a list of
            video files and a 2D list of frame ids (one list per video), submits
            the decode in the background, and returns the decoded frames as
            ``List[List[RGBFrame]]`` indexed ``[v][f]``.

            Args:
                num_of_set: Number of decoder slots per file.
                num_of_file: Maximum number of videos per decode call (V upper bound).
                max_frames_per_decode_call: Maximum number of frames per video per decode
                    call (F upper bound). The internal aggregator pool is sized for
                    this peak.
                iGpu: GPU device id.
                suppressNoColorRangeWarning: Suppress warning when no color range
                    can be extracted (limited / MPEG range is assumed).

            Returns:
                :class:`PyNvBatchAsyncStreamReader` instance configured with the specified parameters

            Example:
                >>> reader = CreateBatchAsyncStreamReader(
                ...     num_of_set=1, num_of_file=6, max_frames_per_decode_call=4)
                >>> reader.Decode(filepaths, frame_ids_2d, as_bgr=False)
                >>> out = reader.GetBuffer(filepaths, frame_ids_2d, as_bgr=False)
                >>> # out[v][f] is an RGBFrame; clone before next Decode() call
        )pbdoc");

    py::class_<PyNvBatchAsyncStreamReader, std::shared_ptr<PyNvBatchAsyncStreamReader>>(
        m, "PyNvBatchAsyncStreamReader", py::module_local(),
        R"pbdoc(
        NVIDIA GPU-accelerated 2D async stream video decoder.

        This class submits a 2D decode request (V videos × F frames per video)
        to a background C++ worker thread and returns the decoded frames as
        ``List[List[RGBFrame]]`` indexed ``[v][f]``. It is async-only (no sync
        ``Decode`` method) and 2D-only. The 1D
        :class:`~accvlab.on_demand_video_decoder.PyNvSampleReader` class
        serves the 1-frame-per-video case.

        Async model
        ~~~~~~~~~~~

        At most one in-flight task at a time; the internal result buffer holds
        a single result. ``Decode()`` returns immediately; ``GetBuffer()``
        blocks until the worker pushes its result.

        Calling ``Decode()`` while a previous task is still pending will:
            1. Print a warning to stderr.
            2. Join the previous worker.
            3. Discard the previous result (whether already pushed or not).
            4. Start the new task.

        Calling ``Decode()`` after a previous task has completed but its result
        has not been retrieved will also discard the previous result and start
        a new task. Always pair every ``Decode()`` with a matching
        ``GetBuffer()`` for the results you want to keep.

        Contracts
        ~~~~~~~~~

        It is important to follow the following two contracts to ensure
        the correct function. Read both before doing anything with the
        returned frames.

        **Contract 1 — GetBuffer() returns when GPU work is complete.**
        The worker performs ``cuStreamSynchronize`` on its internal stream
        before pushing the result. By the time ``GetBuffer()`` returns to
        Python, all decoder kernels and device-to-device copies for the
        returned frames have finished. Downstream torch / CUDA ops can read
        the frame data on any stream without further user-level
        synchronization.

        **Contract 2 — RGBFrames are invalidated on the next Decode() call.**
        The returned ``RGBFrame`` objects are zero-copy views into an internal
        aggregator pool. Submitting the next ``Decode()`` reuses the same
        pool memory for the new batch's frames. You MUST consume or clone
        every frame you want to keep BEFORE the next ``Decode()`` call.
        Typical idiom::

            reader.Decode(files, frame_ids_a, as_bgr=False)
            out = reader.GetBuffer(files, frame_ids_a, as_bgr=False)
            tensors = [[torch.as_tensor(out[v][f], device="cuda").clone()
                        for f in range(F)] for v in range(V)]
            # Safe to call Decode() again — tensors own their own memory.
            reader.Decode(files, frame_ids_b, as_bgr=False)

        Skipping the clone leads to silent data corruption: PyTorch will not
        know its tensor's backing memory got overwritten by the next decode.

        Memory sizing
        ~~~~~~~~~~~~~

        Each video slot has its own aggregator pool, sized lazily on the
        first ``Decode()`` to that slot to ``F * H_v * W_v * 3`` bytes.
        Videos in a single ``Decode()`` call may have different resolutions
        (e.g. mixed-resolution camera rigs). If a later ``Decode()`` swaps in
        a higher-resolution video at the same slot, that slot's pool is
        reallocated automatically. Within one Decode() call, the F frames
        of any given video must share the same shape — this is normally
        true since the F frames come from a single mp4 file.

        .. seealso::

            - ``samples/SampleBatchAsyncStreamAccess.py`` for the canonical
              prefetch loop.
            - :class:`~accvlab.on_demand_video_decoder.PyNvSampleReader` for
              the 1-frame-per-video API.
        )pbdoc")
        .def(py::init<int, int, int, int, bool>(), py::arg("num_of_set"), py::arg("num_of_file"),
             py::arg("max_frames_per_decode_call"), py::arg("iGpu") = 0,
             py::arg("suppressNoColorRangeWarning") = false)
        .def(
            "Decode",
            [](std::shared_ptr<PyNvBatchAsyncStreamReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr) {
                try {
                    reader->Decode(filepaths, frame_ids_2d, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Submit an async 2D decode task. Returns immediately.

            Args:
                filepaths: List of video file paths. ``len(filepaths) <= num_of_file``.
                frame_ids: 2D list of frame ids. ``len(frame_ids) == len(filepaths)``;
                    each inner list must be the same length (no jagged inner dims)
                    and ``<= max_frames_per_decode_call``. ``frame_ids[v][f]`` is the f-th
                    frame requested for video v.
                as_bgr: Output BGR (True) or RGB (False).

            Raises:
                RuntimeError: invalid input dimensions, exceeded construction limits,
                    jagged inner lengths, or non-positive sizes.

            .. note::
                **Discards prior result.** Calling ``Decode()`` unconditionally
                invalidates any prior buffered result. If a previous task is
                still running, it is joined first (with a warning to stderr)
                and its result discarded. Always pair every ``Decode()`` with
                a matching ``GetBuffer()`` for results you want to keep.

            .. warning::
                **Lifetime contract.** Frames previously returned by
                ``GetBuffer()`` become invalid as soon as you call ``Decode()``
                again. Clone everything you need to keep BEFORE this call.
                See class docstring for details.
            )pbdoc")
        .def(
            "GetBuffer",
            [](std::shared_ptr<PyNvBatchAsyncStreamReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<std::vector<int>>& frame_ids_2d, bool as_bgr) {
                try {
                    return reader->GetBuffer(filepaths, frame_ids_2d, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Block until the pending async task completes; return decoded frames.

            Args:
                filepaths: List of video file paths. Must exactly match the
                    request passed to the previous ``Decode()`` call.
                frame_ids: 2D list of frame ids. Must exactly match the request
                    passed to the previous ``Decode()`` call.
                as_bgr: Output format flag. Must exactly match the request
                    passed to the previous ``Decode()`` call.

            Returns:
                A nested list ``List[List[RGBFrame]]`` indexed ``[v][f]``,
                mirroring the shape of the input ``frame_ids``. Each
                ``RGBFrame.shape == (H, W, 3)``, ``dtype == uint8``, lives in
                GPU memory, and is a zero-copy view into the reader's internal
                aggregator pool.

            Raises:
                RuntimeError: No pending task and empty buffer; or request
                    parameters do not match the buffered result (the result
                    is then consumed and unrecoverable — same semantics as
                    the 1D async API). Worker-side exceptions (file not
                    found, invalid frame id, resolution mismatch across V)
                    are propagated unchanged.

            .. note::
                **Contract 1 — GPU-ready on return.** The worker performs
                ``cuStreamSynchronize`` before pushing the result, so by the
                time this call returns, all decoder kernels and D2D copies are
                complete on the GPU. Downstream torch / CUDA ops can read the
                frames on any stream without further user-level synchronization.

            .. warning::
                **Contract 2 — Invalidated on next Decode().** The returned
                ``RGBFrame`` objects share memory with the reader's internal
                pool. Submitting the next ``Decode()`` reuses that memory for
                the new batch. You MUST clone (e.g.
                ``torch.as_tensor(frame, device="cuda").clone()``) every frame
                you want to keep BEFORE calling ``Decode()`` again. Skipping
                the clone leads to silent data corruption — PyTorch tensors
                will not know their backing memory was overwritten.
            )pbdoc")
        .def(
            "clearAllReaders",
            [](std::shared_ptr<PyNvBatchAsyncStreamReader>& reader) { reader->clearAllReaders(); },
            R"pbdoc(
            Clear all underlying video readers. Waits for pending async task first.
            )pbdoc")
        .def(
            "release_device_memory",
            [](std::shared_ptr<PyNvBatchAsyncStreamReader>& reader) { reader->ReleaseMemPools(); },
            R"pbdoc(
            Release per-reader memory pools and the 2D aggregator pool.
            Decoder state is preserved for efficient forward decoding.
            )pbdoc")
        .def(
            "release_decoder",
            [](std::shared_ptr<PyNvBatchAsyncStreamReader>& reader) { reader->ReleaseDecoder(); },
            R"pbdoc(
            Release all decoder instances. Readers are re-created lazily on next decode.
            )pbdoc");
}
