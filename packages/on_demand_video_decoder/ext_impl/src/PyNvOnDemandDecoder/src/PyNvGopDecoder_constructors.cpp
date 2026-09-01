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

namespace fs = std::filesystem;

namespace {

size_t WorkerCountForCpuBudget(int max_num_files, size_t budget_divisor) {
    const size_t file_limit = max_num_files > 0 ? static_cast<size_t>(max_num_files) : 1;
    const size_t cpu_budget = std::max<size_t>(1, ThreadPool::available_cpu_count() / budget_divisor);
    return std::min(file_limit, cpu_budget);
}

}  // namespace

std::vector<FastStreamInfo> GetFastInitInfo(const std::vector<std::string>& filepaths) {
    std::vector<FastStreamInfo> fast_stream_infos;
    fast_stream_infos.reserve(filepaths.size());

    for (const auto& filepath : filepaths) {
        AVFormatContext* fmtc = nullptr;
        int iVideoStream = -1;

        try {
            // Initialize FFmpeg
            av_log_set_level(AV_LOG_QUIET);
            avformat_network_init();

            // Open input file
            if (avformat_open_input(&fmtc, filepath.c_str(), NULL, NULL) < 0) {
                throw std::runtime_error("Failed to open input file: " + filepath);
            }

            // Find stream info
            if (avformat_find_stream_info(fmtc, NULL) < 0) {
                throw std::runtime_error("Failed to find stream info for file: " + filepath);
            }

            // Find best video stream
            iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
            if (iVideoStream < 0) {
                throw std::runtime_error("Could not find video stream in file: " + filepath);
            }

            // Extract stream information directly from AVFormatContext
            FastStreamInfo info;
            info.codec_type = static_cast<int>(fmtc->streams[iVideoStream]->codecpar->codec_type);
            info.codec_id = static_cast<int>(fmtc->streams[iVideoStream]->codecpar->codec_id);
            info.width = fmtc->streams[iVideoStream]->codecpar->width;
            info.height = fmtc->streams[iVideoStream]->codecpar->height;
            info.format = fmtc->streams[iVideoStream]->codecpar->format;

            // Time base information
            info.time_base_num = fmtc->streams[iVideoStream]->time_base.num;
            info.time_base_den = fmtc->streams[iVideoStream]->time_base.den;

            // Frame rate information
            info.avg_frame_rate_num = fmtc->streams[iVideoStream]->avg_frame_rate.num;
            info.avg_frame_rate_den = fmtc->streams[iVideoStream]->avg_frame_rate.den;
            info.r_frame_rate_num = fmtc->streams[iVideoStream]->r_frame_rate.num;
            info.r_frame_rate_den = fmtc->streams[iVideoStream]->r_frame_rate.den;

            // Start time and duration
            info.start_time = fmtc->streams[iVideoStream]->start_time;
            info.duration = fmtc->streams[iVideoStream]->duration;

            fast_stream_infos.push_back(info);

            // Clean up
            avformat_close_input(&fmtc);

        } catch (const std::exception& e) {
            // Clean up on error
            if (fmtc) {
                avformat_close_input(&fmtc);
            }
            throw std::runtime_error("Failed to extract FastStreamInfo from file: " + filepath +
                                     ". Error: " + e.what());
        }
    }

    return fast_stream_infos;
}

void PyNvGopDecoder::ensureCudaContextInitialized() {
    if (this->cu_context != nullptr) {
        return;  // Already initialized
    }

    ck(cuInit(0));

    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (this->gpu_id < 0 || this->gpu_id >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]"
                  << std::endl;
    }
    this->destroy_context = false;

    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, this->gpu_id));
    ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
    this->destroy_context = true;

    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");
    }

    if (!this->cu_stream) {
        // No external stream was provided — create one owned by this object.
        ck(cuCtxPushCurrent(this->cu_context));
        ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
        ck(cuCtxPopCurrent(NULL));
        // owns_stream remains true (default)
    }
}

PyNvGopDecoder::PyNvGopDecoder(int iMaxFileNum, int iGpu, bool bSuppressNoColorRangeWarning,
                               CUstream external_stream)
    : max_num_files(iMaxFileNum),
      gpu_id(iGpu),
      suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning),
      demux_pool(WorkerCountForCpuBudget(iMaxFileNum, 2)),
      decode_pool(WorkerCountForCpuBudget(iMaxFileNum, 2)),
      parallel_pool(WorkerCountForCpuBudget(iMaxFileNum, 1)) {
#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvGopDecoder object" << std::endl;
#endif

    this->last_decoded_frame_infos.resize(this->max_num_files);
    reset_last_decoded_frame_infos(this->last_decoded_frame_infos);

    if (external_stream != nullptr) {
        this->cu_stream = external_stream;
        this->owns_stream = false;
    }
}

void PyNvGopDecoder::force_join_all() {
    demux_pool.force_join();
    decode_pool.force_join();
    parallel_pool.force_join();
}

PyNvGopDecoder::~PyNvGopDecoder() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvGopDecoder object" << std::endl;
#endif

    force_join_all();

    // Temporarily push context for GPU resource cleanup.
    // This ensures the destructor works correctly on any thread.
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));

        // Clean up NvDecoder instances (they need context for GPU memory release)
        for (int i = 0; i < this->max_num_files; ++i) {
            if (i < this->vdec.size()) {
                this->vdec[i].reset();
            }
        }

        // Explicitly release GPU memory pool before automatic member destruction
        gpu_mem_pool.HardRelease();

        if (this->cu_stream && this->owns_stream) {
            ck(cuStreamDestroy(this->cu_stream));
        }

        ck(cuCtxPopCurrent(NULL));
    }

    if (this->destroy_context) {
        // Only release the primary context reference.
        // No need to pop - we use temporary push/pop pattern instead.
        ck(cuDevicePrimaryCtxRelease(this->gpu_id));
    }
}

void Init_PyNvGopDecoder(py::module& m) {
    ExternalBuffer::Export(m);
    CAIMemoryView::Export(m);
    DecodedFrameExt::Export(m);
    RGBFrame::Export(m);
    py::class_<FastStreamInfo>(m, "FastStreamInfo",
                               R"pbdoc(
        Pre-extracted stream metadata used to accelerate the demuxing stage.

        Passing a :class:`FastStreamInfo` to :meth:`PyNvGopDecoder.Decode`,
        :meth:`PyNvGopDecoder.DecodeN12ToRGB`, or :meth:`PyNvGopDecoder.GetGOPList` allows
        the demuxer to skip the stream-probing step, reducing per-call latency. Obtain
        instances via :func:`GetFastInitInfo`.

        Note:
            A :class:`FastStreamInfo` can be reused across multiple video files as long as
            they share the same encoding parameters (codec, resolution, frame rate, etc.).
            This is common in autonomous driving or robotics datasets where all clips are
            recorded from the same camera configuration. Reusing it across files with
            different parameters will cause undefined behavior during demuxing.
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("codec_type", &FastStreamInfo::codec_type,
                       R"pbdoc(FFmpeg codec type (AVMediaType enum value))pbdoc")
        .def_readwrite("codec_id", &FastStreamInfo::codec_id,
                       R"pbdoc(FFmpeg codec ID (AVCodecID enum value, e.g., AV_CODEC_ID_H264=27))pbdoc")
        .def_readwrite("width", &FastStreamInfo::width, R"pbdoc(Video frame width in pixels)pbdoc")
        .def_readwrite("height", &FastStreamInfo::height, R"pbdoc(Video frame height in pixels)pbdoc")
        .def_readwrite("format", &FastStreamInfo::format,
                       R"pbdoc(Pixel format (AVPixelFormat enum value))pbdoc")
        .def_readwrite("time_base_num", &FastStreamInfo::time_base_num,
                       R"pbdoc(Time base numerator for timestamp calculations)pbdoc")
        .def_readwrite("time_base_den", &FastStreamInfo::time_base_den,
                       R"pbdoc(Time base denominator for timestamp calculations)pbdoc")
        .def_readwrite("avg_frame_rate_num", &FastStreamInfo::avg_frame_rate_num,
                       R"pbdoc(Average frame rate numerator)pbdoc")
        .def_readwrite("avg_frame_rate_den", &FastStreamInfo::avg_frame_rate_den,
                       R"pbdoc(Average frame rate denominator)pbdoc")
        .def_readwrite("r_frame_rate_num", &FastStreamInfo::r_frame_rate_num,
                       R"pbdoc(Real frame rate numerator)pbdoc")
        .def_readwrite("r_frame_rate_den", &FastStreamInfo::r_frame_rate_den,
                       R"pbdoc(Real frame rate denominator)pbdoc")
        .def_readwrite("start_time", &FastStreamInfo::start_time,
                       R"pbdoc(Start time of the stream in time base units)pbdoc")
        .def_readwrite("duration", &FastStreamInfo::duration,
                       R"pbdoc(Duration of the stream in time base units)pbdoc");

    m.def(
        "CreateGopDecoder",
        [](int maxfiles, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvGopDecoder>(maxfiles, iGpu, suppressNoColorRangeWarning);
        },
        py::arg("maxfiles"), py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
        Create the native GPU decoder.

        See :func:`~accvlab.on_demand_video_decoder.CreateGopDecoder` for the public API
        and full documentation.
        )pbdoc");

    m.def(
        "GetFastInitInfo",
        [](const std::vector<std::string>& filepaths) {
            try {
                return GetFastInitInfo(filepaths);
            } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
            }
        },
        py::arg("filepaths"), py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
        Extracts :class:`FastStreamInfo` from a list of video files.

        Args:
            filepaths: List of video file paths to analyze

        Returns:
            List of :class:`FastStreamInfo` objects, one per file

        Raises:
            RuntimeError: If files cannot be opened or stream information cannot be extracted

        Example:
            >>> stream_infos = GetFastInitInfo(['video1.mp4', 'video2.mp4'])
            >>> gop_list = decoder.GetGOPList(['video1.mp4', 'video2.mp4'], [0, 10], stream_infos)

        See Also:
            :class:`FastStreamInfo`: Usage and reuse conditions.
        )pbdoc");

    m.def(
        "SaveGopToFile",
        [](const py::array_t<uint8_t>& numpy_data, const std::string& dst_filepath) {
            try {
                // Extract data pointer and size while holding GIL (accessing Python object)
                const uint8_t* data_ptr = static_cast<const uint8_t*>(numpy_data.data());
                size_t data_size = numpy_data.size();

                // Release GIL for file I/O operation
                {
                    py::gil_scoped_release release;
                    SaveBinaryDataToFile(data_ptr, data_size, dst_filepath);
                }
            } catch (const std::exception& e) {
                throw std::runtime_error(e.what());
            }
        },
        py::arg("numpy_data"), py::arg("dst_filepath"),
        R"pbdoc(
        Saves one serialized GOP bundle (for a single video) to a binary file.

        Serialized GOP bundles are obtained from :meth:`PyNvGopDecoder.GetGOPList`, which returns
        one bundle (numpy object) per video. Call this function once per video to save each bundle.
        To reload the bundles later, use :meth:`PyNvGopDecoder.LoadGopsToList`.

        Args:
            numpy_data: Numpy object containing one serialized GOP bundle for a single
                        video. This corresponds to a single element of the list returned
                        by :meth:`PyNvGopDecoder.GetGOPList`
            dst_filepath: Destination file path where the bundle will be written

        Raises:
            RuntimeError: If file cannot be written or data is invalid
            ValueError: If dst_filepath is empty

        Example:
            >>> gop_list = decoder.GetGOPList(['v0.mp4', 'v1.mp4'], [10, 20])
            >>> for i, (packets, _, _) in enumerate(gop_list):
            ...     SaveGopToFile(packets, f'gop_{i}.bin')

        See Also:
            For advanced usage including hierarchical GOP storage with persistent index,
            see ``examples/demuxer_free_decode/gop_storage.py``.
        )pbdoc");

    py::class_<PyNvGopDecoder, shared_ptr<PyNvGopDecoder>>(m, "PyNvGopDecoder", py::module_local(),
                                                           R"pbdoc(
        GPU-accelerated video decoder with GOP-level random access.

        Do not instantiate this class directly. Use :func:`CreateGopDecoder` to obtain
        an instance.

        See Also:
            :func:`CreateGopDecoder`: Factory function with full parameter documentation.
        )pbdoc")
        .def(
            "Decode",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, std::vector<FastStreamInfo> fastStreamInfos) {
                try {
                    std::vector<DecodedFrameExt> result;
                    dec->decode_from_video(filepaths, frame_ids, false, false, &result, nullptr,
                                           fastStreamInfos.empty() ? nullptr : fastStreamInfos.data());
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"),
            py::arg("fastStreamInfos") = std::vector<FastStreamInfo>{},
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Decodes video file stream into YUV data.

            This method performs GPU-accelerated decoding of video frames using NVIDIA hardware.
            It supports multiple video files and can decode specific frame IDs from each file.
            The method uses GOP-based decoding for efficient random access.

            If you need RGB/BGR output, use :meth:`DecodeN12ToRGB` instead.
            
            Args:
                filepaths: List of video file paths to decode from.
                frame_ids: List of frame IDs to decode. Each frame ID corresponds to
                           a specific frame in the video sequence.
                fastStreamInfos: Optional list of FastStreamInfo objects containing
                                pre-extracted stream information by :func:`GetFastInitInfo`.
                                If provided, this can improve performance by avoiding
                                stream analysis.

            Returns:
                List of :class:`DecodedFrameExt` objects containing the decoded frame data.

            Raises:
                RuntimeError: If video files cannot be opened or decoded
                ValueError: If frame_ids contain invalid indices

            Example:
                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> # Convert to PyTorch tensors on GPU (NV12 layout: (height * 3 // 2, width), uint8)
                >>> nv12_tensors = [torch.as_tensor(frame).clone() for frame in frames]
            )pbdoc")
        .def(
            "DecodeN12ToRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, bool as_bgr, std::vector<FastStreamInfo> fastStreamInfos) {
                try {
                    std::vector<RGBFrame> result;
                    dec->decode_from_video(filepaths, frame_ids, true, as_bgr, nullptr, &result,
                                           fastStreamInfos.empty() ? nullptr : fastStreamInfos.data());
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            py::arg("fastStreamInfos") = std::vector<FastStreamInfo>{},
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Decodes video file stream into RGB/BGR data.
            
            This method performs GPU-accelerated decoding and color space conversion
            from YUV to RGB/BGR format.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
                as_bgr: Whether to output in BGR format (True) or RGB format (False). BGR is commonly used in OpenCV applications.
                fastStreamInfos: Optional list of FastStreamInfo objects containing pre-extracted stream information by :func:`GetFastInitInfo`. If provided, this can improve performance by avoiding stream analysis.
            
            Returns:
                List of :class:`RGBFrame` objects containing the decoded and color-converted frame data.

            Raises:
                RuntimeError: If video files cannot be opened or decoded
                ValueError: If frame_ids contain invalid indices
            
            Example:
                
                Ref to Sample: `samples/SampleRandomAccess.py`
                and `samples/SampleRandomAccessWithFastInit.py`
                
                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> rgb_frames = decoder.DecodeN12ToRGB(['video.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
                >>> # Convert to PyTorch tensors on GPU (shape (height, width, 3), uint8)
                >>> rgb_tensors = [torch.as_tensor(frame).clone() for frame in rgb_frames]
            )pbdoc")
        .def(
            "GetGOPList",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, std::vector<FastStreamInfo> fastStreamInfos) {
                try {
                    std::vector<SerializedPacketBundle> bundles;
                    // Release GIL for file I/O and demuxing
                    {
                        py::gil_scoped_release release;
                        bundles = dec->get_gop_list(
                            filepaths, frame_ids, fastStreamInfos.empty() ? nullptr : fastStreamInfos.data());
                    }
                    // GIL is re-acquired here for creating Python objects

                    // Create Python list to hold results
                    py::list result_list;

                    for (auto& bundle : bundles) {
                        // Create numpy array from serialized data for this video
                        auto capsule = py::capsule(bundle.data.release(),
                                                   [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });
                        py::array_t<uint8_t> numpy_data({static_cast<py::ssize_t>(bundle.size)},
                                                        {static_cast<py::ssize_t>(sizeof(uint8_t))},
                                                        static_cast<uint8_t*>(capsule.get_pointer()),
                                                        capsule);

                        // Create tuple (numpy_data, first_frame_ids, gop_lens) for this video
                        py::tuple video_tuple =
                            py::make_tuple(numpy_data, bundle.first_frame_ids, bundle.gop_lens);

                        result_list.append(video_tuple);
                    }

                    return result_list;

                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"),
            py::arg("fastStreamInfos") = std::vector<FastStreamInfo>{},
            R"pbdoc(
            For each video, extracts the GOP(Group of Pictures) containing the requested frame and returns
            it as one serialized GOP bundle (numpy object) per video.

            Note:
                This method performs CPU-side demuxing only and does not use any GPU resources.
                Pass the returned bundles to :meth:`DecodeFromGOPListRGB` (or
                :meth:`DecodeFromGOPList` for YUV output) to run the actual decode step on GPU.

            Args:
                filepaths: List of video file paths to extract GOP data from
                frame_ids: List of frame IDs to extract GOP data for (one per video)
                fastStreamInfos: Optional list of FastStreamInfo objects containing pre-extracted
                                stream information by :func:`GetFastInitInfo`. If provided, this can
                                improve performance by avoiding stream analysis.

            Returns:
                List of tuples, one per video file, each containing

                - serialized GOP bundle (numpy object) for that video
                - list with the first frame ID of the extracted GOP
                - list with the length (frame count) of the extracted GOP

            Treat the bundle as an opaque blob: pass it to :meth:`DecodeFromGOPListRGB` /
            :meth:`DecodeFromGOPList` to decode any frame within the GOP range
            ``[first_frame_id, first_frame_id + gop_len)``, or persist it with
            :func:`SaveGopToFile` and reload it with :meth:`LoadGopsToList`.

            Raises:
                RuntimeError: If video files cannot be opened or GOP extraction fails
            
            Example:

                Ref to Sample: `samples/SampleDemuxerDecoderSeparationAccess.py`

                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> results = decoder.GetGOPList(
                ...     ['video1.mp4', 'video2.mp4'], 
                ...     [0, 10]
                ... )
                >>> for i, (gop_data, first_ids, gop_lens) in enumerate(results):
                ...     print(f"Video {i}: GOP data size = {len(gop_data)}")
                ...     print(f"  First frame IDs: {first_ids}")
                ...     print(f"  GOP lengths: {gop_lens}")
            )pbdoc")
        .def(
            "GetGOPGroups",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const py::list& requests) {
                struct SourceRequest {
                    std::string filepath;
                    // Sorted, unique decode targets. The original request order and
                    // duplicates are retained separately in original_positions.
                    std::vector<int> frame_ids;
                    std::map<int, std::vector<int>> original_positions;
                };
                struct GroupResult {
                    // Serialized demux output for one GOP: encoded packet bytes plus lookup metadata,
                    // not decoded pixels. It becomes the group's ``gop_data`` returned to Python.
                    SerializedPacketBundle bundle;
                    // Unique requested frame IDs inside this GOP, ordered for decoding.
                    std::vector<int> frame_ids;
                    // frame_positions[i] contains every index where frame_ids[i] appeared in the
                    // original request. Example: [8, 6, 8] becomes IDs [6, 8] with [[1], [0, 2]].
                    std::vector<std::vector<int>> frame_positions;
                    // Half-open display-frame range [first_frame_id,
                    // first_frame_id + gop_len) covered by bundle.
                    int first_frame_id;
                    int gop_len;
                };

                std::vector<SourceRequest> source_requests;
                source_requests.reserve(requests.size());
                for (const auto& item : requests) {
                    const py::dict request = py::cast<py::dict>(item);
                    SourceRequest source_request{
                        request["filepath"].cast<std::string>(),
                        request["frame_ids"].cast<std::vector<int>>(),
                        {},
                    };
                    for (size_t frame_position = 0; frame_position < source_request.frame_ids.size();
                         ++frame_position) {
                        const int frame_id = source_request.frame_ids[frame_position];
                        if (frame_id < 0) {
                            throw std::invalid_argument("frame IDs must be non-negative");
                        }
                        source_request.original_positions[frame_id].push_back(
                            static_cast<int>(frame_position));
                    }

                    source_request.frame_ids.clear();
                    source_request.frame_ids.reserve(source_request.original_positions.size());
                    for (const auto& [frame_id, _] : source_request.original_positions) {
                        source_request.frame_ids.push_back(frame_id);
                    }
                    source_requests.push_back(std::move(source_request));
                }

                std::vector<std::vector<GroupResult>> results_by_source(source_requests.size());
                // Index of the next sorted frame ID not yet assigned to a GOP for
                // each source request.
                std::vector<size_t> next_frame_indices(source_requests.size(), 0);
                {
                    py::gil_scoped_release release;
                    while (true) {
                        // get_gop_list extracts one GOP per source in a call. A source
                        // stays pending while it still has target IDs beyond the GOPs
                        // extracted in previous rounds.
                        std::vector<size_t> pending_source_indices;
                        for (size_t source_idx = 0; source_idx < source_requests.size(); ++source_idx) {
                            if (next_frame_indices[source_idx] <
                                source_requests[source_idx].frame_ids.size()) {
                                pending_source_indices.push_back(source_idx);
                            }
                        }
                        if (pending_source_indices.empty()) {
                            break;
                        }

                        std::vector<std::string> pending_filepaths;
                        std::vector<int> representative_ids;
                        pending_filepaths.reserve(pending_source_indices.size());
                        representative_ids.reserve(pending_source_indices.size());
                        for (const size_t source_idx : pending_source_indices) {
                            const auto& source_request = source_requests[source_idx];
                            pending_filepaths.push_back(source_request.filepath);
                            // The first unassigned target locates the next GOP for
                            // this source; all remaining targets in that GOP are
                            // consumed together below.
                            representative_ids.push_back(
                                source_request.frame_ids[next_frame_indices[source_idx]]);
                        }

                        auto bundles = dec->get_gop_list(pending_filepaths, representative_ids);
                        // get_gop_list promises one result per input path in the same
                        // order. Check that contract before mapping round-local results
                        // back to their original request indices.
                        if (bundles.size() != pending_source_indices.size()) {
                            throw std::runtime_error(
                                "GetGOPList returned a different number of bundles than requested");
                        }

                        for (size_t pending_idx = 0; pending_idx < pending_source_indices.size();
                             ++pending_idx) {
                            const size_t source_idx = pending_source_indices[pending_idx];
                            auto& bundle = bundles[pending_idx];
                            if (bundle.first_frame_ids.size() != 1 || bundle.gop_lens.size() != 1) {
                                throw std::runtime_error(
                                    "GetGOPList returned invalid per-source GOP metadata");
                            }

                            const int first_frame_id = bundle.first_frame_ids[0];
                            const int gop_len = bundle.gop_lens[0];
                            const int64_t gop_end =
                                static_cast<int64_t>(first_frame_id) + static_cast<int64_t>(gop_len);
                            const auto& source_request = source_requests[source_idx];
                            const int representative_id =
                                source_request.frame_ids[next_frame_indices[source_idx]];
                            if (gop_len <= 0 || representative_id < first_frame_id ||
                                representative_id >= gop_end) {
                                throw std::runtime_error(
                                    "demuxed GOP range does not contain its representative frame");
                            }

                            std::vector<int> grouped_ids;
                            std::vector<std::vector<int>> grouped_positions;
                            while (next_frame_indices[source_idx] < source_request.frame_ids.size()) {
                                const int frame_id = source_request.frame_ids[next_frame_indices[source_idx]];
                                if (frame_id >= gop_end) {
                                    break;
                                }
                                if (frame_id < first_frame_id) {
                                    throw std::runtime_error("target frame precedes its demuxed GOP start");
                                }
                                grouped_ids.push_back(frame_id);
                                grouped_positions.push_back(source_request.original_positions.at(frame_id));
                                ++next_frame_indices[source_idx];
                            }

                            results_by_source[source_idx].push_back(
                                {std::move(bundle), std::move(grouped_ids), std::move(grouped_positions),
                                 first_frame_id, gop_len});
                        }
                    }
                }

                py::list result;
                for (size_t source_idx = 0; source_idx < results_by_source.size(); ++source_idx) {
                    for (auto& group : results_by_source[source_idx]) {
                        auto& bundle = group.bundle;
                        auto capsule = py::capsule(bundle.data.release(),
                                                   [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });
                        py::array_t<uint8_t> numpy_data({bundle.size}, {sizeof(uint8_t)},
                                                        static_cast<uint8_t*>(capsule.get_pointer()),
                                                        capsule);
                        py::dict group_dict;
                        group_dict["gop_data"] = std::move(numpy_data);
                        // Zero-based index into the input requests list. All GOPs
                        // split from the same source request keep this index so the
                        // caller can scatter decoded frames back to that request.
                        group_dict["source_index"] = source_idx;
                        group_dict["source_name"] = source_requests[source_idx].filepath;
                        group_dict["frame_ids"] = group.frame_ids;
                        group_dict["frame_positions"] = group.frame_positions;
                        group_dict["first_frame_id"] = group.first_frame_id;
                        group_dict["gop_len"] = group.gop_len;
                        result.append(std::move(group_dict));
                    }
                }
                return result;
            },
            py::arg("requests"),
            R"pbdoc(
            Extract one serialized payload for each unique source/GOP.

            Args:
                requests: Source request dictionaries. Each dictionary must contain
                    ``filepath`` and ``frame_ids``. Decode targets are sorted and
                    de-duplicated per request, while every original position is
                    retained in ``frame_positions``.

            Returns:
                A source-major list of group dictionaries. Requests spanning GOP
                boundaries are split automatically. Each dictionary contains:

                Group dictionaries are variable length: their ``frame_ids`` lists
                need not be aligned across groups. The length of each list is the
                number of unique requested frames contained in that source/GOP,
                rather than a conventional batch dimension.

                - ``gop_data``: encoded packets and packet metadata for one GOP.
                - ``source_index``: zero-based index of the originating item in
                  ``requests``; groups split from one request share this value.
                - ``source_name``: the request's filepath.
                - ``frame_ids``: sorted unique targets contained in this GOP.
                - ``frame_positions``: original request positions for each target.
                - ``first_frame_id`` and ``gop_len``: the GOP's half-open display
                  frame range ``[first_frame_id, first_frame_id + gop_len)``.

                Pass the returned list directly to :meth:`DecodeFromGOPGroupsRGB`.

            Example:
                >>> groups = decoder.GetGOPGroups(
                ...     [{"filepath": camera_5_path, "frame_ids": [6, 7, 8, 9]}])
                >>> decoded_groups = decoder.DecodeFromGOPGroupsRGB(groups)
            )pbdoc")
        .def(
            "DecodeFromGOPRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const py::array_t<uint8_t>& numpy_data,
               const std::vector<std::string>& filepaths, const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    // Extract data pointer while holding GIL
                    const uint8_t* data_ptr = static_cast<const uint8_t*>(numpy_data.data());
                    size_t data_size = numpy_data.size();

                    std::vector<RGBFrame> result;
                    // Release GIL for GPU decoding
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_gop(data_ptr, data_size, filepaths, frame_ids, true, as_bgr, nullptr,
                                             &result);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_data"), py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            .. warning::
                **Deprecated — will be removed in version 0.3.0.**
                Use :meth:`GetGOPList` + :meth:`DecodeFromGOPListRGB` instead;
                this method is kept temporarily for backwards compatibility.

            Decodes a merged serialized GOP bundle into RGB frames without demuxing again.

            Args:
                numpy_data: Numpy array containing a merged serialized GOP bundle. No current
                            API produces data in this format anymore — do not use this method.
                filepaths: List of video file paths (for metadata purposes)
                frame_ids: List of frame IDs to decode from the bundle
                as_bgr: Whether to output in BGR format (True) or RGB format (False)

            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data

            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If frame_ids don't match the GOP data
            )pbdoc")
        .def(
            "DecodeFromPacketListRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec,
               const std::vector<std::vector<py::array_t<uint8_t>>>& numpy_datas,
               const std::vector<std::vector<int>>& packet_idxs, const std::vector<int>& widths,
               const std::vector<int>& heights, const std::vector<int>& frame_ids, bool as_bgr) {
                try {
                    // Extract packets_bytes and packet_binary_data_ptrs from numpy_datas (requires GIL)
                    std::vector<std::vector<int>> packets_bytes;
                    std::vector<std::vector<const uint8_t*>> packet_binary_data_ptrs;

                    packets_bytes.reserve(numpy_datas.size());
                    packet_binary_data_ptrs.reserve(numpy_datas.size());

                    for (size_t i = 0; i < numpy_datas.size(); ++i) {
                        const auto& frame_numpy_arrays = numpy_datas[i];
                        std::vector<int> frame_packets_bytes;
                        std::vector<const uint8_t*> frame_packet_ptrs;

                        frame_packets_bytes.reserve(frame_numpy_arrays.size());
                        frame_packet_ptrs.reserve(frame_numpy_arrays.size());

                        for (const auto& numpy_array : frame_numpy_arrays) {
                            // packets_bytes is the size of each numpy array
                            frame_packets_bytes.push_back(static_cast<int>(numpy_array.size()));
                            // packet_binary_data_ptrs is the data pointer of each numpy array
                            frame_packet_ptrs.push_back(static_cast<const uint8_t*>(numpy_array.data()));
                        }
                        frame_packets_bytes.push_back(0);
                        frame_packets_bytes.push_back(-1);

                        packets_bytes.push_back(std::move(frame_packets_bytes));
                        packet_binary_data_ptrs.push_back(std::move(frame_packet_ptrs));
                    }

                    std::vector<std::vector<int>> packet_idxs_fix = packet_idxs;
                    for (auto& packet_idx : packet_idxs_fix) {
                        packet_idx.push_back(0);
                        packet_idx.push_back(0);
                    }

                    std::vector<RGBFrame> result;
                    // Release GIL for GPU decoding
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_packet_list(packets_bytes, packet_idxs_fix, widths, heights,
                                                     packet_binary_data_ptrs, frame_ids, as_bgr, &result);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_datas"), py::arg("packet_idxs"), py::arg("widths"), py::arg("heights"),
            py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            .. warning::
                **Under development — API is unstable and subject to change without notice.**
                Do not use in production code.

            Decodes video packets into RGB frames from raw per-frame packet data arrays.

            This advanced interface takes one list of numpy arrays per frame, holding that
            frame's raw packet data — possibly produced by an external demuxer — and decodes
            them directly, without the serialized GOP bundle format used by :meth:`GetGOPList`.

            Args:
                numpy_datas: List of lists of numpy arrays containing binary packet data for each frame.
                            Each inner list contains numpy arrays for packets of one frame.
                            The function automatically extracts packet sizes and data pointers from these arrays.
                packet_idxs: List of lists containing decode indices for each frame
                widths: List of frame widths for each frame
                heights: List of frame heights for each frame
                frame_ids: List of frame IDs to decode
                as_bgr: Whether to output in BGR format (True) or RGB format (False)

            Returns:
                List of decoded RGB/BGR frames

            Raises:
                RuntimeError: If packet data is invalid or decoding fails
                ValueError: If input arrays have mismatched dimensions

            Example:
                Ref to Sample: `samples/SampleDecodeFromBinaryData.py`
            )pbdoc")
        .def(
            "DecodeFromGOPListRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec,
               const std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>&
                   numpy_datas,
               const std::vector<std::string>& filepaths, const std::vector<int>& frame_ids, bool as_bgr) {
                try {
                    // Convert numpy arrays to pointers and sizes (requires GIL)
                    std::vector<const uint8_t*> datas;
                    std::vector<size_t> sizes;
                    datas.reserve(numpy_datas.size());
                    sizes.reserve(numpy_datas.size());

                    for (const auto& arr : numpy_datas) {
                        datas.push_back(static_cast<const uint8_t*>(arr.data()));
                        sizes.push_back(arr.size());
                    }

                    std::vector<RGBFrame> result;
                    // Release GIL for GPU decoding
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_gop_list(datas, sizes, filepaths, frame_ids, true, as_bgr, nullptr,
                                                  &result);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_datas"), py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            Decodes multiple serialized GOP bundles into RGB/BGR frames.

            Args:
                numpy_datas: List of numpy arrays, each containing one serialized GOP bundle
                             from :meth:`GetGOPList` or :meth:`LoadGopsToList` (one per video)
                filepaths: List of source file paths, one for each requested frame
                frame_ids: List of target frame IDs, one for each requested frame
                as_bgr: Whether to output in BGR format (True) or RGB format (False)

            Returns:
                List of :class:`RGBFrame` objects containing the decoded RGB/BGR frames

            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If input arrays have mismatched dimensions

            Example:

                Ref to Sample: `samples/SampleDemuxerDecoderSeparationAccess.py`

                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> gop_list = decoder.GetGOPList(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> gop_data_list = [gop_data for gop_data, _, _ in gop_list]
                >>> rgb_frames = decoder.DecodeFromGOPListRGB(
                ...     gop_data_list, ['video1.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
                >>> # Convert to PyTorch tensors on GPU (shape (height, width, 3), uint8)
                >>> rgb_tensors = [torch.as_tensor(frame).clone() for frame in rgb_frames]
            )pbdoc")
        .def(
            "DecodeFromGOPGroupsRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const py::list& groups, bool as_bgr) {
                try {
                    using ByteArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
                    struct GroupLayout {
                        // Index of the originating GetGOPGroups request. Multiple
                        // GOPs split from one request have the same source_index.
                        int source_index;
                        std::string source_name;
                        std::vector<int> frame_ids;
                        std::vector<std::vector<int>> frame_positions;
                        int first_frame_id;
                        int gop_len;
                    };

                    const size_t num_groups = groups.size();
                    std::vector<ByteArray> gop_datas(num_groups);
                    std::vector<const uint8_t*> datas(num_groups);
                    std::vector<size_t> sizes(num_groups);
                    std::vector<std::string> source_names(num_groups);
                    std::vector<std::vector<int>> frame_id_groups(num_groups);
                    std::vector<GroupLayout> layouts(num_groups);

                    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
                        py::dict group = py::cast<py::dict>(groups[group_idx]);
                        for (const char* required_key :
                             {"gop_data", "source_index", "source_name", "frame_ids", "frame_positions",
                              "first_frame_id", "gop_len"}) {
                            if (!group.contains(required_key)) {
                                throw std::invalid_argument(std::string("group is missing '") + required_key +
                                                            "'");
                            }
                        }

                        gop_datas[group_idx] = group["gop_data"].cast<ByteArray>();
                        const auto& data = gop_datas[group_idx];
                        datas[group_idx] = static_cast<const uint8_t*>(data.data());
                        sizes[group_idx] = data.size();

                        GroupLayout layout{
                            group["source_index"].cast<int>(),
                            group["source_name"].cast<std::string>(),
                            group["frame_ids"].cast<std::vector<int>>(),
                            group["frame_positions"].cast<std::vector<std::vector<int>>>(),
                            group["first_frame_id"].cast<int>(),
                            group["gop_len"].cast<int>(),
                        };
                        if (layout.source_index < 0) {
                            throw std::invalid_argument("group source_index must be non-negative");
                        }
                        if (layout.frame_positions.size() != layout.frame_ids.size()) {
                            throw std::invalid_argument(
                                "group frame_positions must have one entry per frame_id");
                        }
                        for (const auto& positions : layout.frame_positions) {
                            if (positions.empty() || std::any_of(positions.begin(), positions.end(),
                                                                 [](int position) { return position < 0; })) {
                                throw std::invalid_argument(
                                    "group frame_positions entries must be non-empty and non-negative");
                            }
                        }

                        source_names[group_idx] = layout.source_name;
                        frame_id_groups[group_idx] = layout.frame_ids;
                        layouts[group_idx] = std::move(layout);
                    }

                    std::vector<RGBFrame> result;
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_gop_groups(datas, sizes, source_names, frame_id_groups, as_bgr,
                                                    result);
                    }

                    py::list decoded_groups;
                    size_t result_offset = 0;
                    for (const auto& layout : layouts) {
                        py::list frames;
                        for (size_t frame_idx = 0; frame_idx < layout.frame_ids.size(); ++frame_idx) {
                            if (result_offset >= result.size()) {
                                throw std::runtime_error(
                                    "grouped decode returned fewer frames than requested");
                            }
                            frames.append(py::cast(result[result_offset++]));
                        }

                        py::dict decoded_group;
                        decoded_group["source_index"] = layout.source_index;
                        decoded_group["source_name"] = layout.source_name;
                        decoded_group["frame_ids"] = layout.frame_ids;
                        decoded_group["frame_positions"] = layout.frame_positions;
                        decoded_group["first_frame_id"] = layout.first_frame_id;
                        decoded_group["gop_len"] = layout.gop_len;
                        decoded_group["frames"] = std::move(frames);
                        decoded_groups.append(std::move(decoded_group));
                    }
                    if (result_offset != result.size()) {
                        throw std::runtime_error("grouped decode returned more frames than requested");
                    }
                    return decoded_groups;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("groups"), py::arg("as_bgr") = false,
            R"pbdoc(
            Decode several target frames from each unique source/GOP bundle.

            Unlike :meth:`DecodeFromGOPListRGB`, which assigns one decoder task to
            every output frame, this method assigns one decoder task to every GOP
            group. All target frames in a group are produced during the same packet
            traversal and NVDEC decode chain.

            Groups are variable length: each group's ``frame_ids`` and returned
            ``frames`` lists contain the unique requested frames in that source/GOP,
            so their lengths need not be aligned across groups.

            Args:
                groups: Group dictionaries returned by :meth:`GetGOPGroups`. Each
                    dictionary carries one serialized GOP, its unique target frame
                    IDs, and every target's original positions.
                as_bgr: Return BGR when true, RGB when false.

            Returns:
                One dictionary per input group. Metadata and ``frame_positions`` are
                preserved, and ``frames`` contains one RGBFrame per unique frame ID.
                Callers can scatter each frame to all corresponding original
                positions without decoding duplicates.

            Example:
                >>> groups = demuxer.GetGOPGroups(
                ...     [{"filepath": camera_5_path, "frame_ids": [6, 7, 8, 9]}])
                >>> decoded_groups = decoder.DecodeFromGOPGroupsRGB(groups)
            )pbdoc")
        .def(
            "DecodeFromGOPList",
            [](std::shared_ptr<PyNvGopDecoder>& dec,
               const std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>&
                   numpy_datas,
               const std::vector<std::string>& filepaths, const std::vector<int>& frame_ids) {
                try {
                    // Convert numpy arrays to pointers and sizes (requires GIL)
                    std::vector<const uint8_t*> datas;
                    std::vector<size_t> sizes;
                    datas.reserve(numpy_datas.size());
                    sizes.reserve(numpy_datas.size());

                    for (const auto& arr : numpy_datas) {
                        datas.push_back(static_cast<const uint8_t*>(arr.data()));
                        sizes.push_back(arr.size());
                    }

                    std::vector<DecodedFrameExt> result;
                    // Release GIL for GPU decoding
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_gop_list(datas, sizes, filepaths, frame_ids, false, false, &result,
                                                  nullptr);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_datas"), py::arg("filepaths"), py::arg("frame_ids"),
            R"pbdoc(
            Decodes multiple serialized GOP bundles into native YUV frames.

            If you need RGB/BGR output, use :meth:`DecodeFromGOPListRGB` instead.

            Args:
                numpy_datas: List of numpy arrays, each containing one serialized GOP bundle
                             from :meth:`GetGOPList` or :meth:`LoadGopsToList` (one per video)
                filepaths: List of source file paths, one for each requested frame
                frame_ids: List of target frame IDs, one for each requested frame

            Returns:
                List of :class:`DecodedFrameExt` objects containing decoded native YUV frame data

            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If input arrays have mismatched dimensions

            Example:
                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> gop_list = decoder.GetGOPList(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> gop_data_list = [gop_data for gop_data, _, _ in gop_list]
                >>> frames = decoder.DecodeFromGOPList(gop_data_list, ['video1.mp4', 'video2.mp4'], [0, 10])
                >>> # Convert to PyTorch tensors on GPU (NV12 layout: (height * 3 // 2, width), uint8)
                >>> nv12_tensors = [torch.as_tensor(frame).clone() for frame in frames]
            )pbdoc")
        .def(
            "LoadGopsToList",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& file_paths) {
                try {
                    std::vector<std::vector<uint8_t>> gop_data_list;
                    // Release GIL for file I/O
                    {
                        py::gil_scoped_release release;
                        dec->LoadGOPFromFiles(file_paths, gop_data_list);
                    }
                    // GIL is re-acquired here for creating Python objects

                    py::list result_list;

                    for (auto& gop_data : gop_data_list) {
                        if (gop_data.empty()) {
                            throw std::runtime_error("[ERROR] Loaded GOP data is empty");
                        }

                        size_t size = gop_data.size();

                        // Allocate new memory and transfer data to Python
                        // Use unique_ptr for exception safety
                        std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
                        std::memcpy(buffer.get(), gop_data.data(), size);

                        // Transfer ownership to capsule (exception-safe)
                        uint8_t* raw_ptr = buffer.release();
                        auto capsule =
                            py::capsule(raw_ptr, [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });

                        // Create numpy array
                        py::array_t<uint8_t> numpy_data({static_cast<py::ssize_t>(size)},
                                                        {static_cast<py::ssize_t>(sizeof(uint8_t))}, raw_ptr,
                                                        capsule);

                        result_list.append(std::move(numpy_data));
                    }

                    return result_list;

                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("file_paths"),
            R"pbdoc(
            Load serialized GOP bundles from multiple binary files and return as a list of numpy arrays.

            This method loads serialized GOP bundles from binary files (previously saved with
            :func:`SaveGopToFile`) and returns one bundle (numpy array) per file, ready to be
            decoded with :meth:`DecodeFromGOPListRGB` or :meth:`DecodeFromGOPList`.

            Args:
                file_paths: List of paths to GOP binary files to load

            Returns:
                List of numpy arrays, each containing the serialized GOP bundle from one file,
                in the same format as returned by :meth:`GetGOPList`.
            
            Raises:
                RuntimeError: If any file cannot be read or has invalid format
                ValueError: If file_paths is empty or files have invalid GOP format
            
            Example:
                Ref to Sample: `samples/SampleDecodeFromGopFiles.py`

                >>> # GOP files previously saved with SaveGopToFile()
                >>> gop_data_list = decoder.LoadGopsToList(['gop_0.bin', 'gop_1.bin'])
                >>> frames = decoder.DecodeFromGOPListRGB(
                ...     gop_data_list, ['v1.mp4', 'v2.mp4'], [0, 10], as_bgr=True)
                >>> rgb_tensors = [torch.as_tensor(frame).clone() for frame in frames]
            )pbdoc")
        .def(
            "DecodeFromPacketListInitialize",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<int>& codec_ids) {
                try {
                    // Parameter validation
                    if (codec_ids.empty()) {
                        throw std::runtime_error("codec_ids cannot be empty");
                    }

                    // Call the C++ method
                    int result = dec->InitializeDecoders(codec_ids);

                    if (result != 0) {
                        throw std::runtime_error("InitializeDecoders failed with error code: " +
                                                 std::to_string(result));
                    }

                    return 0;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("codec_ids"), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            .. warning::
                **Under development — API is unstable and subject to change without notice.**
                Do not use in production code.

            Initializes NvDecoder instances for video files.

            This method creates NvDecoder instances for each video file, preparing
            them for efficient decoding operations. It is used before :meth:`DecodeFromPacketListRGB`.

            Args:
                codec_ids: List of video codec IDs

            Returns:
                0 if initialization successful

            Raises:
                RuntimeError: If any parameters are invalid or initialization fails
                ValueError: If codec_ids is empty

            Example:
                Ref to Sample: `samples/SampleDecodeFromBinaryData.py`

            )pbdoc")
        .def(
            "release_device_memory", [](std::shared_ptr<PyNvGopDecoder>& dec) { dec->ReleaseMemPools(); },
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Release GPU device memory pool to free up GPU memory.
            
            This method releases the GPU memory pool and resets the pool state.
            This is useful for temporarily freeing excessive GPU memory usage.
            
            Note: After calling this method, the memory pool will need to be
            re-allocated on the next decode operation.
            
            Example:
                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4'], [0, 10, 20])
                >>> tensors = [torch.as_tensor(frame).clone() for frame in frames]
                >>> decoder.release_device_memory()  # Free GPU memory pool
            )pbdoc")
        .def(
            "release_decoder", [](std::shared_ptr<PyNvGopDecoder>& dec) { dec->ReleaseDecoder(); },
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Release all decoder instances to free up GPU memory.
            
            This method clears all decoder instances, which releases 
            NvDecoder instances and their GPU frame buffers
            
            This is useful for freeing GPU memory occupied by decoder instances.
            
            Note: After calling this method, decoder instances will need to be
            re-created on the next decode operation.
            
            Example:
                >>> decoder = CreateGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4'], [0, 10, 20])
                >>> tensors = [torch.as_tensor(frame).clone() for frame in frames]
                >>> decoder.release_decoder()  # Free decoder instances
            )pbdoc");
}
