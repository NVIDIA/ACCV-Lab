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

    // To do, we can reuse current context, we can check its func
    // ck(cuCtxGetCurrent(&cuContext));
    this->cu_context = nullptr;
    if (!this->cu_context) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, this->gpu_id));
        ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
        this->destroy_context = true;
    }
    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");
    }

    // Temporarily push context for stream creation, then immediately pop.
    // This ensures the destructor can run on any thread without issues.
    ck(cuCtxPushCurrent(this->cu_context));
    ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
    ck(cuCtxPopCurrent(NULL));
}

void PyNvGopDecoder::ensureDemuxRunnersInitialized() {
    if (!demux_runners.empty()) {
        return;  // Already initialized
    }

    demux_runners.reserve(max_num_files);
    for (size_t i = 0; i < max_num_files; ++i) {
        demux_runners.emplace_back();
    }
}

void PyNvGopDecoder::ensureDecodeRunnersInitialized() {
    if (!decode_runners.empty()) {
        return;  // Already initialized
    }

    decode_runners.reserve(max_num_files);
    for (size_t i = 0; i < max_num_files; ++i) {
        decode_runners.emplace_back();
    }
}

void PyNvGopDecoder::ensureMergeRunnersInitialized() {
    if (!merge_runners.empty()) {
        return;  // Already initialized
    }

    // Initialize merge thread pool with max_num_files threads for parallel file processing
    merge_runners.reserve(max_num_files);
    for (size_t i = 0; i < max_num_files; ++i) {
        merge_runners.emplace_back();
    }
}

PyNvGopDecoder::PyNvGopDecoder(int iMaxFileNum, int iGpu, bool bSuppressNoColorRangeWarning)
    : max_num_files(iMaxFileNum),
      gpu_id(iGpu),
      suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning) {
#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvGopDecoder object" << std::endl;
#endif

    this->last_decoded_frame_infos.resize(this->max_num_files);
    reset_last_decoded_frame_infos(this->last_decoded_frame_infos);
}

void PyNvGopDecoder::force_join_all() {
    // Force join all demux runners
    for (auto& runner : demux_runners) {
        runner.force_join();
    }

    // Force join all decode runners
    for (auto& runner : decode_runners) {
        runner.force_join();
    }

    // Force join all merge runners
    for (auto& runner : merge_runners) {
        runner.force_join();
    }
}

PyNvGopDecoder::~PyNvGopDecoder() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvGopDecoder object" << std::endl;
#endif

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

        if (this->cu_stream) {
            ck(cuStreamDestroy(this->cu_stream));
        }

        ck(cuCtxPopCurrent(NULL));
    }

    if (this->destroy_context) {
        // Only release the primary context reference.
        // No need to pop - we use temporary push/pop pattern instead.
        ck(cuDevicePrimaryCtxRelease(this->gpu_id));
    }

    // Clean up thread runners
    for (auto& runner : demux_runners) {
        runner.join();
    }
    for (auto& runner : decode_runners) {
        runner.join();
    }
    for (auto& runner : merge_runners) {
        runner.join();
    }
}

void Init_PyNvGopDecoder(py::module& m) {
    py::class_<FastStreamInfo>(m, "FastStreamInfo",
                               R"pbdoc(
        Fast stream information structure for video files.
        
        This structure contains essential metadata about video streams that can be
        extracted quickly without full video analysis. Used for performance optimization
        in multi-file decoding scenarios.
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("codec_type", &FastStreamInfo::codec_type,
                       R"pbdoc(FFmpeg codec type (AVMediaType enum value))pbdoc")
        .def_readwrite("codec_id", &FastStreamInfo::codec_id,
                       R"pbdoc(FFmpeg codec ID (AVCodecID enum value, e.g., AV_CODEC_ID_H264=27))pbdoc")
        .def_readwrite("width", &FastStreamInfo::width, R"pbdoc(Video frame width in pixels))pbdoc")
        .def_readwrite("height", &FastStreamInfo::height, R"pbdoc(Video frame height in pixels))pbdoc")
        .def_readwrite("format", &FastStreamInfo::format,
                       R"pbdoc(Pixel format (AVPixelFormat enum value))pbdoc")
        .def_readwrite("time_base_num", &FastStreamInfo::time_base_num,
                       R"pbdoc(Time base numerator for timestamp calculations))pbdoc")
        .def_readwrite("time_base_den", &FastStreamInfo::time_base_den,
                       R"pbdoc(Time base denominator for timestamp calculations))pbdoc")
        .def_readwrite("avg_frame_rate_num", &FastStreamInfo::avg_frame_rate_num,
                       R"pbdoc(Average frame rate numerator))pbdoc")
        .def_readwrite("avg_frame_rate_den", &FastStreamInfo::avg_frame_rate_den,
                       R"pbdoc(Average frame rate denominator))pbdoc")
        .def_readwrite("r_frame_rate_num", &FastStreamInfo::r_frame_rate_num,
                       R"pbdoc(Real frame rate numerator))pbdoc")
        .def_readwrite("r_frame_rate_den", &FastStreamInfo::r_frame_rate_den,
                       R"pbdoc(Real frame rate denominator))pbdoc")
        .def_readwrite("start_time", &FastStreamInfo::start_time,
                       R"pbdoc(Start time of the stream in time base units))pbdoc")
        .def_readwrite("duration", &FastStreamInfo::duration,
                       R"pbdoc(Duration of the stream in time base units))pbdoc");

    m.def(
        "CreateGopDecoder",
        [](int maxfiles, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvGopDecoder>(maxfiles, iGpu, suppressNoColorRangeWarning);
        },
        py::arg("maxfiles"), py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
        Initialize GOP decoder with set of particular parameters.
        
        This factory function creates a PyNvGopDecoder instance with the specified
        configuration. It's the recommended way to create decoder instances.
        
        Args:
            maxfiles: Maximum number of unique files that can be processed concurrently
            iGpu: GPU device ID to use for decoding (0 for primary GPU)
            suppressNoColorRangeWarning: Suppress warning when no color range can be extracted from video files (limited/MPEG range is assumed)
        
        Returns:
            PyNvGopDecoder instance configured with the specified parameters
        
        Raises:
            RuntimeError: If GPU initialization fails or parameters are invalid
        
        Example:
            >>> decoder = CreateGopDecoder(maxfiles=3, iGpu=0)
            >>> frames = decoder.Decode(['v0.mp4', 'v1.mp4', 'v2.mp4'], [0, 10, 20])
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
        Extracts FastStreamInfo from video files automatically.
        
        This function quickly extracts essential stream information from video files
        without performing full video analysis. The extracted information can be used
        to optimize decoding performance in multi-file scenarios.
        
        Args:
            filepaths: List of video file paths to analyze
        
        Returns:
            List of FastStreamInfo objects containing stream information for each file
        
        Raises:
            RuntimeError: If files cannot be opened or stream information cannot be extracted
            ValueError: If filepaths is empty
        
        Example:
            >>> stream_infos = GetFastInitInfo(['video1.mp4', 'video2.mp4'])
        )pbdoc");

    m.def(
        "SavePacketsToFile",
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
        Saves numpy array data to a binary file.
        
        This function saves serialized packet data to a binary file for later use.
        Useful for caching packet data to avoid repeated extraction.
        
        Args:
            numpy_data: Numpy array containing binary data to save
            dst_filepath: Destination file path where data will be saved
        
        Raises:
            RuntimeError: If file cannot be written or data is invalid
            ValueError: If dst_filepath is empty
        
        Example:
            >>> gop_data, first_ids, gop_lens = decoder.GetGOP(['v0.mp4', 'v1.mp4', 'v2.mp4'], [0, 10, 20])
            >>> SavePacketsToFile(packets, 'cached_packets.bin')
        )pbdoc");

    py::class_<PyNvGopDecoder, shared_ptr<PyNvGopDecoder>>(m, "PyNvGopDecoder", py::module_local())
        .def(py::init<int, int, bool>(), py::arg("maxfiles"), py::arg("iGpu") = 0,
             py::arg("suppressNoColorRangeWarning") = false,
             R"pbdoc(
            Initialize decoder with set of particular parameters.
            
            Args:
                maxfiles: Maximum number of unique files that can be processed concurrently
                iGpu: GPU device ID to use for decoding (0 for primary GPU)
                suppressNoColorRangeWarning: Suppress warning when no color range can be extracted
                                            from video files (limited/MPEG range is assumed)
            
            Raises:
                RuntimeError: If GPU initialization fails or parameters are invalid
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
            Decodes video file stream into uncompressed YUV data.
            
            This method performs GPU-accelerated decoding of video frames using NVIDIA hardware.
            It supports multiple video files and can decode specific frame IDs from each file.
            The method uses GOP-based decoding for efficient random access.
            
            Args:
                filepaths: List of video file paths to decode from. All files must be
                           accessible and contain valid video streams.
                frame_ids: List of frame IDs to decode. Each frame ID corresponds to
                           a specific frame in the video sequence.
                fastStreamInfos: Optional list of FastStreamInfo objects containing
                                pre-extracted stream information by `GetFastInitInfo`.
                                If provided, this can improve performance by avoiding 
                                stream analysis.
            
            Returns:
                List of DecodedFrameExt objects containing the decoded frame data.
                Each frame includes YUV pixel data, metadata, and timing information.
            
            Raises:
                RuntimeError: If video files cannot be opened or decoded
                ValueError: If frame_ids contain invalid indices
            
            Example:
                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> print(f"Decoded {len(frames)} frames")
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
            Decodes video file stream into uncompressed RGB/BGR data.
            
            This method performs GPU-accelerated decoding and color space conversion
            from YUV to RGB/BGR format. It's optimized for machine learning applications
            that require RGB input data.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
                as_bgr: Whether to output in BGR format (True) or RGB format (False). BGR is commonly used in OpenCV applications.
                fastStreamInfos: Optional list of FastStreamInfo objects containing pre-extracted stream information by `GetFastInitInfo`. If provided, this can improve performance by avoiding stream analysis.
            
            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data.
                Each frame includes RGB/BGR pixel data and metadata.
            
            Raises:
                RuntimeError: If video files cannot be opened or decoded
                ValueError: If frame_ids contain invalid indices
            
            Example:
                
                Ref to Sample: `samples/SampleRandomAccess.py`
                and `samples/SampleRandomAccessWithFastInit.py`
                
                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> rgb_frames = decoder.DecodeN12ToRGB(['video.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
                >>> print(f"Decoded {len(rgb_frames)} RGB frames")
            )pbdoc")
        .def(
            "GetGOP",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, std::vector<FastStreamInfo> fastStreamInfos) {
                try {
                    SerializedPacketBundle serialized_data;
                    // Release GIL for file I/O and demuxing
                    {
                        py::gil_scoped_release release;
                        serialized_data = dec->get_gop(
                            filepaths, frame_ids, fastStreamInfos.empty() ? nullptr : fastStreamInfos.data());
                    }
                    // GIL is re-acquired here for creating Python objects

                    // Create numpy array from serialized data
                    auto capsule = py::capsule(serialized_data.data.release(),
                                               [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });
                    py::array_t<uint8_t> numpy_data(serialized_data.size,
                                                    static_cast<uint8_t*>(capsule.get_pointer()), capsule);

                    // Return tuple with numpy_data, gop_lens, and first_frame_ids directly from the bundle
                    return py::make_tuple(numpy_data, serialized_data.first_frame_ids,
                                          serialized_data.gop_lens);

                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"),
            py::arg("fastStreamInfos") = std::vector<FastStreamInfo>{},
            R"pbdoc(
            Extracts video GOP data without performing the decode step.
            
            This method extracts video GOP (Group of Pictures) data from the specified frames and returns
            them in a self-contained binary format. The GOP data can be decoded later
            using `DecodeFromPacket` or `DecodeFromPacketRGB` methods, enabling separation 
            of demuxing and decoding.
            
            Args:
                filepaths: List of video file paths to extract GOP data from
                frame_ids: List of frame IDs to extract GOP data for
                fastStreamInfos: Optional list of FastStreamInfo objects containing pre-extracted stream information by `GetFastInitInfo`. If provided, this can improve performance by avoiding stream analysis.
            
            Returns:
                Tuple containing:
                - numpy array with serialized GOP data
                - list of first frame IDs for each GOP
                - list of GOP lengths for each GOP
            
            The numpy array contains a self-contained binary format with embedded frame offset table:
            - Header: total_frames (uint32_t) + frame_offsets array (size_t[total_frames])
            - Frame data blocks follow the header
            - Parse the header once to get direct access to any frame
            - Enables efficient random access and parallel processing
            - No external metadata files needed
            
            Raises:
                RuntimeError: If video files cannot be opened or GOP extraction fails
            
            Example:

                Ref to Sample: `samples/SampleSeparationAccess.py`

                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> gop_data, first_ids, gop_lens = decoder.GetGOP(['video.mp4', 'video2.mp4'], [0, 10])
                >>> print(f"Extracted GOP data for {len(first_ids)} GOPs")
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
                        py::array_t<uint8_t> numpy_data(
                            bundle.size, static_cast<uint8_t*>(capsule.get_pointer()), capsule);

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
            Extracts video GOP data for multiple videos and returns them as separate bundles.
            
            This method is similar to GetGOP but returns a separate bundle for each video file
            instead of merging all data into one bundle. This is useful when you want to cache
            or process each video's data independently.
            
            Args:
                filepaths: List of video file paths to extract GOP data from
                frame_ids: List of frame IDs to extract GOP data for (one per video)
                fastStreamInfos: Optional list of FastStreamInfo objects containing pre-extracted 
                                stream information by `GetFastInitInfo`. If provided, this can 
                                improve performance by avoiding stream analysis.
            
            Returns:
                List of tuples, one per video file, each containing:
                - numpy array with serialized GOP data for that video
                - list of first frame IDs for each GOP in that video
                - list of GOP lengths for each GOP in that video
            
            Each numpy array contains a self-contained binary format with embedded frame offset table:
            - Header: total_frames (uint32_t) + frame_offsets array (size_t[total_frames])
            - Frame data blocks follow the header
            - Parse the header once to get direct access to any frame
            - Enables efficient random access and parallel processing
            - No external metadata files needed
            
            Raises:
                RuntimeError: If video files cannot be opened or GOP extraction fails
            
            Example:

                Ref to Sample: `samples/SampleSeparationAccessGOPListAPI.py`

                >>> decoder = PyNvGopDecoder(maxfiles=10)
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
            "DecodeFromGOP",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const py::array_t<uint8_t>& numpy_data,
               const std::vector<std::string>& filepaths, const std::vector<int> frame_ids) {
                try {
                    // Extract data pointer while holding GIL
                    const uint8_t* data_ptr = static_cast<const uint8_t*>(numpy_data.data());
                    size_t data_size = numpy_data.size();

                    std::vector<DecodedFrameExt> result;
                    // Release GIL for GPU decoding
                    {
                        py::gil_scoped_release release;
                        dec->decode_from_gop(data_ptr, data_size, filepaths, frame_ids, false, false, &result,
                                             nullptr);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_data"), py::arg("filepaths"), py::arg("frame_ids"),
            R"pbdoc(
            Decodes video GOP data into YUV frames without demuxing again.
            
            This method decodes previously extracted GOP data into YUV frames. It's
            useful for scenarios where you want to separate GOP extraction from
            decoding, or when you have pre-extracted GOP data.
            
            Args:
                numpy_data: Numpy array containing serialized GOP data from `GetGOP`
                filepaths: List of video file paths (for metadata purposes)
                frame_ids: List of frame IDs to decode from the GOP data
            
            Returns:
                List of DecodedFrameExt objects containing the decoded YUV frame data
            
            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If frame_ids don't match the GOP data
            
            Example:
                >>> gop_data, first_ids, gop_lens = decoder.GetGOP(['video.mp4', 'video2.mp4'], [0, 10])
                >>> frames = decoder.DecodeFromGOP(gop_data, ['video.mp4', 'video2.mp4'], [0, 10])
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
            Decodes video GOP data into RGB frames without demuxing again.
            
            This method decodes previously extracted GOP data into RGB/BGR frames.
            It's useful for scenarios where you want to separate GOP extraction
            from decoding and need RGB output.
            
            Args:
                numpy_data: Numpy array containing serialized GOP data from `GetGOP`
                filepaths: List of video file paths (for metadata purposes)
                frame_ids: List of frame IDs to decode from the GOP data
                as_bgr: Whether to output in BGR format (True) or RGB format (False)
            
            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data
            
            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If frame_ids don't match the GOP data
            
            Example:

                Ref to Sample: `samples/SampleSeparationAccess.py`
                
                >>> gop_data, first_ids, gop_lens = decoder.GetGOP(['video.mp4', 'video2.mp4'], [0, 10])
                >>> rgb_frames = decoder.DecodeFromGOPRGB(gop_data, ['video.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
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
            Decodes video packets into RGB frames using separate packet data arrays (V2 interface).
            
            This advanced interface allows direct control over packet data by providing
            separate numpy arrays for each frame's binary data even from other demuxer lib, 
            enabling more flexible packet management and processing.
            
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
            
            Note:
                This interface allows direct control over packet data by providing separate numpy arrays
                for each frame's binary data. The function automatically extracts packet sizes and data pointers
                from the numpy arrays, enabling more flexible packet management and processing.
            )pbdoc")
        .def(
            "DecodeFromGOPListRGB",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<py::array_t<uint8_t>>& numpy_datas,
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
                        dec->decode_from_gop_list(datas, sizes, filepaths, frame_ids, as_bgr, &result);
                    }
                    return result;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_datas"), py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            Decodes multiple serialized GOP bundles into RGB/BGR frames.
            
            This method parses each bundle, reconstructs per-frame packet queues, and decodes
            via a unified pipeline. Useful for processing multiple GOP bundles simultaneously.
            
            Args:
                numpy_datas: List of numpy arrays, each containing a SerializedPacketBundle from `GetGOP`
                filepaths: List of source file paths for each requested frame (aggregated)
                frame_ids: List of target frame IDs (aggregated across all bundles)
                as_bgr: Whether to output in BGR format (True) or RGB format (False)
            
            Returns:
                List of decoded RGB/BGR frames
            
            Raises:
                RuntimeError: If GOP data is invalid or decoding fails
                ValueError: If input arrays have mismatched dimensions

            Example:
                Ref to Sample: `samples/SampleSeparationAccessGOPListAPI.py`
            
            Note:
                The method parses each bundle, reconstructs per-frame packet queues, and decodes
                via a unified pipeline.
            )pbdoc")
        .def(
            "LoadGops",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const std::vector<std::string>& file_paths) {
                try {
                    std::unique_ptr<uint8_t[]> merged_data;
                    size_t merged_size;

                    // Release GIL for file I/O
                    {
                        py::gil_scoped_release release;
                        dec->MergeBinaryFilesToPacketData(file_paths, merged_data, merged_size);
                    }
                    // GIL is re-acquired here for creating Python objects

                    // Create numpy array from merged data
                    auto capsule = py::capsule(merged_data.release(),
                                               [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });
                    py::array_t<uint8_t> numpy_data(merged_size, static_cast<uint8_t*>(capsule.get_pointer()),
                                                    capsule);

                    return numpy_data;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("file_paths"),
            R"pbdoc(
            Merges multiple binary packet files into a single numpy array.
            
            This method merges multiple binary packet files into a single contiguous
            numpy array for efficient processing.
            
            Args:
                file_paths: List of file paths to binary packet files
            
            Returns:
                Numpy array containing merged packet data compatible with decode_from_packet
            
            Raises:
                RuntimeError: If files cannot be read or merged
                ValueError: If file_paths is empty

            Example:
                Ref to Sample: `samples/SampleDecodeFromGopFiles.py`

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
                        py::array_t<uint8_t> numpy_data(size, raw_ptr, capsule);

                        result_list.append(std::move(numpy_data));
                    }

                    return result_list;

                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("file_paths"),
            R"pbdoc(
            Load GOP data from multiple binary files and return as a list of numpy arrays.
            
            This method loads serialized GOP bundles from binary files (previously saved
            with SavePacketsToFile) and returns them as separate numpy arrays, one per file.
            This is the companion function to GetGOPList, enabling distributed GOP caching
            and selective loading workflows.
            
            Key Differences from LoadGops:
            - LoadGops: Merges all files into ONE numpy array (for use with DecodeFromGOPRGB)
            - LoadGopList: Returns separate numpy arrays (for use with DecodeFromGOPListRGB)
            
            Args:
                file_paths: List of paths to GOP binary files to load
            
            Returns:
                List of numpy arrays, each containing the GOP data from one file.
                Each numpy array has the same format as returned by GetGOP/GetGOPList.
            
            Raises:
                RuntimeError: If any file cannot be read or has invalid format
                ValueError: If file_paths is empty or files have invalid GOP format
            
            Example:
                Ref to Sample: `samples/SampleSeparationAccessGOPToListAPI.py`
                
                >>> # Workflow 1: Save GOP data to separate files (from GetGOPList)
                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> gop_list = decoder.GetGOPList(['v1.mp4', 'v2.mp4'], [0, 10])
                >>> for i, (gop_data, _, _) in enumerate(gop_list):
                ...     SavePacketsToFile(gop_data, f'gop_{i}.bin')
                
                >>> # Workflow 2: Load GOP data back and decode
                >>> loaded_gop_list = decoder.LoadGopList(['gop_0.bin', 'gop_1.bin'])
                >>> frames = decoder.DecodeFromGOPListRGB(
                ...     loaded_gop_list, 
                ...     ['v1.mp4', 'v2.mp4'],
                ...     [0, 10],
                ...     as_bgr=True
                ... )
                
                >>> # Workflow 3: Selective loading (only load needed videos)
                >>> # Only load video 1's GOP data
                >>> loaded_gop = decoder.LoadGopList(['gop_1.bin'])
                >>> frames = decoder.DecodeFromGOPListRGB(
                ...     loaded_gop,
                ...     ['v2.mp4'],
                ...     [10],
                ...     as_bgr=True
                ... )
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
            Initializes NvDecoder instances for video files.
            
            This method creates NvDecoder instances for each video file, preparing
            them for efficient decoding operations. It is used before `DecodeFromPacketListRGB`.
            
            Args:
                codec_ids: List of video codec IDs from `ParseSerializedPacketBundle`
            
            Returns:
                0 if initialization successful
            
            Raises:
                RuntimeError: If any parameters are invalid or initialization fails
                ValueError: If codec_ids is empty
            
            Example:
                Ref to Sample: `samples/SampleDecodeFromBinaryData.py`

            )pbdoc")
        .def(
            "ParseSerializedPacketBundle",
            [](std::shared_ptr<PyNvGopDecoder>& dec, const py::array_t<uint8_t>& numpy_data) {
                try {
                    // Parse serialized packet data using the existing C++ function
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

                    const uint32_t total_frames = PyNvGopDecoder::parseSerializedPacketData(
                        static_cast<const uint8_t*>(numpy_data.data()), numpy_data.size(), color_ranges,
                        codec_ids, widths, heights, frame_sizes, gop_lens, first_frame_ids, packets_bytes,
                        decode_idxs, packet_binary_data_ptrs, packet_binary_data_sizes);

                    // Create numpy arrays for packet_binary_data_arrays
                    std::vector<py::array_t<uint8_t>> packet_binary_data_arrays;
                    packet_binary_data_arrays.reserve(total_frames);

                    for (uint32_t i = 0; i < total_frames; ++i) {
                        // Create numpy array view from the binary data pointer and size
                        // Data ownership belongs to the input numpy_data, so we create a view without capsule
                        py::array_t<uint8_t> numpy_array(
                            packet_binary_data_sizes[i],
                            static_cast<const uint8_t*>(packet_binary_data_ptrs[i]));
                        packet_binary_data_arrays.push_back(numpy_array);
                    }

                    // Return all the required parameters as a tuple
                    return py::make_tuple(color_ranges, codec_ids, widths, heights, frame_sizes, gop_lens,
                                          first_frame_ids, packet_binary_data_arrays,
                                          packet_binary_data_sizes, packets_bytes, decode_idxs);

                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("numpy_data"),
            R"pbdoc(
            Parses a SerializedPacketBundle and extracts parameters for DecodeFromPacketListRGB.
            
            This method takes serialized packet data and extracts all the necessary parameters
            required for the DecodeFromPacketListRGB interface, including creating numpy arrays
            for the packet binary data. You can use this method to get the parameters from `GetGOP`.
            
            Args:
                numpy_data: Numpy array containing serialized packet data from GetGOP
            
            Returns:
                Tuple containing:
                - color_ranges: List of color ranges for each frame
                - codec_ids: List of codec IDs for each frame
                - widths: List of frame widths for each frame
                - heights: List of frame heights for each frame
                - frame_sizes: List of frame sizes for each frame
                - gop_lens: List of GOP lengths for each frame
                - first_frame_ids: List of first frame IDs for each GOP
                - packet_binary_data_arrays: List of numpy arrays containing binary packet data for each frame
                - packet_binary_data_sizes: List of sizes of the packet binary data for each frame
                - packets_bytes: List of lists containing packet sizes for each frame
                - decode_idxs: List of lists containing decode indices for each frame
            
            Raises:
                RuntimeError: If packet data is invalid or parsing fails
            
            Example:
                Ref to Sample: `samples/SampleDecodeFromBinaryData.py`
            )pbdoc")
        .def(
            "MergePacketDataToOne",
            [](std::shared_ptr<PyNvGopDecoder>& dec,
               const std::vector<py::array_t<uint8_t>>& packet_data_arrays) {
                try {
                    // Validate input
                    if (packet_data_arrays.empty()) {
                        throw std::runtime_error("packet_data_arrays cannot be empty");
                    }

                    // Convert numpy arrays to C++ pointers and sizes (requires GIL)
                    std::vector<uint8_t*> buffer_pointers;
                    std::vector<size_t> buffer_sizes;

                    buffer_pointers.reserve(packet_data_arrays.size());
                    buffer_sizes.reserve(packet_data_arrays.size());

                    for (const auto& numpy_array : packet_data_arrays) {
                        buffer_pointers.push_back(
                            const_cast<uint8_t*>(static_cast<const uint8_t*>(numpy_array.data())));
                        buffer_sizes.push_back(numpy_array.size());
                    }

                    std::unique_ptr<uint8_t[]> merged_data;
                    size_t merged_size;

                    // Release GIL for memory operation
                    {
                        py::gil_scoped_release release;
                        dec->MergePacketDataToOne(buffer_pointers, buffer_sizes, merged_data, merged_size);
                    }
                    // GIL is re-acquired here for creating Python objects

                    // Create numpy array from merged data
                    auto capsule = py::capsule(merged_data.release(),
                                               [](void* ptr) { delete[] static_cast<uint8_t*>(ptr); });
                    py::array_t<uint8_t> numpy_data(merged_size, static_cast<uint8_t*>(capsule.get_pointer()),
                                                    capsule);

                    return numpy_data;
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("packet_data_arrays"),
            R"pbdoc(
            Merges multiple packet data arrays into a single numpy array.
            
            This method takes multiple numpy arrays containing SerializedPacketBundle data
            and merges them into a single contiguous numpy array. This is useful for
            combining packet data from different sources or files into one unified
            data structure for processing.
            
            Args:
                packet_data_arrays: List of numpy arrays, each containing SerializedPacketBundle data
            
            Returns:
                Numpy array containing merged packet data
            
            Raises:
                RuntimeError: If arrays cannot be merged
                ValueError: If packet_data_arrays is empty
            
            Note:
                This method is designed to work efficiently with large datasets and uses
                parallel processing for optimal performance.
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
                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4'], [0, 10, 20])
                >>> decoder.release_device_memory()  # Free GPU memory pool
            )pbdoc")
        .def(
            "release_decoder", [](std::shared_ptr<PyNvGopDecoder>& dec) { dec->ReleaseDecoder(); },
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Release all decoder instances to free up GPU memory.
            
            This method clears all decoder instances, which releases:
            - NvDecoder instances and their GPU frame buffers
            
            This is useful for freeing GPU memory occupied by decoder instances.
            
            Note: After calling this method, decoder instances will need to be
            re-created on the next decode operation.
            
            Example:
                >>> decoder = PyNvGopDecoder(maxfiles=10)
                >>> frames = decoder.Decode(['video1.mp4'], [0, 10, 20])
                >>> decoder.release_decoder()  # Free decoder instances
            )pbdoc");
}
