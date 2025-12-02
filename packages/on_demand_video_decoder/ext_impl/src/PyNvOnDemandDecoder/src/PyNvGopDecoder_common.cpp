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
#include "GopDecoderUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

#include "ColorConvertKernels.cuh"

namespace fs = std::filesystem;

Pixel_Format PyNvGopDecoder::GetNativeFormat(const cudaVideoSurfaceFormat inputFormat) {
    switch (inputFormat) {
        case cudaVideoSurfaceFormat_NV12:
            return Pixel_Format_NV12;
        case cudaVideoSurfaceFormat_P016:
            return Pixel_Format_P016;
        case cudaVideoSurfaceFormat_YUV444:
            return Pixel_Format_YUV444;
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            return Pixel_Format_YUV444_16Bit;
        default:
            break;
    }
    return Pixel_Format_UNDEFINED;
}

int PyNvGopDecoder::GetRGBFromFrame(NvDecoder* decoder, const uint8_t* pFrame, uint8_t* pFrame_buffer,
                                    AVColorRange color_range, bool use_bgr_format, RGBFrame& rgb_frame) {
    Pixel_Format format = GetNativeFormat(decoder->GetOutputFormat());
    auto width = size_t(decoder->GetWidth());
    auto height = size_t(decoder->GetHeight());

    const std::vector<size_t> frame_shape{height, width, 3};
    const std::vector<size_t> frame_stride{width * 3, 3, 1};

    rgb_frame = RGBFrame(frame_shape, frame_stride, "|u1", reinterpret_cast<size_t>(decoder->GetStream()),
                         reinterpret_cast<CUdeviceptr>(pFrame_buffer), false, false);

    switch (format) {
        case Pixel_Format_NV12: {
            const CAIMemoryView y_view{{height, width, 1},
                                       {width, 1, 1},
                                       "|u1",
                                       reinterpret_cast<size_t>(decoder->GetStream()),
                                       reinterpret_cast<CUdeviceptr>(pFrame),
                                       false};
            const CAIMemoryView uv_view{{height / 2, width / 2, 2},
                                        {width / 2 * 2, 2, 1},
                                        "|u1",
                                        reinterpret_cast<size_t>(decoder->GetStream()),
                                        reinterpret_cast<CUdeviceptr>(pFrame + width * height),
                                        false};  // todo: data+width*height assumes both planes are
                                                 // contiguous. Actual NVENC allocation can have padding?

            nvtxRangePushA("Color convert");
            bool is_full_range = color_range == AVColorRange::AVCOL_RANGE_JPEG;
            if ((color_range != AVColorRange::AVCOL_RANGE_JPEG &&
                 color_range != AVColorRange::AVCOL_RANGE_MPEG)) {
                // LOG(WARNING) << "Color range is not supported with color range: " << color_range;
                is_full_range = false;
            }

            convert_nv12_to_rgb(y_view, uv_view, rgb_frame, is_full_range, use_bgr_format);
            nvtxRangePop();
        } break;
        default: {
            LOG(ERROR) << "Conversion to RGB/BGR only supported for videos in NV12-format";
            return -1;
        }
    }

    return 0;
}

int PyNvGopDecoder::GetYUVFromFrame(NvDecoder* decoder, const uint8_t* pFrame, uint8_t* pFrame_buffer,
                                    AVColorRange color_range, int64_t timestamp,
                                    DecodedFrameExt& decoded_frame) {
    decoded_frame.format = GetNativeFormat(decoder->GetOutputFormat());
    auto width = size_t(decoder->GetWidth());
    auto height = size_t(decoder->GetHeight());
    decoded_frame.timestamp = timestamp;
    decoded_frame.SetColorRange(color_range);

    // Copy the decode frames from device
    CUDA_DRVAPI_CALL(cuMemcpyDtoD((CUdeviceptr)pFrame_buffer, (CUdeviceptr)pFrame, decoder->GetFrameSize()));

    switch (decoded_frame.format) {
        case Pixel_Format_NV12: {
            decoded_frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                        {width, 1, 1},
                                                        "|u1",
                                                        reinterpret_cast<size_t>(decoder->GetStream()),
                                                        (CUdeviceptr)(pFrame_buffer),
                                                        false});
            decoded_frame.views.push_back(
                CAIMemoryView{{height / 2, width / 2, 2},
                              {width / 2 * 2, 2, 1},
                              "|u1",
                              reinterpret_cast<size_t>(decoder->GetStream()),
                              (CUdeviceptr)(pFrame_buffer + width * height),
                              false});  // todo: data+width*height assumes both planes are
                                        // contiguous. Actual NVENC allocation can have padding?
            // Load DLPack Tensor
            std::vector<size_t> shape{(size_t)(height * 1.5), width};
            std::vector<size_t> stride{size_t(width), 1};
            int returntype = decoded_frame.extBuf->LoadDLPack(shape, stride, "|u1",
                                                              reinterpret_cast<size_t>(decoder->GetStream()),
                                                              (CUdeviceptr)(pFrame_buffer), false);
        } break;
        case Pixel_Format_P016: {
            decoded_frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                        {width, 1, 1},
                                                        "|u2",
                                                        reinterpret_cast<size_t>(decoder->GetStream()),
                                                        (CUdeviceptr)(pFrame_buffer),
                                                        false});
            decoded_frame.views.push_back(
                CAIMemoryView{{height / 2, width / 2, 2},
                              {width / 2 * 2, 2, 1},
                              "|u2",
                              reinterpret_cast<size_t>(decoder->GetStream()),
                              (CUdeviceptr)(pFrame_buffer + 2 * (width * height)),
                              false});  // todo: data+width*height assumes both planes are
                                        // contiguous. Actual NVENC allocation can have padding?
        } break;
        case Pixel_Format_YUV444: {
            decoded_frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                        {width, 1, 1},
                                                        "|u1",
                                                        reinterpret_cast<size_t>(decoder->GetStream()),
                                                        (CUdeviceptr)(pFrame_buffer),
                                                        false});
            decoded_frame.views.push_back(
                CAIMemoryView{{height, width, 1},
                              {width, 1, 1},
                              "|u1",
                              reinterpret_cast<size_t>(decoder->GetStream()),
                              (CUdeviceptr)(pFrame_buffer + width * height),
                              false});  // todo: data+width*height assumes both planes are
                                        // contiguous. Actual NVENC allocation can have padding?
        } break;
        case Pixel_Format_YUV444_16Bit: {
            decoded_frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                        {width, 1, 1},
                                                        "|u2",
                                                        reinterpret_cast<size_t>(decoder->GetStream()),
                                                        (CUdeviceptr)(pFrame_buffer),
                                                        false});
            decoded_frame.views.push_back(
                CAIMemoryView{{height, width, 1},
                              {width, 1, 1},
                              "|u2",
                              reinterpret_cast<size_t>(decoder->GetStream()),
                              (CUdeviceptr)(pFrame_buffer + 2 * (width * height)),
                              false});  // todo: data+width*height assumes both planes are
                                        // contiguous. Actual NVENC allocation can have padding?
            decoded_frame.views.push_back(
                CAIMemoryView{{height, width, 1},
                              {width, 1, 1},
                              "|u2",
                              reinterpret_cast<size_t>(decoder->GetStream()),
                              (CUdeviceptr)(pFrame_buffer + 4 * (width * height)),
                              false});  // todo: data+width*height assumes both planes are
                                        // contiguous. Actual NVENC allocation can have padding?
        } break;
        default: {
            LOG(ERROR) << "Unsupported pixel format for DecodedFrameExt creation";
            return -1;
        }
    }

    return 0;
}

/*The video packet is ordered in decoding order, For example

  while (av_read_frame(pFormatContext, pPacket) >= 0)
  {
    // if it's the video stream
    if (pPacket->stream_index == video_stream_index) {
     //https://ffmpeg.org/doxygen/trunk/structAVPacket.html
      logging("pts %" PRId64, pPacket->pts);
      logging("dts %" PRId64, pPacket->dts);
    }
    av_packet_unref(pPacket);
  }

  (...)
LOG: AVPacket->pts 0
LOG:         ->dts -512
LOG: AVPacket->pts 1024
LOG:         ->dts -256
LOG: AVPacket->pts 512
LOG:         ->dts 0

av_seek_frame with mode AVSEEK_FLAG_BACKWARD will seek to the previous keyframe
(in decoding order) timestamp of av_seek_frame is pts (presenting order)
*/

void PyNvGopDecoder::DemuxGopProc(PyNvGopDemuxer* demuxer,
                                  ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                                  std::vector<int> sorted_frame_ids, std::vector<int>& first_frame_ids,
                                  std::vector<int>& gop_lens,
                                  std::vector<std::unique_ptr<uint8_t[]>>& packet_array, bool use_dd) {
    std::stringstream ss;
    ss << "DemuxGopProc_Thread" << std::this_thread::get_id();
    nvtxRangePushA(ss.str().c_str());

    int ret = 0;

    if (first_frame_ids.size() != gop_lens.size()) {
        LOG(ERROR) << "first_frame_ids and gop_lens must have the same length" << gop_lens.size() << " "
                   << first_frame_ids.size();
        packet_queue->push_back(std::make_tuple(nullptr, -1, 0));
        return;
    }

    bool seek_with_no_map = first_frame_ids.empty() ? true : false;
    if (seek_with_no_map) {
        gop_lens.reserve(sorted_frame_ids.size());
    }

    try {
        packet_array.reserve(std::accumulate(gop_lens.begin(), gop_lens.end(), 0));
        int nVideoBytes = 0;
        uint8_t* pVideo = NULL;

        for (int i = 0; i < sorted_frame_ids.size(); ++i) {
            bool seeking = true;
            int frame_id_out = 0;
            int key_frame_id = 0;
            int next_key_frame_id = INT_MAX;

            bool bRef = true;
            int flags;
            bool find_next_key_frame = false;
            int gop_len = 0;
            do {
                if (seeking) {
                    if (seek_with_no_map) {
                        ret = demuxer->SeekGopFirstFrameNoMap(&pVideo, &nVideoBytes, sorted_frame_ids[i],
                                                              frame_id_out);
                    } else {
                        ret = demuxer->Seek(&pVideo, &nVideoBytes, first_frame_ids[i], frame_id_out);
                    }
                    if (!ret) {
                        throw std::out_of_range("[Error] Seeking for frame_id:" +
                                                std::to_string(first_frame_ids[i]));
                    }
                    seeking = false;
                    key_frame_id = frame_id_out;
                    if (seek_with_no_map) {
                        first_frame_ids.push_back(key_frame_id);
                    }
                } else {
                    auto ret = demuxer->Demux(&pVideo, &nVideoBytes, frame_id_out, &flags, &bRef);
                    if (nVideoBytes) {
                        if (!ret) {
                            throw std::out_of_range("[Error] Demuxing");
                        }
                    } else {
                        continue;
                    }
                    bool find_key_frame = iskeyFrame(demuxer->GetDemuxer()->GetVideoCodec(), pVideo, flags);
                    find_next_key_frame |= find_key_frame;
                    if (find_key_frame) next_key_frame_id = frame_id_out;
                }

                if (frame_id_out < next_key_frame_id)
                    gop_len++;
                else if (frame_id_out > next_key_frame_id)
                    break;

                auto it = std::find(sorted_frame_ids.begin(), sorted_frame_ids.end(), frame_id_out);

                if (use_dd && frame_id_out >= key_frame_id) {
                    /*
           * if use dumuxer-decoder seperately, we need to push all frames in the GOP to the queue, 
           * and display none of them.
           */
                    auto packet_buffer = std::make_unique<uint8_t[]>(nVideoBytes);
                    memcpy(packet_buffer.get(), pVideo, nVideoBytes);
                    // SavePacketBufferToFile(packet_buffer.get(), nVideoBytes, frame_id_out);
                    packet_array.push_back(std::move(packet_buffer));
                    packet_queue->push_back(
                        std::make_tuple(packet_array.back().get(), nVideoBytes, frame_id_out * 2));
#ifdef IS_DEBUG_BUILD
                    std::cout << "Push GOP frame: " << frame_id_out
                              << " into queue with timestamp: " << frame_id_out * 2 << std::endl;
#endif
                } else if (it != sorted_frame_ids.end()) {
                    auto packet_buffer = std::make_unique<uint8_t[]>(nVideoBytes);
                    memcpy(packet_buffer.get(), pVideo, nVideoBytes);
                    // SavePacketBufferToFile(packet_buffer.get(), nVideoBytes, frame_id_out);
                    packet_array.push_back(std::move(packet_buffer));
                    if (use_dd) {  // if use_dd, don't need to display any frame
                        packet_queue->push_back(
                            std::make_tuple(packet_array.back().get(), nVideoBytes, frame_id_out * 2));
                    } else {
                        packet_queue->push_back(
                            std::make_tuple(packet_array.back().get(), nVideoBytes, frame_id_out * 2 + 1));
                    }
                    sorted_frame_ids.erase(it);
#ifdef IS_DEBUG_BUILD
                    std::cout << "Push target frame: " << frame_id_out
                              << " into queue with timestamp: " << frame_id_out * 2 + 1 << std::endl;
#endif
                } else if (bRef && frame_id_out >= key_frame_id) {
                    /*
           * 0(previous key) ... 250(current key) 248* 249 251 252 ...
           * frame 248 belongs to the previous GOP [0:250), but it's behind the current key frame 250
           * so we only need to push the frame with frameid > 250
           */
                    auto packet_buffer = std::make_unique<uint8_t[]>(nVideoBytes);
                    memcpy(packet_buffer.get(), pVideo, nVideoBytes);
                    // SavePacketBufferToFile(packet_buffer.get(), nVideoBytes, frame_id_out);
                    packet_array.push_back(std::move(packet_buffer));
                    packet_queue->push_back(
                        std::make_tuple(packet_array.back().get(), nVideoBytes, frame_id_out * 2));
#ifdef IS_DEBUG_BUILD
                    std::cout << "Push bRef frame: " << frame_id_out
                              << " into queue with timestamp: " << frame_id_out * 2 << std::endl;
#endif
                } else {
                    /*
           * we don't need to push all the packets to the queue,
           */
                    continue;
                }
            } while (nVideoBytes);
            packet_queue->push_back(std::make_tuple(nullptr, 0, 0));
            if (seek_with_no_map) gop_lens.push_back(gop_len);
        }
        packet_queue->push_back(std::make_tuple(nullptr, -1, 0));
    } catch (const std::exception& e) {
        packet_queue->push_back(std::make_tuple(nullptr, -1, 0));
        LOG(ERROR) << "DemuxGopProc failed: " << e.what();
    }
    nvtxRangePop();  //DemuxGopProc
}

template <typename OutputFrame>
void PyNvGopDecoder::DecProc(AVColorRange color_range, NvDecoder* decoder,
                             std::vector<OutputFrame>& output_frames, std::vector<uint8_t*> p_frames,
                             ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                             const std::vector<int> sorted_frame_ids, bool use_bgr_format,
                             const std::string& filename, LastDecodedFrameInfo& last_decoded_frame_info) {
    std::stringstream ss;
    ss << "DecProc_Thread: fid[" << sorted_frame_ids[0] << "], " << std::this_thread::get_id();
    nvtxRangePushA(ss.str().c_str());

    ck(cuCtxSetCurrent(decoder->GetContext()));

    // flush old decoder
    if (last_decoded_frame_info.filename == "") {
        // LOG(INFO) << "flush old decoder" << filename << " " << sorted_frame_ids[0];
        decoder->Decode(nullptr, 0, 0);
    }

    int nVideoBytes = 0, nFrameReturned = 0, frame_idx;
    uint8_t *pVideo = nullptr, *pFrame = nullptr;
    auto frame_id_iter = sorted_frame_ids.begin();
    auto pFrame_iter = p_frames.begin();
    int packet_id = 0;

    do {
        std::tuple<uint8_t*, int, int> packet_to_decode = packet_queue->pop_front();
        packet_id++;

        pVideo = std::get<0>(packet_to_decode);
        nVideoBytes = std::get<1>(packet_to_decode);
        frame_idx = std::get<2>(packet_to_decode);

        nFrameReturned = decoder->Decode(pVideo, nVideoBytes, 2, frame_idx);
        // LOG(INFO) << "Decode!!!, frame_idx: " << frame_idx << " nVideoBytes: " << nVideoBytes << " nFrameReturned: " << nFrameReturned << " target frame_id: " << *frame_id_iter;
        for (int i = 0; i < nFrameReturned; ++i) {
            int64_t timestamp = 0;
            pFrame = decoder->GetFrame(&timestamp);
            // LOG(INFO) << "    after get frame, frame_idx: " << frame_idx << " timestamp: " << timestamp;
            if (timestamp % 2 || timestamp / 2 == *frame_id_iter) {
                OutputFrame output_frame;
                if constexpr (std::is_same_v<OutputFrame, RGBFrame>) {
                    int st = PyNvGopDecoder::GetRGBFromFrame(decoder, pFrame, *pFrame_iter, color_range,
                                                             use_bgr_format, output_frame);
                    if (st) {
                        throw std::runtime_error("[ERROR] Failed to convert frame to RGB for file: " +
                                                 filename);
                    }
                } else {
                    int st = PyNvGopDecoder::GetYUVFromFrame(decoder, pFrame, *pFrame_iter, color_range,
                                                             timestamp, output_frame);
                    if (st) {
                        throw std::runtime_error("[ERROR] Failed to convert frame to YUV for file: " +
                                                 filename);
                    }
                }
                output_frames.push_back(std::move(output_frame));
                ++frame_id_iter;
                ++pFrame_iter;

                if (nFrameReturned == 1) {
                    last_decoded_frame_info.filename = filename;
                    last_decoded_frame_info.frame_id = timestamp / 2;
                } else {
                    if (nFrameReturned > 1) {
                        //TODO: TRICK, I don't know how to deal with this case
                        reset_last_decoded_frame_info(last_decoded_frame_info);
                    }
                }
            }
        }
        if (frame_id_iter == sorted_frame_ids.end()) {
            // nFrameReturned = decoder->Decode(nullptr, 0, 0);
            break;
        }
    } while ((nVideoBytes + 1));
    last_decoded_frame_info.packet_id += packet_id;

    // Frame count validation
    if (output_frames.size() != sorted_frame_ids.size()) {
        LOG(ERROR) << "number of decoded rgb frames: " << std::to_string(output_frames.size())
                   << " is different with number of frame id:" << std::to_string(sorted_frame_ids.size());
    }

    nvtxRangePop();
}

void PyNvGopDecoder::CreateDemuxer(std::unique_ptr<PyNvGopDemuxer>& demuxer, const std::string& filename,
                                   const FastStreamInfo* fastStreamInfo) {
    if (fastStreamInfo) {
        demuxer.reset(new PyNvGopDemuxer(filename.c_str(), fastStreamInfo));
    } else {
        demuxer.reset(new PyNvGopDemuxer(filename.c_str()));
    }
}

int PyNvGopDecoder::processFrameIds(const std::vector<std::string>& filepaths,
                                    const std::vector<int>& frame_ids,
                                    std::map<std::string, std::vector<int>>& fileFrameIds,
                                    std::map<std::string, std::vector<int>>& fileInverseFrameIds,
                                    std::vector<std::string>& fileList) {
    nvtxRangePushA("Process Frame IDs");

    const size_t total_frames = frame_ids.size();
    for (int idx = 0; idx < total_frames; ++idx) {
        if (fileFrameIds.count(filepaths[idx]) == 0) {
            fileList.push_back(filepaths[idx]);
        }
        fileFrameIds[filepaths[idx]].push_back(frame_ids[idx]);
        fileInverseFrameIds[filepaths[idx]].push_back(idx);
    }

    const size_t num_of_files = fileList.size();
    if (num_of_files > this->max_num_files) {
        LOG(ERROR) << "[ERROR] number of files are larger than maximum number of files";
        nvtxRangePop();  // Process Frame IDs
        return -1;
    }

    nvtxRangePop();  // Process Frame IDs

    return 0;
}

int PyNvGopDecoder::InitializeDemuxers(const std::vector<std::string>& filepaths,
                                       std::vector<std::unique_ptr<PyNvGopDemuxer>>& demuxers,
                                       const FastStreamInfo* fastStreamInfos) {
    nvtxRangePushA("Initialize Demuxers");

    int num_of_files = filepaths.size();
    demuxers.resize(num_of_files);

#ifdef PROCESS_SYNC
    for (int i = 0; i < num_of_files; ++i) {
        nvtxRangePushA((std::string("Demuxer creation : ") + std::to_string(i)).c_str());
        if (fastStreamInfos) {
            CreateDemuxer(demuxers[i], filepaths[i], fastStreamInfos + i);
        } else {
            CreateDemuxer(demuxers[i], filepaths[i], nullptr);
        }
        if (!demuxers[i]->IsValid()) {
            LOG(ERROR) << "create demuxer failed with video files: " << filepaths[i];
            nvtxRangePop();  // Demuxer creation
            nvtxRangePop();  // Initialize Demuxers
            return -1;
        }
        nvtxRangePop();  // Demuxer creation
    }
#endif

#ifndef PROCESS_SYNC
    for (int i = 0; i < num_of_files; ++i) {
        nvtxRangePushA((std::string("Demuxer creation thread start: ") + std::to_string(i)).c_str());
        demux_runners[i].join();
        if (fastStreamInfos) {
            demux_runners[i].start(PyNvGopDecoder::CreateDemuxer, std::ref(demuxers[i]), filepaths[i],
                                   fastStreamInfos + i);
        } else {
            demux_runners[i].start(PyNvGopDecoder::CreateDemuxer, std::ref(demuxers[i]), filepaths[i],
                                   nullptr);
        }
        nvtxRangePop();  // Demuxer creation thread start
    }

    for (int i = 0; i < num_of_files; ++i) {
        nvtxRangePushA((std::string("Demuxer creation thread join: ") + std::to_string(i)).c_str());
        demux_runners[i].join();
        nvtxRangePop();
        if (!demuxers[i]->IsValid()) {
            for (int index = i; index < num_of_files; index++) {
                demux_runners[index].join();
            }
            LOG(ERROR) << "create demuxer failed with video files " << filepaths[i];
            nvtxRangePop();  // Demuxer creation thread join
            nvtxRangePop();  // Initialize Demuxers
            return -1;
        }
    }
#endif

    // check decoder and demuxer must have the same resolution
    for (int i = 0; i < num_of_files; ++i) {
        if (i < vdec.size() && (demuxers[i]->GetHeight() != vdec[i]->GetHeight() ||
                                demuxers[i]->GetWidth() != vdec[i]->GetWidth())) {
            if (vdec[i]->GetHeight() != 0 || vdec[i]->GetWidth() != 0) {  //skip if decoder is not initialized
                LOG(ERROR) << "decoder and demuxer have different resolution: " << filepaths[i]
                           << ", demuxer (HxW): " << demuxers[i]->GetHeight() << "x"
                           << demuxers[i]->GetWidth() << ", decoder (HxW): " << vdec[i]->GetHeight() << "x"
                           << vdec[i]->GetWidth();
                nvtxRangePop();  // Initialize Demuxers
                return -1;
            }
        }
    }

    nvtxRangePop();  // Initialize Demuxers
    return 0;
}

int PyNvGopDecoder::InitGpuMemPool(const std::vector<int>& heights, const std::vector<int>& widths,
                                   const std::vector<int>& frame_sizes, bool convert_to_rgb) {
    size_t needed_size = 0;
    int len = frame_sizes.size();
    if (len == 0) {
        len = heights.size();
    }

    for (int i = 0; i < len; ++i) {
        if (convert_to_rgb) {
            needed_size += widths[i] * heights[i] * 3;
        } else {
            needed_size += frame_sizes[i];
        }
    }

    // Temporarily push context for GPU memory allocation
    ck(cuCtxPushCurrent(this->cu_context));
    this->gpu_mem_pool.EnsureSizeAndSoftReset(needed_size, false);
    ck(cuCtxPopCurrent(NULL));

    return 0;
}

int PyNvGopDecoder::InitializeDecoders(const std::vector<int>& codec_ids) {
    nvtxRangePushA("Initialize Decoders");

    const int num_of_files = static_cast<int>(codec_ids.size());

    ensureCudaContextInitialized();
    ensureDecodeRunnersInitialized();

    // Temporarily push context for decoder creation
    ck(cuCtxPushCurrent(this->cu_context));

    // Create decoders only
    for (int i = 0; i < num_of_files; ++i) {
        nvtxRangePushA("Decoder creation");
        if (i >= this->vdec.size()) {
            std::unique_ptr<NvDecoder> dec(new NvDecoder(this->cu_stream, this->cu_context, true,
                                                         static_cast<cudaVideoCodec>(codec_ids[i]), false,
                                                         false, false));
            this->vdec.push_back(std::move(dec));
        }
        nvtxRangePop();  //Decoder creation
    }

    ck(cuCtxPopCurrent(NULL));

    nvtxRangePop();  //Initialize Decoders
    return 0;
}

int PyNvGopDecoder::GetFileFrameBuffers(const std::vector<int>* widths, const std::vector<int>* heights,
                                        const std::vector<int>* frame_sizes, bool convert_to_rgb,
                                        std::vector<std::vector<uint8_t*>>& per_file_frame_buffers) {
    nvtxRangePushA("Get File Frame Buffers");

    const int num_of_files = static_cast<int>(widths->size());
    per_file_frame_buffers.resize(num_of_files);

    // Allocate memory for frames
    for (int i = 0; i < num_of_files; ++i) {
        nvtxRangePushA("Frame memory allocation");
        uint8_t* pFrame;
        if (convert_to_rgb) {
            pFrame =
                reinterpret_cast<uint8_t*>(this->gpu_mem_pool.AddElement(widths->at(i) * heights->at(i) * 3));
        } else {
            pFrame = reinterpret_cast<uint8_t*>(this->gpu_mem_pool.AddElement(frame_sizes->at(i)));
        }
        per_file_frame_buffers[i].push_back(pFrame);
        nvtxRangePop();  //Frame memory allocation
    }

    nvtxRangePop();  //Get File Frame Buffers
    return 0;
}

int PyNvGopDecoder::ExtractAndProcessGopInfo(const std::unique_ptr<PyNvGopDemuxer>& demuxer,
                                             std::vector<int>& sorted_frame_ids,
                                             std::vector<int>& first_frame_ids,
                                             std::vector<int>& gop_length) {
    nvtxRangePushA("ExtractAndProcessGopInfo_VFR");

    // This function should only be called for VFR videos
    std::vector<int> gop_start_id_list;
    std::map<int, int64_t> frame2pts;
    std::map<int64_t, int> pts2frame;

    try {
        gop_start_id_list = parse_gop_start_idx(demuxer->GetDemuxer(), frame2pts, pts2frame, true);
        demuxer->set_pts_frameid_mapping(std::move(frame2pts), std::move(pts2frame));
        gop_length = parse_gop_length(gop_start_id_list, sorted_frame_ids, first_frame_ids);

        if (gop_length.size() == 0) {
            LOG(ERROR) << "[ERROR] Wrong gop_length for VFR video" << std::endl;
            nvtxRangePop();  // ExtractAndProcessGopInfo_VFR
            return -1;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] VFR GOP extraction failed: " << e.what() << std::endl;
        nvtxRangePop();  // ExtractAndProcessGopInfo_VFR
        return -1;
    }

    nvtxRangePop();  // ExtractAndProcessGopInfo_VFR
    return 0;
}

// Explicit template instantiations for DecProc
template void PyNvGopDecoder::DecProc<RGBFrame>(AVColorRange color_range, NvDecoder* decoder,
                                                std::vector<RGBFrame>& output_frames,
                                                std::vector<uint8_t*> p_frames,
                                                ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                                                const std::vector<int> sorted_frame_ids, bool use_bgr_format,
                                                const std::string& filename,
                                                LastDecodedFrameInfo& last_decoded_frame_info);

template void PyNvGopDecoder::DecProc<DecodedFrameExt>(
    AVColorRange color_range, NvDecoder* decoder, std::vector<DecodedFrameExt>& output_frames,
    std::vector<uint8_t*> p_frames, ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
    const std::vector<int> sorted_frame_ids, bool use_bgr_format, const std::string& filename,
    LastDecodedFrameInfo& last_decoded_frame_info);

void PyNvGopDecoder::ReleaseMemPools() {
    // Temporarily push context for GPU memory release
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));
    }
    // Release only the GPU memory pool
    gpu_mem_pool.HardRelease();
    if (this->cu_context) {
        ck(cuCtxPopCurrent(NULL));
    }
}

void PyNvGopDecoder::ReleaseDecoder() {
    // Clear all decoder instances to release their GPU memory
    // (NvDecoder instances allocate GPU memory for frame buffers)
    for (auto& decoder : vdec) {
        decoder.reset();
    }
    vdec.clear();
}
