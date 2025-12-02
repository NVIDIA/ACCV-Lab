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

#include "PyNvVideoReader.hpp"
#include "GopDecoderUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

#include "ColorConvertKernels.cuh"

#define MAX_SIZE 2000

namespace fs = std::filesystem;

/* If a frame is a I-frame and key_frame in the same time, then the frame is the
 start of a new GOP The keyframe is the frame which has flag AV_FRAME_FLAG_KEY
 (1 << 1), if open_gop is false, each keyframe is a IDR picture
 For video with open_gop is true, to parse NAL unit to get IDR picture id, The
keyframe is the frame which has flag AV_FRAME_FLAG_KEY is a Recovery Point. The
recovery point SEI message assists a decoder in determining when the decoding
process will produce acceptable pictures for display after the decoder initiates
random access or after the encoder indicates a broken link in the coded video
sequence.*/

void PyNvVideoReader::parse_keyframe_idx(std::vector<int>& key_frame_ids, std::map<int, int64_t>& frame2pts,
                                         std::map<int64_t, int>& pts2frame) {
    std::vector<std::pair<int64_t, bool>> pts_keyFrame_pair;
    int nVideoBytes = 0, flags = 0;
    uint8_t* pVideo = NULL;
    int64_t timestamp;
    int frame_cnt = 0;
    std::unique_ptr<FFmpegDemuxer> cur_demuxer(new FFmpegDemuxer(this->filename.c_str()));

    bool bPS = false;
    do {
        auto ret = cur_demuxer->Demux(&pVideo, &nVideoBytes, &timestamp, &flags);
        ++frame_cnt;

        if (nVideoBytes) {
            if (!ret) {
                throw std::invalid_argument("[ERROR] Demux error");
            }
            bool is_key_frame = iskeyFrame(cur_demuxer->GetVideoCodec(), pVideo, flags);
            pts_keyFrame_pair.emplace_back(timestamp, is_key_frame);
        }
    } while (nVideoBytes);

    // Sort the combined vector
    std::sort(pts_keyFrame_pair.begin(), pts_keyFrame_pair.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    int current_frame_idx = 0;
    for (current_frame_idx = 0; current_frame_idx < pts_keyFrame_pair.size(); ++current_frame_idx) {
        auto current_timestamp = pts_keyFrame_pair[current_frame_idx].first;
        auto isKeyFrame = pts_keyFrame_pair[current_frame_idx].second;

        if (cur_demuxer->IsVFR()) {
            frame2pts[current_frame_idx] = current_timestamp;
            pts2frame[current_timestamp] = current_frame_idx;
        }

        if (isKeyFrame) {
            key_frame_ids.push_back(current_frame_idx);
        }
    }
    key_frame_ids.push_back(current_frame_idx);

    if (key_frame_ids.size() <= 0) {
        throw std::out_of_range("[ERROR] The video must have at least one GOP");
    }
}

Pixel_Format PyNvVideoReader::GetNativeFormat(const cudaVideoSurfaceFormat inputFormat) {
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

PyNvVideoReader::PyNvVideoReader(const std::string filename, int iGpu, CUcontext cu_context,
                                 CUstream cu_stream, bool bSuppressNoColorRangeWarning)
    : filename(filename), gpu_id(iGpu), suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning) {
#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvVideoReader object" << std::endl;
#endif
    this->destroy_context = false;

    // To do, we can reuse current context, we can check its func
    if (cu_context == nullptr) {
        CUresult res = cuCtxGetCurrent(&this->cu_context);
        if (res == CUDA_ERROR_NOT_INITIALIZED) {
            ck(cuInit(0));
            int nGpu = 0;
            ck(cuDeviceGetCount(&nGpu));
            if (iGpu < 0 || iGpu >= nGpu) {
                std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]"
                          << std::endl;
            }
        }
    } else {
        this->cu_context = cu_context;
        this->cu_stream = cu_stream;
        this->owner_stream = false;
    }
    if (!this->cu_context) {
        std::cout << "Get and push primary context" << std::endl;
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, this->gpu_id));
        ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
        this->destroy_context = true;
        // Temporarily push context for stream creation, then immediately pop.
        ck(cuCtxPushCurrent(this->cu_context));
        if (cu_stream == nullptr) {
            ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
            this->owner_stream = true;
        } else {
            this->cu_stream = cu_stream;
            this->owner_stream = false;
        }
        ck(cuCtxPopCurrent(NULL));
    }
    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");
    }
    this->packet_queue.reset(new ConcurrentQueue<std::tuple<uint8_t*, int, int>>);
    this->packet_queue->setSize(MAX_SIZE);
    nvtxRangePushA("Reset Demuxer");
    this->demuxer.reset(new PyNvGopDemuxer(filename.c_str()));
    nvtxRangePop();
    nvtxRangePushA("Reset Decoder");
    // Temporarily push context for NvDecoder creation
    ck(cuCtxPushCurrent(this->cu_context));
    // Check the param for NvDecoder [TODO]
    this->decoder.reset(
        new NvDecoder(this->cu_stream, this->cu_context, true, demuxer->GetNvCodecId(), false, false, false));
    ck(cuCtxPopCurrent(NULL));
    nvtxRangePop();
    std::vector<int> key_frame_ids;
    std::map<int, int64_t> frame2pts;
    std::map<int64_t, int> pts2frame;

    nvtxRangePushA("Parse Keyframe Index");
    this->parse_keyframe_idx(key_frame_ids, frame2pts, pts2frame);
    nvtxRangePop();

    this->cur_frame_ = key_frame_ids[0] - 1;
    this->next_keyframe_ = key_frame_ids[0];

    this->demuxer->set_key_frame_ids(std::move(key_frame_ids));
    this->demuxer->set_pts_frameid_mapping(std::move(frame2pts), std::move(pts2frame));
}

PyNvVideoReader::~PyNvVideoReader() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvVideoReader object" << std::endl;
#endif
    // Maybe this part can be replaced by ReleasePacket
    this->releasePacketArray();

    // Temporarily push context for GPU resource cleanup.
    // This ensures the destructor works correctly on any thread.
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));

        // Explicitly release GPU memory pool before automatic member destruction
        // (gpu_mem_pool destructor would run after this, but HardRelease is idempotent)
        gpu_mem_pool.HardRelease();

        if (this->owner_stream && this->cu_stream) {
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

void PyNvVideoReader::ReleaseMemPools() {
    // Only release GPU memory pool, preserve decoder state (cur_frame_, packet_queue, etc.)
    // This allows efficient forward decoding after memory release.
    //
    // Note: After release, requesting frame_id <= cur_frame_ will trigger re-decode
    // because the decoded frame data is no longer available in the memory pool.

    // Temporarily push context for GPU memory release
    if (this->cu_context) {
        ck(cuCtxPushCurrent(this->cu_context));
    }
    gpu_mem_pool.HardRelease();
    if (this->cu_context) {
        ck(cuCtxPopCurrent(NULL));
    }
}

void PyNvVideoReader::releasePacketArray() {
    nvtxRangePushA("Release Packet Array");

    for (uint8_t* packet : this->packet_array) {
        if (packet) {
            delete[] packet;
        }
    }

    this->packet_array.clear();
    this->packet_queue->clear();
    nvtxRangePop();
}

void PyNvVideoReader::startNewGop() {
    this->releasePacketArray();
    int64_t timestamp = 0;

    auto nFrameReturned =
        this->decoder->Decode(NULL, 0, 2, 0);  // We may also need to clear the cached frames? Check it
    for (int i = 0; i < this->return_frames_ + nFrameReturned; i++) {
        auto pFrame = this->decoder->GetFrame(&timestamp);
    }
    this->return_frames_ = 0;
}

void PyNvVideoReader::startNextGop() { this->releasePacketArray(); }

void PyNvVideoReader::ReplaceWithFile(const std::string filename) {
    this->releasePacketArray();
    nvtxRangePushA("Reset Demuxer");
    this->demuxer.reset(new PyNvGopDemuxer(filename.c_str()));
    nvtxRangePop();
    this->filename = filename;
    if (this->demuxer->IsValid()) {
        std::vector<int> key_frame_ids;
        std::map<int, int64_t> frame2pts;
        std::map<int64_t, int> pts2frame;

        nvtxRangePushA("Parse Keyframe Index");
        this->parse_keyframe_idx(key_frame_ids, frame2pts, pts2frame);
        nvtxRangePop();

        this->cur_frame_ = key_frame_ids[0] - 1;
        this->next_keyframe_ = key_frame_ids[0];

        this->demuxer->set_key_frame_ids(std::move(key_frame_ids));
        this->demuxer->set_pts_frameid_mapping(std::move(frame2pts), std::move(pts2frame));
    }
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

void PyNvVideoReader::DemuxGopProc(PyNvGopDemuxer* demuxer,
                                   ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                                   const int key_frame_ids, const int gop_len,
                                   std::vector<uint8_t*>& packet_array, bool seeking) {
    int nVideoBytes = 0, frame_id_out = 0;
    uint8_t* pVideo = NULL;
    int flags;
    bool bRef = true;

    nvtxRangePushA("Demux Gop Proc");
    // Handle special case where gop_len is 0
    if (gop_len == 0) {
        demuxGopProcZeroLen(demuxer, packet_queue, key_frame_ids, packet_array, seeking);
        nvtxRangePop();
        // throw std::invalid_argument("gop len is 0");
        return;
    }

    // Handle error case
    if (gop_len < 0) {
        nvtxRangePop();
        throw std::invalid_argument("[ERROR] Can not find a demux gop for frame: " +
                                    std::to_string(key_frame_ids));
    }

    // Process GOP with normal length
    try {
        packet_array.reserve(gop_len);
        int frame_cnt = 0;

        do {
            // Get next frame data
            if (seeking) {
                if (!demuxer->Seek(&pVideo, &nVideoBytes, key_frame_ids, frame_id_out)) {
                    nvtxRangePop();
                    throw std::out_of_range("[Error] Seeking for frame_id:" + std::to_string(key_frame_ids));
                }
                seeking = false;
            } else {
                if (!demuxer->Demux(&pVideo, &nVideoBytes, frame_id_out, &flags, &bRef)) {
                    if (nVideoBytes) {
                        nvtxRangePop();
                        throw std::out_of_range("[Error] Demuxing");
                    }
                    continue;
                }
            }
            ++frame_cnt;

            // Allocate and copy frame data
            nvtxRangePushA("Alloc Packet Array");
            uint8_t* packet_buffer = new uint8_t[nVideoBytes];
            nvtxRangePop();

            memcpy(packet_buffer, pVideo, nVideoBytes);
            packet_array.push_back(packet_buffer);

            // Queue the frame
            packet_queue->push_back(std::make_tuple(packet_buffer, nVideoBytes, frame_id_out));

#ifdef IS_DEBUG_BUILD
            std::cout << "Push demuxed frame: " << frame_id_out << " length: " << nVideoBytes
                      << " offset: " << static_cast<void*>(pVideo)
                      << " into queue with timestamp: " << frame_id_out << std::endl;
#endif
        } while (nVideoBytes && frame_cnt < gop_len);
    } catch (const std::exception& e) {
        nvtxRangePop();
        std::cerr << e.what() << std::endl;
    }
    nvtxRangePop();
}

// Helper function to handle gop_len == 0 case
void PyNvVideoReader::demuxGopProcZeroLen(PyNvGopDemuxer* demuxer,
                                          ConcurrentQueue<std::tuple<uint8_t*, int, int>>* packet_queue,
                                          const int key_frame_ids, std::vector<uint8_t*>& packet_array,
                                          bool seeking) {
    int nVideoBytes = 0, frame_id_out = 0;
    uint8_t* pVideo = NULL;
    int flags;
    bool bRef = true;

    do {
        // Get frame data
        if (seeking) {
            if (!demuxer->Seek(&pVideo, &nVideoBytes, key_frame_ids, frame_id_out)) {
                throw std::out_of_range("[Error] Seeking for frame_id:" + std::to_string(key_frame_ids));
            }
            seeking = false;
        } else {
            if (!demuxer->Demux(&pVideo, &nVideoBytes, frame_id_out, &flags, &bRef)) {
                if (nVideoBytes) {
                    throw std::out_of_range("[Error] Demuxing");
                }
                continue;
            }
        }

        // Allocate and copy frame data
        nvtxRangePushA("Alloc Packet Array");
        uint8_t* packet_buffer = new uint8_t[nVideoBytes];
        nvtxRangePop();

        memcpy(packet_buffer, pVideo, nVideoBytes);
        packet_array.push_back(packet_buffer);

        // Queue the frame
        packet_queue->push_back(std::make_tuple(packet_buffer, nVideoBytes, frame_id_out));

#ifdef IS_DEBUG_BUILD
        std::cout << "Push demuxed frame: " << frame_id_out << " length: " << nVideoBytes
                  << " offset: " << static_cast<void*>(pVideo)
                  << " into queue with timestamp: " << frame_id_out << std::endl;
#endif
    } while (nVideoBytes);

    packet_queue->push_back(std::make_tuple(nullptr, 0, 0));
}

std::vector<DecodedFrameExt> PyNvVideoReader::run(const std::vector<int> frame_ids) {
    std::vector<DecodedFrameExt> res;
    res.resize(frame_ids.size());
    try {
        for (int i = 0; i < frame_ids.size(); i++) {
            run_single_frame_internal(frame_ids[i], false, false, &res[i], nullptr);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return res;
}

DecodedFrameExt PyNvVideoReader::run_single(const int frame_id) {
    DecodedFrameExt res;
    try {
        run_single_frame_internal(frame_id, false, false, &res, nullptr);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return res;
}

std::vector<RGBFrame> PyNvVideoReader::run_rgb_out(const std::vector<int> frame_ids, bool as_bgr) {
    std::vector<RGBFrame> res;
    res.resize(frame_ids.size());
    try {
        for (int i = 0; i < frame_ids.size(); i++) {
            run_single_frame_internal(frame_ids[i], true, as_bgr, nullptr, &res[i]);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return res;
}

RGBFrame PyNvVideoReader::run_single_rgb_out(const int frame_id, bool as_bgr) {
    RGBFrame res;
    try {
        run_single_frame_internal(frame_id, true, as_bgr, nullptr, &res);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return res;
}

void PyNvVideoReader::run_single_frame_internal(const int frame_id, bool convert_to_rgb, bool use_bgr_format,
                                                DecodedFrameExt* out_if_no_color_conversion,
                                                RGBFrame* out_if_color_converted) {
    assert(!convert_to_rgb || (out_if_color_converted != nullptr && out_if_no_color_conversion == nullptr) &&
                                  "If color conversion is used, out_if_color_converted has to point "
                                  "to vector where to write the results, and "
                                  "out_if_no_color_conversion has to be nullptr");
    assert(convert_to_rgb || (out_if_color_converted == nullptr && out_if_no_color_conversion != nullptr) &&
                                 "If color conversion is not used, out_if_color_converted has to "
                                 "be nullptr, and out_if_no_color_conversion has to point to "
                                 "vector where to write the results");
    if (!this->demuxer->IsValid()) {
        throw std::invalid_argument("Invalid demuxer");
    }

    if (frame_id >= this->demuxer->getLastKeyFrameId()) {
        throw std::invalid_argument("a frame after the last keyframe is currently not supported");
    }

    if (frame_id < demuxer->getFirstKeyFrameId()) {
        throw std::invalid_argument("a frame before the first keyframe is currently not supported");
    }

    ck(cuCtxPushCurrent(this->cu_context));
    nvtxRangePushA("Decode Single Frame");

    size_t needed_size = 0;

    if (convert_to_rgb) {
        needed_size = this->demuxer->GetWidth() * this->demuxer->GetHeight() * 3;
    } else {
        needed_size = this->demuxer->GetFrameSize();
    }

    gpu_mem_pool.EnsureSizeAndSoftReset(needed_size, false);

    // Note that the following difference needs to be computed with signed
    // integers as the difference may be negative
    // TODO, the size of the allocated buffer for decoded Frames should be
    // dec->GetFrameSize()
    uint8_t *pReturnFrame = nullptr, *pFrame = nullptr;
    int64_t timestamp = 0;
    bool file_added_for_warning = false;

    if (convert_to_rgb) {
        pFrame = reinterpret_cast<uint8_t*>(
            this->gpu_mem_pool.AddElement(this->demuxer->GetWidth() * this->demuxer->GetHeight() * 3));
    } else {
        pFrame = reinterpret_cast<uint8_t*>(this->gpu_mem_pool.AddElement(this->demuxer->GetFrameSize()));
    }

    if (frame_id >= this->next_keyframe_) {
        auto cur_keyframe_ = demuxer->getKeyFrameId(frame_id);
        if (cur_keyframe_ == this->next_keyframe_ && cur_frame_ >= cur_keyframe_) {
            this->fetchNextGop();
        } else {
            this->cur_frame_ = cur_keyframe_ - 1;
            this->fetchNewGop(cur_keyframe_);
        }
    }
    // Use <= instead of < to handle:
    // 1. Requesting the same frame again (frame_id == cur_frame_): re-decode needed
    //    because memory pool may have been released or data may have been modified
    //    by user's in-place operations (e.g., PyTorch tensor += 1)
    // 2. Requesting a previous frame (frame_id < cur_frame_): re-decode needed
    if (frame_id <= this->cur_frame_) {
        auto cur_keyframe_ = demuxer->getKeyFrameId(frame_id);
        this->cur_frame_ = cur_keyframe_ - 1;

        this->fetchNewGop(cur_keyframe_);
    }

    while (this->cur_frame_ < frame_id) {
        // First try to process any pending decoded frames
        if (this->processDecodedFrames(frame_id, pFrame, pReturnFrame, convert_to_rgb, use_bgr_format,
                                       file_added_for_warning, out_if_color_converted,
                                       out_if_no_color_conversion)) {
            nvtxRangePop();
            ck(cuCtxPopCurrent(NULL));
            return;
        }

        if (this->packet_queue->empty()) {  // To handle the case that we need to demux&decode more
                                            // packets to get the desired frame_id
            fetchNextGop();
        }
        // Decode next packet from queue
        decodeNextPacket();

        // Process any newly decoded frames
        if (processDecodedFrames(frame_id, pFrame, pReturnFrame, convert_to_rgb, use_bgr_format,
                                 file_added_for_warning, out_if_color_converted,
                                 out_if_no_color_conversion)) {
            nvtxRangePop();
            ck(cuCtxPopCurrent(NULL));
            return;
        }
    }

    if (!suppress_no_color_range_given_warning) {
        if (file_added_for_warning) {
            std::cout << "WARNING: PyNVGopDecoder could not obtain color range for "
                         "the following files:\n";
            std::cout << "  " << this->filename << "\n";
            std::cout << "  --> Limited range (MPEG range) assumed for these files." << std::endl;
        }
    }

    ck(cuCtxPopCurrent(NULL));
    nvtxRangePop();
}

void PyNvVideoReader::fetchNextGop() {
    auto next_next_key_frame = demuxer->getNextKeyFrameId(this->next_keyframe_);
    auto gop_len = next_next_key_frame - this->next_keyframe_;

    if (gop_len == 0 && this->next_keyframe_ != demuxer->getLastKeyFrameId()) {
        std::cerr << "WARNING: PyNvVideoReader could not find a demux gop for frame: " << this->next_keyframe_
                  << " Just demux the whole left packets" << std::endl;
    }

    this->startNextGop();
    try {
        PyNvVideoReader::DemuxGopProc(this->demuxer.get(), this->packet_queue.get(), this->next_keyframe_,
                                      gop_len, this->packet_array, false);
    } catch (const std::exception& e) {
        LOG(ERROR) << "DemuxGopProc failed: " << e.what();
        throw std::runtime_error(e.what());
    }

    this->next_keyframe_ = next_next_key_frame;
}

void PyNvVideoReader::fetchNewGop(int cur_keyframe) {
    auto next_key_frame = demuxer->getNextKeyFrameId(cur_keyframe);
    auto gop_len = next_key_frame - cur_keyframe;
    this->startNewGop();
    try {
        PyNvVideoReader::DemuxGopProc(this->demuxer.get(), this->packet_queue.get(), cur_keyframe, gop_len,
                                      this->packet_array, true);
    } catch (const std::exception& e) {
        LOG(ERROR) << "DemuxGopProc failed: " << e.what();
        throw std::runtime_error(e.what());
    }
    this->next_keyframe_ = next_key_frame;
}

void PyNvVideoReader::decodeNextPacket() {
    uint8_t* pVideo = nullptr;
    int nVideoBytes = 0;
    int frame_idx = 0;

    std::tuple<uint8_t*, int, int> packet_to_decode = this->packet_queue->pop_front();
    pVideo = std::get<0>(packet_to_decode);
    nVideoBytes = std::get<1>(packet_to_decode);
    frame_idx = std::get<2>(packet_to_decode);
#ifdef IS_DEBUG_BUILD
    std::cout << "[Push to decoder] frame: " << frame_idx << std::endl;
#endif
    int nFrameReturned = decoder->Decode(pVideo, nVideoBytes, 2, frame_idx);
    this->return_frames_ += nFrameReturned;
}

bool PyNvVideoReader::processDecodedFrames(const int frame_id, uint8_t* pFrame, uint8_t* pReturnFrame,
                                           bool convert_to_rgb, bool use_bgr_format,
                                           bool& file_added_for_warning, RGBFrame* out_if_color_converted,
                                           DecodedFrameExt* out_if_no_color_conversion) {
    while (this->return_frames_) {
        int64_t timestamp;
        pReturnFrame = decoder->GetFrame(&timestamp);
#ifdef IS_DEBUG_BUILD
        std::cout << "Get " << timestamp << " -id frame" << std::endl;
#endif
        --this->return_frames_;
        ++this->cur_frame_;

        if (this->cur_frame_ == frame_id) {
#ifdef IS_DEBUG_BUILD
            std::cout << "Return " << frame_id << " -id frame" << std::endl;
#endif
            // assert(timestamp == frame_id);
            // TODO This line is important, but sometimes the timestamp returned by
            // nvdec is wrong
            if (convert_to_rgb) {
                *out_if_color_converted =
                    this->returnRGBFrame(pFrame, pReturnFrame, use_bgr_format, file_added_for_warning);
            } else {
                *out_if_no_color_conversion = this->returnYUVFrame(pFrame, pReturnFrame);
            }
            return true;
        }
    }
    return false;
}

DecodedFrameExt PyNvVideoReader::returnYUVFrame(void* pFrame_buffer, void* pFrame) {
    DecodedFrameExt frame;
    frame.format = GetNativeFormat(this->decoder->GetOutputFormat());
    auto width = size_t(this->decoder->GetWidth());
    auto height = size_t(this->decoder->GetHeight());
    // Currently no timestamp, any bad ?
    // frame.timestamp = timestamp;
    frame.SetColorRange(demuxer->GetColorRange());

    // Copy the decode frames from device
    CUDA_DRVAPI_CALL(
        cuMemcpyDtoD((CUdeviceptr)pFrame_buffer, (CUdeviceptr)pFrame, this->decoder->GetFrameSize()));

    switch (frame.format) {
        case Pixel_Format_NV12: {
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u1",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)pFrame_buffer,
                                                false});
            frame.views.push_back(CAIMemoryView{{height / 2, width / 2, 2},
                                                {width / 2 * 2, 2, 1},
                                                "|u1",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer + width * height),
                                                false});  // todo: data+width*height assumes both planes
                                                          // are Load DLPack Tensor
            std::vector<size_t> shape{(size_t)(height * 1.5), width};
            std::vector<size_t> stride{size_t(width), 1};
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1",
                                                      reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                      (CUdeviceptr)(pFrame_buffer), false);
        } break;
        case Pixel_Format_P016: {
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u2",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer),
                                                false});
            frame.views.push_back(CAIMemoryView{{height / 2, width / 2, 2},
                                                {width / 2 * 2, 2, 1},
                                                "|u2",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer + 2 * (width * height)),
                                                false});  // todo: data+width*height assumes both planes are
            // contiguous. Actual NVENC allocation can have padding?
        } break;
        case Pixel_Format_YUV444: {
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u1",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer),
                                                false});
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u1",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer + width * height),
                                                false});  // todo: data+width*height assumes both planes are
            // contiguous. Actual NVENC allocation can have padding?
        }
        case Pixel_Format_YUV444_16Bit: {
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u2",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer),
                                                false});
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u2",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer + 2 * (width * height)),
                                                false});  // todo: data+width*height assumes both planes are
            // contiguous. Actual NVENC allocation can have padding?
            frame.views.push_back(CAIMemoryView{{height, width, 1},
                                                {width, 1, 1},
                                                "|u2",
                                                reinterpret_cast<size_t>(this->decoder->GetStream()),
                                                (CUdeviceptr)(pFrame_buffer + 4 * (width * height)),
                                                false});  // todo: data+width*height assumes both planes are
            // contiguous. Actual NVENC allocation can have padding?
        }
    }
    return frame;
}

RGBFrame PyNvVideoReader::returnRGBFrame(void* pFrame_buffer, void* pFrame, bool use_bgr_format,
                                         bool& file_added_for_warning) {
    Pixel_Format format = GetNativeFormat(this->decoder->GetOutputFormat());
    auto width = size_t(this->decoder->GetWidth());
    auto height = size_t(this->decoder->GetHeight());

    const std::vector<size_t> frame_shape{height, width, 3};
    const std::vector<size_t> frame_stride{width * 3, 3, 1};
    RGBFrame rgb_frame(frame_shape, frame_stride, "|u1", reinterpret_cast<size_t>(decoder->GetStream()),
                       reinterpret_cast<CUdeviceptr>(pFrame_buffer), false, false);

    switch (format) {
        case Pixel_Format_NV12: {
            const CAIMemoryView y_view{{height, width, 1},
                                       {width, 1, 1},
                                       "|u1",
                                       reinterpret_cast<size_t>(this->decoder->GetStream()),
                                       reinterpret_cast<CUdeviceptr>(pFrame),
                                       false};
            const CAIMemoryView uv_view{{height / 2, width / 2, 2},
                                        {width / 2 * 2, 2, 1},
                                        "|u1",
                                        reinterpret_cast<size_t>(this->decoder->GetStream()),
                                        reinterpret_cast<CUdeviceptr>(pFrame + width * height),
                                        false};  // todo: data+width*height assumes both planes are
                                                 // contiguous. Actual NVENC allocation can have padding?
            const AVColorRange color_range = this->demuxer->GetColorRange();
            if (!file_added_for_warning && (color_range != AVColorRange::AVCOL_RANGE_JPEG &&
                                            color_range != AVColorRange::AVCOL_RANGE_MPEG)) {
                file_added_for_warning = true;
            }
            const bool is_full_range = color_range == AVColorRange::AVCOL_RANGE_JPEG;
            convert_nv12_to_rgb(y_view, uv_view, rgb_frame, is_full_range, use_bgr_format);
        } break;
        default: {
            throw std::invalid_argument(
                "[ERROR] Conversion to RGB/BGR only supported "
                "for videos in NV12-format");
        }
    }
    return rgb_frame;
}
void Init_PyNvVideoReader(py::module& m) {
    ExternalBuffer::Export(m);
    DecodedFrame::Export(m);
    CAIMemoryView::Export(m);
    DecodedFrameExt::Export(m);
    RGBFrame::Export(m);
    // Create a factory function similar to CreateGopDecoder
    m.def(
        "CreateVideoReader",
        [](const std::string& filename, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvVideoReader>(filename, iGpu, suppressNoColorRangeWarning);
        },
        py::arg("filename"), py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
            Initialize video reader for a specific video file
            :param filename: path to the video file
            :param iGpu: GPU Id to use for decoding
            :param suppressNoColorRangeWarning: Suppress warning when no color range can be extracted for a video file (and limited (MPEG) range is assumed in this case)
        )pbdoc");

    // Define the PyNvVideoReader class and its methods
    py::class_<PyNvVideoReader, shared_ptr<PyNvVideoReader>>(m, "PyNvVideoReader", py::module_local())
        .def(py::init<const std::string, int, bool>(), R"pbdoc(
            Constructor method. Initialize video reader with set of particular parameters
            :param filename: path to the video file
            :param iGpu: GPU Id to use for decoding
            :param suppressNoColorRangeWarning: Suppress warning when no color range can be extracted
            )pbdoc")
        .def(
            "Decode",
            [](std::shared_ptr<PyNvVideoReader>& reader, const std::vector<int> frame_ids) {
                try {
                    return reader->run(frame_ids);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            R"pbdoc(
                Decodes video frames into uncompressed data
                :param frame_ids: a list of frame IDs to decode from the video file
                :return: uncompressed data is returned as List of Decoded Frames
                )pbdoc")
        .def(
            "DecodeN12ToRGB",
            [](std::shared_ptr<PyNvVideoReader>& reader, const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    return reader->run_rgb_out(frame_ids, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
                Decodes video frames and converts them to RGB/BGR format
                :param frame_ids: a list of frame IDs to decode from the video file
                :param as_bgr: whether to output in BGR format (RGB otherwise)
                :return: uncompressed data is returned as List of RGB Frames
                )pbdoc")
        .def("ReplaceWithFile", &PyNvVideoReader::ReplaceWithFile, py::arg("filename"),
             R"pbdoc(
                Replace current video file with a new one
                :param filename: path to the new video file
                )pbdoc");
}
