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

#include "PyNvGopDemuxer.hpp"
#include "GopDecoderUtils.hpp"
#include "nvtx3/nvtx3.hpp"

using namespace std;
using namespace chrono;

namespace py = pybind11;

PyNvGopDemuxer::PyNvGopDemuxer(const std::string& filePath) {
    nvtxRangePushA("FFmpegDemuxer_Create");
    demuxer.reset(new FFmpegDemuxer(filePath.c_str()));
    this->filename = filePath;
    nvtxRangePop();
}

PyNvGopDemuxer::PyNvGopDemuxer(const std::string& filePath, const FastStreamInfo* fastStreamInfo) {
    nvtxRangePushA("FFmpegDemuxer_Create");
    demuxer.reset(new FFmpegDemuxer(filePath.c_str(), fastStreamInfo));
    this->filename = filePath;
    nvtxRangePop();
}

// Frame and PTS mapping methods
void PyNvGopDemuxer::set_pts_frameid_mapping(std::map<int, int64_t>&& frame2pts,
                                             std::map<int64_t, int>&& pts2frame) {
    this->frame2pts = frame2pts;
    this->pts2frame = pts2frame;
}

void PyNvGopDemuxer::set_key_frame_ids(std::vector<int>&& key_frame_ids) {
    this->key_frame_ids = key_frame_ids;
}

// Key frame handling methods
int PyNvGopDemuxer::getNextKeyFrameId(int frame_id) {
    auto next_key_it = std::upper_bound(this->key_frame_ids.begin(), this->key_frame_ids.end(), frame_id);
    if (next_key_it == this->key_frame_ids.begin()) {
        throw std::invalid_argument("[ERROR] Can not find a gop for frame: " + std::to_string(frame_id) +
                                    " only with next gop_start_id: " + std::to_string(*next_key_it));
    }
    if (next_key_it == this->key_frame_ids.end()) {
        return this->key_frame_ids.back();
    }
    return *next_key_it;
}

int PyNvGopDemuxer::getKeyFrameId(int frame_id) {
    auto cur_key_it = std::lower_bound(this->key_frame_ids.begin(), this->key_frame_ids.end(), frame_id);
    if (*cur_key_it == frame_id) {
        return *cur_key_it;
    } else {
        return *(cur_key_it - 1);
    }
}

// Color space and range methods
AVColorSpace PyNvGopDemuxer::GetColorSpace() const { return demuxer->GetColorSpace(); }

AVColorRange PyNvGopDemuxer::GetColorRange() const { return (demuxer->GetColorRange()); }

// Demuxing methods
bool PyNvGopDemuxer::Demux(uint8_t** ppVideo, int* pnVideoBytes, int& frame_id, int* pFlags, bool* pbRef) {
    int64_t timestamp;
    int flags;
    auto ret = demuxer->Demux(ppVideo, pnVideoBytes, &timestamp, pFlags);

    if (!ret) {
        return false;
    }

#ifdef IS_DEBUG_BUILD
    /*if (*pFlags & AV_PKT_FLAG_KEY) {
    std::cout << "KeyFrame Packet" << std::endl;
  }
  if (*pFlags & AV_PKT_FLAG_DISCARD) {
    std::cout << "Discard Packet" << std::endl;
  }
  if (*pFlags & AV_PKT_FLAG_DISPOSABLE) {
    std::cout << "Disposable Packet" << std::endl;
  }*/
#endif

    if (demuxer->IsVFRV2()) {
        try {
            frame_id = this->pts2frame.at(timestamp);
        } catch (const std::out_of_range& e) {
            // Handle the exception
            std::cerr << "timestamp not found in pts2frame: for timestamp: " << std::to_string(timestamp)
                      << " and filename: " << this->filename << e.what() << std::endl;
            std::cerr << "timestamp in pts2frame" << std::endl;
            for (const auto& pair : this->pts2frame) {
                std::cerr << pair.first << "\t";
            }
            std::cerr << std::endl;
            return false;
        }
    } else {
        frame_id = demuxer->FrameNumFromTs(timestamp);
    }

    if (pbRef) {
        *pbRef = true;
        if (*pFlags & AV_PKT_FLAG_DISPOSABLE) {
            *pbRef = false;
        }
    }

    if (pbRef && *pnVideoBytes && demuxer->GetVideoCodec() == AV_CODEC_ID_H264) {
        uint8_t b = (*ppVideo)[2] == 1 ? (*ppVideo)[3] : (*ppVideo)[4];
        int nal_ref_idc = b >> 5;
        int nal_unit_type = b & 0x1f;
        if (!nal_ref_idc && nal_unit_type == 1) {
            *pbRef = false;
        }
    }

    return true;
}

/*ctx is set to seek to the previous KeyFrame
The definition of KeyFrame in FFmepg/FFprobe is recovery point*/
bool PyNvGopDemuxer::Seek(uint8_t** ppVideo, int* pnVideoBytes, int frame_id_to_seek, int& frame_id_out) {
    bool condition = false;
    int64_t timestamp_keyframe, timestamp_to_seek, timestamp_out;
    if (demuxer->IsVFRV2()) {
        try {
            timestamp_keyframe = this->frame2pts.at(frame_id_to_seek);
        } catch (const std::out_of_range& e) {
            // Handle the exception
            std::cerr << "frame id not found in frame2pts frame_id: " << std::to_string(frame_id_to_seek)
                      << " and filename: " << this->filename << e.what() << std::endl;

            std::cerr << "frame_id in frame2pts" << std::endl;
            for (const auto& pair : this->frame2pts) {
                std::cerr << pair.first << "\t";
            }
            std::cerr << std::endl;

            return false;
        }
    } else {
        timestamp_keyframe = demuxer->TsFromFrameNumber(frame_id_to_seek);
    }
    // Seek to the key frame
    do {
        if (demuxer->IsVFRV2()) {
            try {
                timestamp_to_seek = this->frame2pts.at(frame_id_to_seek);
            } catch (const std::out_of_range& e) {
                // Handle the exception
                std::cerr << "frame id not found in frame2pts for frame_id: "
                          << std::to_string(frame_id_to_seek) << " and filename:" << this->filename
                          << e.what() << std::endl;

                std::cerr << "frame_id in frame2pts" << std::endl;
                for (const auto& pair : this->frame2pts) {
                    std::cerr << pair.first << "\t";
                }
                std::cerr << std::endl;

                return false;
            }
#ifdef IS_DEBUG_BUILD
            std::cout << "Seeking VFR video frame: " << frame_id_to_seek << " timestamp " << timestamp_to_seek
                      << std::endl;
#endif
        } else {
            timestamp_to_seek = demuxer->TsFromFrameNumber(frame_id_to_seek);
#ifdef IS_DEBUG_BUILD
            std::cout << "Seeking CFR video frame: " << frame_id_to_seek << " timestamp " << timestamp_to_seek
                      << std::endl;
#endif
        }
        SeekContext ctx(timestamp_to_seek, false);
        auto ret = demuxer->SeekWithTS(ctx, ppVideo, pnVideoBytes, &timestamp_out);

        condition = (timestamp_out == timestamp_keyframe);
        ++frame_id_to_seek;

        if (!ret) {
            return false;
        }
    } while (condition == false);

#ifdef IS_DEBUG_BUILD
    if (*pnVideoBytes && demuxer->GetVideoCodec() == AV_CODEC_ID_HEVC) {
        uint8_t b = (*ppVideo)[2] == 1 ? (*ppVideo)[3] : (*ppVideo)[4];
        int nal_unit_type = b >> 1;
        std::cout << "HEVC NAL_UNIT_TYPE:" << nal_unit_type << std::endl;
    } else if (*pnVideoBytes && demuxer->GetVideoCodec() == AV_CODEC_ID_H264) {
        uint8_t b = (*ppVideo)[2] == 1 ? (*ppVideo)[3] : (*ppVideo)[4];
        int nal_ref_idc = b >> 5;
        int nal_unit_type = b & 0x1f;
        std::cout << "H264 NAL Type" << nal_unit_type << std::endl;
    } else if (*pnVideoBytes && demuxer->GetVideoCodec() == AV_CODEC_ID_AV1) {
        // AV1 uses OBU (Open Bitstream Unit) format instead of NAL units
        uint8_t obu_header = (*ppVideo)[0];
        int obu_type = (obu_header >> 3) & 0x0F;
        std::cout << "AV1 OBU_TYPE:" << obu_type << std::endl;
    } else {
        throw std::domain_error("[ERROR] Unsupported video codec: " +
                                std::to_string(demuxer->GetVideoCodec()));
    }
#endif

    if (demuxer->IsVFRV2()) {
        try {
            frame_id_out = this->pts2frame.at(timestamp_out);
        } catch (const std::out_of_range& e) {
            // Handle the exception
            std::cerr << "timestamp not found in pts2frame: "
                      << "for timestamp: " << std::to_string(timestamp_out) << " and filename"
                      << this->filename << e.what() << std::endl;

            std::cerr << "timestamp in pts2frame" << std::endl;
            for (const auto& pair : this->pts2frame) {
                std::cerr << pair.first << "\t";
            }
            std::cerr << std::endl;

            return false;
        }
    } else {
        frame_id_out = demuxer->FrameNumFromTs(timestamp_out);
    }
#ifdef IS_DEBUG_BUILD
    std::cout << "Seeking to frame: " << frame_id_out << std::endl;
#endif
    return true;
}

/*Seek to the first frame of the GOP*/
bool PyNvGopDemuxer::SeekGopFirstFrameNoMap(uint8_t** ppVideo, int* pnVideoBytes, int frame_id_to_seek,
                                            int& frame_id_out) {
    if (demuxer->IsVFRV2()) {
        throw std::domain_error("[ERROR] VFR video is not supported for GOP seeking");
    }

    // std::cout << "SeekGopFirstFrameNoMap - Seeking to frame: " << frame_id_to_seek << std::endl;

    int current_frame_to_seek = frame_id_to_seek;
    bool found_key_frame = false;

    while (!found_key_frame && current_frame_to_seek >= 0) {
        int ret = 0;
        int current_timestamp_to_seek = demuxer->TsFromFrameNumber(current_frame_to_seek);
        // std::cout << "SeekGopFirstFrameNoMap - Checking current_timestamp_to_seek: " << current_timestamp_to_seek
        //           << ", current_frame_to_seek: " << current_frame_to_seek << std::endl;

        // Create seek context using frame number directly
        SeekContext ctx(current_timestamp_to_seek, false);

        int64_t timestamp_out;
        int flags;
        // Seek to the previous key frame
        // ret = demuxer->Seek(ctx, ppVideo, pnVideoBytes);
        ret = demuxer->SeekWithTS(ctx, ppVideo, pnVideoBytes, &timestamp_out);
        if (!ret) {
            return false;
        }

        frame_id_out = demuxer->FrameNumFromTs(timestamp_out);
        // printf("SeekGopFirstFrameNoMap - Seeking output frame_id_out: %d, frame_id_to_seek: %lld, timestamp_out: %lld\n", frame_id_out, frame_id_to_seek, timestamp_out);

        // std::cout << "SeekGopFirstFrameNoMap - Seeking output frame_id_out: " << frame_id_out
        //           << " timestamp_out: " << timestamp_out << std::endl;

        // Verify the packet type for H.264, HEVC, or AV1
        const uint8_t* pVideo = *ppVideo;
        if (pVideo == nullptr) {
            return false;
        }
        // Use the utility function to check for key frame NAL/OBU unit type (without checking flags)
        found_key_frame = hasKeyFrameNalType(demuxer->GetVideoCodec(), pVideo);

        if (found_key_frame) {
            // 0(key) ... 250(key) 248 247 249 254 252 251 253 ...
            if (frame_id_out > frame_id_to_seek) {
                current_frame_to_seek = current_frame_to_seek - 1;
                found_key_frame = false;
            }
        } else {
            current_frame_to_seek = frame_id_out - 1;
            if (current_frame_to_seek < 0) {
                return false;
            }
        }

        /*
         * TRICK:
         * I meet a case, gop0:[0, 29], gop1:[30, 59], gop2:[60, 89]
         * Seek to target_frame_id=30, it reture key_frame_id=29....
         * So I add this trick to count the gop_len
         */
        //TODO
    }

    // std::cout << "SeekGopFirstFrameNoMap - Found key frame at frame: " << frame_id_out << std::endl;

    return found_key_frame;
}
