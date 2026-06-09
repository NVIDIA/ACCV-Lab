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
#include <map>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
extern "C" {
#include <libavutil/frame.h>
}

class PyNvGopDemuxer {
   protected:
    std::unique_ptr<FFmpegDemuxer> demuxer;
    std::string filename;
    std::map<int, int64_t> frame2pts;
    std::map<int64_t, int> pts2frame;
    std::vector<int> key_frame_ids;

   public:
    explicit PyNvGopDemuxer(const std::string&);
    explicit PyNvGopDemuxer(const std::string& filePath, const FastStreamInfo* fastStreamInfo);

    uint32_t GetHeight() { return demuxer->GetHeight(); }

    uint32_t GetWidth() { return demuxer->GetWidth(); }

    uint32_t GetFrameSize() { return demuxer->GetFrameSize(); }

    FFmpegDemuxer* GetDemuxer() { return demuxer.get(); }

    bool HasDemuxError() const { return demuxer && demuxer->HasDemuxError(); }

    std::string GetLastDemuxError() const { return demuxer ? demuxer->GetLastDemuxError() : ""; }

    const std::string& GetFilename() const { return filename; }

    bool IsVFR() { return demuxer->IsVFR(); }
    bool IsVFRV2() { return demuxer->IsVFRV2(); }

    bool IsValid() { return demuxer->IsValid(); }

    AVColorSpace GetColorSpace() const;

    AVColorRange GetColorRange() const;

    double GetFrameRate() const { return demuxer->GetFrameRate(); };

    cudaVideoCodec GetNvCodecId() { return FFmpeg2NvCodecId(demuxer->GetVideoCodec()); }

    void set_pts_frameid_mapping(std::map<int, int64_t>&& frame2pts, std::map<int64_t, int>&& pts2frame);

    void set_key_frame_ids(std::vector<int>&& key_frame_ids);

    int getNextKeyFrameId(int frame_id);

    int getKeyFrameId(int frame_id);

    int getFirstKeyFrameId() const { return key_frame_ids.front(); }

    int getLastKeyFrameId() const { return key_frame_ids.back(); }

    bool Demux(uint8_t** ppVideo, int* pnVideoBytes, int& frame_id, int* pFlags, bool* pbRef = nullptr);

    bool Seek(uint8_t** ppVideo, int* pnVideoBytes, int frame_id_to_seek, int& frame_id_out);

    /*Seek to the first frame of the GOP*/
    bool SeekGopFirstFrameNoMap(uint8_t** ppVideo, int* pnVideoBytes, int frame_id_to_seek,
                                int& frame_id_out);

    // bool isEndOfStream() { return demuxer->isEOF(); }
};
