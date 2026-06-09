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

extern "C" {
#include <fcntl.h>
#include <cstdlib>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
/* Explicitly include bsf.h when building against FFmpeg 4.3 (libavcodec 58.45.100) or later for backward compatibility */
#if LIBAVCODEC_VERSION_INT >= 3824484
#include <libavcodec/bsf.h>
#endif
}
#include "cuviddec.h"
#include "nvcuvid.h"
#include "NvCodecUtils.h"
#include "nvtx3/nvtx3.hpp"
#include <string>
using namespace std;
//---------------------------------------------------------------------------
//! \file FFmpegDemuxer.h
//! \brief Provides functionality for stream demuxing
//!
//! This header file is used by Decode/Transcode apps to demux input video clips before decoding frames from it.
//---------------------------------------------------------------------------

enum SeekMode {
    /* Seek for exact frame number.
     * Suited for standalone demuxer seek. */
    EXACT_FRAME = 0,
    /* Seek for previous key frame in past.
     * Suitable for seek & decode.  */
    PREV_KEY_FRAME = 1,
    /* Seek for nearest key frame in future.
     * Suitable for seek & decode.  */
    NEAREST_FUTURE_KEY_FRAME = 2,

    SEEK_MODE_NUM_ELEMS
};

enum SeekCriteria {
    /* Seek frame by number.
     */
    BY_NUMBER = 0,
    /* Seek frame by timestamp.
     */
    BY_TIMESTAMP = 1,

    SEEK_CRITERIA_NUM_ELEMS
};

struct FastStreamInfo {
    int codec_type;  // will be cast to enum AVMediaType
    int codec_id;    // will be cast to enum AVCodecID
    int width;
    int height;
    int format;
    
    int time_base_num;
    int time_base_den;
    
    int avg_frame_rate_num;
    int avg_frame_rate_den;
    
    int r_frame_rate_num;
    int r_frame_rate_den;
    
    int64_t start_time;
    int64_t duration;
};

struct SeekContext {
    /* Will be set to false for default ctor, true otherwise;
     */
    bool use_seek;

    /* Frame we want to get. Set by user.
     * Shall be set to frame timestamp in case seek is done by time.
     */
    uint64_t seek_frame;

    /* Mode in which we seek. */
    SeekMode mode;

    /* Criteria by which we seek. */
    SeekCriteria crit;

    /* PTS of frame found after seek. */
    int64_t out_frame_pts;

    /* Duration of frame found after seek. */
    int64_t out_frame_duration;

    /* Number of frames that were decoded during seek. */
    uint64_t num_frames_decoded;

    SeekContext()
        : use_seek(false),
          seek_frame(0),
          mode(PREV_KEY_FRAME),
          crit(BY_NUMBER),
          out_frame_pts(0),
          out_frame_duration(0),
          num_frames_decoded(0U) {}

    SeekContext(uint64_t frame_id)
        : use_seek(true),
          seek_frame(frame_id),
          mode(PREV_KEY_FRAME),
          crit(BY_NUMBER),
          out_frame_pts(0),
          out_frame_duration(0),
          num_frames_decoded(0U) {}

    SeekContext(uint64_t timestamp, bool by_frame_id = true)
        : use_seek(true),
          seek_frame(timestamp),
          mode(PREV_KEY_FRAME),
          out_frame_pts(0),
          out_frame_duration(0),
          num_frames_decoded(0U) {
        if (by_frame_id) {
            crit = BY_NUMBER;
        } else {
            crit = BY_TIMESTAMP;
        }
    }

    SeekContext& operator=(const SeekContext& other) {
        use_seek = other.use_seek;
        seek_frame = other.seek_frame;
        mode = other.mode;
        crit = other.crit;
        out_frame_pts = other.out_frame_pts;
        out_frame_duration = other.out_frame_duration;
        num_frames_decoded = other.num_frames_decoded;
        return *this;
    }
};

/**
 * 
 * ============================== buffered IO start ===============================
 */
struct FFBufferIO {
    int fd;
    char* file;
};

static int ffbufio_read_packet(void* opaque, uint8_t* buf, int buf_size) {
    if (!buf_size) {
        return AVERROR_EOF;
    }
    //
    assert(opaque != NULL);
    struct FFBufferIO* bufio = (struct FFBufferIO*)opaque;
    int ret = ::read(bufio->fd, buf, buf_size);
    if (ret == 0) {
        return AVERROR_EOF;
    } else if (ret < 0) {
        int eno = errno;

        LOG(ERROR) << "error occurred when reading file: "
                   << (bufio->file ? bufio->file : to_string(bufio->fd))
                   << ", POSIX error message: " << strerror(errno);
        return AVUNERROR(eno);
    }
    return ret;
}

static int64_t ffbufio_seek(void* opaque, int64_t offset, int whence) {
    assert(opaque != NULL);
    struct FFBufferIO* bufio = (struct FFBufferIO*)opaque;
    return ::lseek(bufio->fd, offset, whence);
}

/**
* @brief libavformat wrapper class. Retrieves the elementary encoded stream from the container format.
*/
// =====================================================================
// SPS extradata parser — fallback for pixel format detection
// ---------------------------------------------------------------------
// `avformat_find_stream_info` usually fills `AVCodecParameters::format`
// through codec-side parsing. In some builds, pixel format information
// may remain unavailable in `codecpar->format`, which can cause later
// logic to fall back to a default chroma layout.
//
// The helpers in this namespace parse the SPS NAL stored in container
// extradata (HVCC for HEVC, avcC for H.264) and extract the minimum
// fields needed to infer pixel format, specifically `bit_depth_luma`
// and `chroma_format_idc`.
//
// This is a lightweight metadata parser: it reads stream parameter
// fields only and is used for format detection rather than frame
// reconstruction.
// =====================================================================
namespace ffmpeg_demuxer_detail {

struct BitReader {
    const uint8_t* buf;
    int byte;
    int bit;
    int size;
};

inline int br_read_bit(BitReader* b) {
    if (b->byte >= b->size) return 0;
    int v = (b->buf[b->byte] >> (7 - b->bit)) & 1;
    if (++b->bit == 8) { b->bit = 0; b->byte++; }
    return v;
}

inline unsigned br_read_bits(BitReader* b, int n) {
    unsigned v = 0;
    while (n-- > 0) { v = (v << 1) | (unsigned)br_read_bit(b); }
    return v;
}

inline unsigned br_read_ue(BitReader* b) {
    int zeros = 0;
    while (zeros < 32 && b->byte < b->size && br_read_bit(b) == 0) zeros++;
    return (1u << zeros) - 1 + br_read_bits(b, zeros);
}

// Strip H.264/HEVC emulation prevention bytes (0x00 0x00 0x03 -> 0x00 0x00).
inline int rbsp_strip(const uint8_t* src, int n, uint8_t* dst, int dst_capacity) {
    int o = 0;
    for (int i = 0; i < n && o < dst_capacity; i++) {
        if (i + 2 < n && src[i] == 0 && src[i+1] == 0 && src[i+2] == 3) {
            if (o + 2 > dst_capacity) break;
            dst[o++] = 0; dst[o++] = 0; i += 2;
        } else {
            dst[o++] = src[i];
        }
    }
    return o;
}

// Locate an SPS NAL inside HEVC HVCC extradata.
inline bool find_hevc_sps(const uint8_t* ed, int n,
                          const uint8_t** sps_out, int* sps_len_out) {
    if (n < 23 || ed[0] != 1) return false;  // configurationVersion=1 expected
    int p = 22;
    int num_arrays = ed[p++];
    for (int a = 0; a < num_arrays && p < n; a++) {
        if (p + 3 > n) return false;
        int nal_type = ed[p] & 0x3F;
        int num_nalus = (ed[p+1] << 8) | ed[p+2];
        p += 3;
        for (int i = 0; i < num_nalus; i++) {
            if (p + 2 > n) return false;
            int nl = (ed[p] << 8) | ed[p+1];
            p += 2;
            if (p + nl > n) return false;
            if (nal_type == 33 /* HEVC NAL_SPS */) {
                *sps_out = ed + p;
                *sps_len_out = nl;
                return true;
            }
            p += nl;
        }
    }
    return false;
}

// Locate the first SPS NAL inside H.264 avcC extradata.
inline bool find_h264_sps(const uint8_t* ed, int n,
                          const uint8_t** sps_out, int* sps_len_out) {
    if (n < 7 || ed[0] != 1) return false;  // configurationVersion=1
    int num_sps = ed[5] & 0x1F;
    if (num_sps < 1) return false;
    int p = 6;
    if (p + 2 > n) return false;
    int nl = (ed[p] << 8) | ed[p+1];
    p += 2;
    if (p + nl > n) return false;
    *sps_out = ed + p;
    *sps_len_out = nl;
    return true;
}

// Parse a HEVC SPS RBSP for chroma_format_idc and bit_depth_luma.
inline bool parse_hevc_sps(const uint8_t* sps_nal, int len,
                           int* bit_depth_out, int* chroma_idc_out) {
    if (len < 3) return false;
    uint8_t buf[8192];
    int copy_len = len > (int)sizeof(buf) ? (int)sizeof(buf) : len;
    int rl = rbsp_strip(sps_nal, copy_len, buf, (int)sizeof(buf));
    BitReader b{ buf, 0, 0, rl };

    br_read_bits(&b, 16);                       // 2-byte HEVC NAL header
    br_read_bits(&b, 4);                        // sps_video_parameter_set_id
    int max_sub = (int)br_read_bits(&b, 3);     // sps_max_sub_layers_minus1
    br_read_bits(&b, 1);                        // sps_temporal_id_nesting_flag

    // profile_tier_level (profilePresentFlag = 1)
    br_read_bits(&b, 2 + 1 + 5);                // general_profile_space/tier/idc
    br_read_bits(&b, 32);                       // general_profile_compatibility_flag
    br_read_bits(&b, 4);                        // 4 source-format flags
    br_read_bits(&b, 43);                       // constraint flags
    br_read_bits(&b, 1);                        // general_inbld_flag
    br_read_bits(&b, 8);                        // general_level_idc

    int sub_p[8] = {0}, sub_l[8] = {0};
    for (int i = 0; i < max_sub; i++) {
        sub_p[i] = br_read_bit(&b);
        sub_l[i] = br_read_bit(&b);
    }
    if (max_sub > 0) {
        for (int i = max_sub; i < 8; i++) br_read_bits(&b, 2);
    }
    for (int i = 0; i < max_sub; i++) {
        if (sub_p[i]) {
            br_read_bits(&b, 2 + 1 + 5);
            br_read_bits(&b, 32);
            br_read_bits(&b, 4);
            br_read_bits(&b, 43);
            br_read_bits(&b, 1);
        }
        if (sub_l[i]) br_read_bits(&b, 8);
    }

    br_read_ue(&b);                             // sps_seq_parameter_set_id
    *chroma_idc_out = (int)br_read_ue(&b);
    if (*chroma_idc_out == 3) br_read_bit(&b);  // separate_colour_plane_flag
    br_read_ue(&b);                             // pic_width_in_luma_samples
    br_read_ue(&b);                             // pic_height_in_luma_samples
    if (br_read_bit(&b)) {                      // conformance_window_flag
        br_read_ue(&b); br_read_ue(&b);
        br_read_ue(&b); br_read_ue(&b);
    }
    *bit_depth_out = 8 + (int)br_read_ue(&b);
    return true;
}

// Parse an H.264 SPS RBSP. Baseline / Main profiles imply 4:2:0 8-bit.
inline bool parse_h264_sps(const uint8_t* sps_nal, int len,
                           int* bit_depth_out, int* chroma_idc_out) {
    if (len < 4) return false;
    uint8_t buf[4096];
    int copy_len = len > (int)sizeof(buf) ? (int)sizeof(buf) : len;
    int rl = rbsp_strip(sps_nal, copy_len, buf, (int)sizeof(buf));
    BitReader b{ buf, 0, 0, rl };

    br_read_bits(&b, 8);                        // 1-byte H.264 NAL header
    int profile_idc = (int)br_read_bits(&b, 8);
    br_read_bits(&b, 8);                        // constraint_set + reserved
    br_read_bits(&b, 8);                        // level_idc
    br_read_ue(&b);                             // seq_parameter_set_id

    // Only the high profiles carry chroma_format_idc / bit_depth fields.
    // ITU-T H.264 (V14) 7.4.2.1.1.
    static const int hi_profiles[] = {
        100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135
    };
    bool is_high = false;
    for (size_t i = 0; i < sizeof(hi_profiles) / sizeof(hi_profiles[0]); i++) {
        if (profile_idc == hi_profiles[i]) { is_high = true; break; }
    }
    if (!is_high) {
        *chroma_idc_out = 1;   // implied 4:2:0
        *bit_depth_out  = 8;   // implied 8-bit
        return true;
    }

    *chroma_idc_out = (int)br_read_ue(&b);
    if (*chroma_idc_out == 3) br_read_bit(&b);  // separate_colour_plane_flag
    *bit_depth_out = 8 + (int)br_read_ue(&b);
    return true;
}

// Map (chroma_format_idc, bit_depth) to the AVPixelFormat enum values the
// existing eChromaFormat switch handles. Returns AV_PIX_FMT_NONE for
// combinations we do not synthesize; the existing default-branch fallback
// will then preserve current behavior.
inline AVPixelFormat pix_fmt_from_sps(int chroma_idc, int bit_depth) {
    switch (chroma_idc) {
        case 0:  // monochrome
            if (bit_depth == 8)  return AV_PIX_FMT_GRAY8;
            if (bit_depth == 10) return AV_PIX_FMT_GRAY10LE;
            break;
        case 1:  // 4:2:0
            if (bit_depth == 8)  return AV_PIX_FMT_YUV420P;
            if (bit_depth == 10) return AV_PIX_FMT_YUV420P10LE;
            if (bit_depth == 12) return AV_PIX_FMT_YUV420P12LE;
            break;
        case 3:  // 4:4:4
            if (bit_depth == 8)  return AV_PIX_FMT_YUV444P;
            if (bit_depth == 10) return AV_PIX_FMT_YUV444P10LE;
            if (bit_depth == 12) return AV_PIX_FMT_YUV444P12LE;
            break;
        default:
            break;
    }
    return AV_PIX_FMT_NONE;
}

// Top-level entry point. Returns AV_PIX_FMT_NONE when recovery is not
// possible (unsupported codec, missing extradata, malformed SPS, or an
// unmappable chroma/bit-depth combination).
inline AVPixelFormat recover_pix_fmt_from_extradata(const AVCodecParameters* cp) {
    if (!cp || !cp->extradata || cp->extradata_size <= 0) {
        return AV_PIX_FMT_NONE;
    }
    const uint8_t* sps = nullptr;
    int sps_len = 0;
    int bit_depth = 8;
    int chroma_idc = 1;

    if (cp->codec_id == AV_CODEC_ID_HEVC) {
        if (!find_hevc_sps(cp->extradata, cp->extradata_size, &sps, &sps_len)) {
            return AV_PIX_FMT_NONE;
        }
        if (!parse_hevc_sps(sps, sps_len, &bit_depth, &chroma_idc)) {
            return AV_PIX_FMT_NONE;
        }
    } else if (cp->codec_id == AV_CODEC_ID_H264) {
        if (!find_h264_sps(cp->extradata, cp->extradata_size, &sps, &sps_len)) {
            return AV_PIX_FMT_NONE;
        }
        if (!parse_h264_sps(sps, sps_len, &bit_depth, &chroma_idc)) {
            return AV_PIX_FMT_NONE;
        }
    } else {
        return AV_PIX_FMT_NONE;
    }

    return pix_fmt_from_sps(chroma_idc, bit_depth);
}

}  // namespace ffmpeg_demuxer_detail

class FFmpegDemuxer {
   private:
    AVFormatContext* fmtc = NULL;
    AVIOContext* avioc = NULL;
    AVPacket* pkt =
        NULL; /*!< AVPacket stores compressed data typically exported by demuxers and then passed as input to decoders */
    AVPacket* pktFiltered = NULL;
    AVBSFContext* bsfc = NULL;
    AVCodec* codec;
    AVCodecContext* codecContext;

    int iVideoStream;
    bool bMp4H264, bMp4HEVC, bMp4MPEG4, is_seekable;
    AVCodecID eVideoCodec;
    AVPixelFormat eChromaFormat;
    int nWidth, nHeight, nBitDepth, nBPP, nChromaHeight;
    double timeBase = 0.0;
    int64_t userTimeScale = 0;
    double framerate;
    double avg_framerate;
    AVColorSpace color_space;
    AVColorRange color_range;

    uint8_t* pDataWithHeader = NULL;

    unsigned int frameCount = 0;

    FFBufferIO* ffbufIO = NULL;
    bool demux_error = false;
    std::string demux_error_message;

    void ClearDemuxError() {
        demux_error = false;
        demux_error_message.clear();
    }

    void SetDemuxError(const std::string& message) {
        demux_error = true;
        demux_error_message = message;
        LOG(ERROR) << message;
    }

   public:
    class DataProvider {
       public:
        virtual ~DataProvider() {}
        virtual int GetData(uint8_t* pBuf, int nBuf) = 0;
    };

   private:
    void init(AVFormatContext* fmtc_, int64_t timeScale = 1000 /*Hz*/, const FastStreamInfo* fastStreamInfo = nullptr) {
        if (!fmtc_) {
            LOG(ERROR) << "No AVFormatContext provided.";
            return;
        }
        fmtc = fmtc_;

        // Allocate the AVPackets and initialize to default values
        pkt = av_packet_alloc();
        pktFiltered = av_packet_alloc();
        if (!pkt || !pktFiltered) {
            LOG(ERROR) << "AVPacket allocation failed";
            return;
        }

        // LOG(INFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

        if (fastStreamInfo) {
            if (fmtc->nb_streams > 1) {
                LOG(ERROR) << "nb_streams = " << fmtc->nb_streams << ", fast set stream info is not supported";
                return;
            }
            iVideoStream = 0;
            fmtc->streams[iVideoStream]->codecpar->codec_type = static_cast<AVMediaType>(fastStreamInfo->codec_type);
            fmtc->streams[iVideoStream]->codecpar->codec_id = static_cast<AVCodecID>(fastStreamInfo->codec_id);
            fmtc->streams[iVideoStream]->codecpar->width = fastStreamInfo->width;
            fmtc->streams[iVideoStream]->codecpar->height = fastStreamInfo->height;
            fmtc->streams[iVideoStream]->codecpar->format = fastStreamInfo->format;
            fmtc->streams[iVideoStream]->time_base = av_make_q(fastStreamInfo->time_base_num, fastStreamInfo->time_base_den);
            fmtc->streams[iVideoStream]->avg_frame_rate = av_make_q(fastStreamInfo->avg_frame_rate_num, fastStreamInfo->avg_frame_rate_den);
            fmtc->streams[iVideoStream]->r_frame_rate = av_make_q(fastStreamInfo->r_frame_rate_num, fastStreamInfo->r_frame_rate_den);
            fmtc->streams[iVideoStream]->start_time = fastStreamInfo->start_time;
            fmtc->streams[iVideoStream]->duration = fastStreamInfo->duration;
        } else {
            nvtxRangePushA("avformat_find_stream_info");
            // fmtc->probesize = 512 * 1024;
            // fmtc->max_analyze_duration = 500000;
            // fmtc->fps_probe_size = 0;
            ck(avformat_find_stream_info(fmtc, NULL));
            iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
            nvtxRangePop();

            // printf("codecpar: %p\n", fmtc->streams[0]->codecpar);
            // printf("codecpar->codec_type: %d\n", fmtc->streams[0]->codecpar->codec_type);
            // printf("codecpar->codec_id: %d\n", fmtc->streams[0]->codecpar->codec_id);
            // printf("codecpar->width: %d\n", fmtc->streams[0]->codecpar->width);
            // printf("codecpar->height: %d\n", fmtc->streams[0]->codecpar->height);
            // printf("codecpar->format: %d\n", fmtc->streams[0]->codecpar->format);
            // printf("codecpar->time_base: %d/%d\n", fmtc->streams[0]->time_base.num, fmtc->streams[0]->time_base.den);
            // printf("codecpar->avg_frame_rate: %d/%d\n", fmtc->streams[0]->avg_frame_rate.num, fmtc->streams[0]->avg_frame_rate.den);
            // printf("codecpar->r_frame_rate: %d/%d\n", fmtc->streams[0]->r_frame_rate.num, fmtc->streams[0]->r_frame_rate.den);
            // printf("codecpar->start_time: %d\n", fmtc->streams[0]->start_time);
            // printf("codecpar->duration: %d\n", fmtc->streams[0]->duration);
        }
        
        if (iVideoStream < 0) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                       << "Could not find stream in input file";
            av_packet_free(&pkt);
            av_packet_free(&pktFiltered);
            return;
        }

        // int iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

        const AVCodec* codec = avcodec_find_decoder(fmtc->streams[iVideoStream]->codecpar->codec_id);
        // Get the codec context
        codecContext = avcodec_alloc_context3(codec);
        if (!codecContext) {
            // Out of memory
            avformat_close_input(&fmtc);
        }

        // Set the parameters of the codec context from the stream
        int result = avcodec_parameters_to_context(codecContext, fmtc->streams[iVideoStream]->codecpar);

        // Ready to open stream based on previous parameters
        // Third parameter (NULL) is optional dictionary settings
        if (avcodec_open2(codecContext, codec, NULL) < 0) {
            // Cannot open the video codec
            codecContext = nullptr;
        }

        //fmtc->streams[iVideoStream]->need_parsing = AVSTREAM_PARSE_NONE;
        eVideoCodec = fmtc->streams[iVideoStream]->codecpar->codec_id;
        nWidth = fmtc->streams[iVideoStream]->codecpar->width;
        nHeight = fmtc->streams[iVideoStream]->codecpar->height;
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        AVRational rTimeBase = fmtc->streams[iVideoStream]->time_base;
        timeBase = av_q2d(rTimeBase);
        userTimeScale = timeScale;
        framerate = (double)fmtc->streams[iVideoStream]->r_frame_rate.num /
                    (double)fmtc->streams[iVideoStream]->r_frame_rate.den;
        avg_framerate = (double)fmtc->streams[iVideoStream]->avg_frame_rate.num /
                        (double)fmtc->streams[iVideoStream]->avg_frame_rate.den;
        // Set bit depth, chroma height, bits per pixel based on eChromaFormat of input
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        color_space = fmtc->streams[iVideoStream]->codecpar->color_space;
        color_range = fmtc->streams[iVideoStream]->codecpar->color_range;

        // FFmpeg builds without the HEVC / H.264 decoder cannot determine pix_fmt
        // during stream-info probing. Fall back to parsing SPS extradata so the
        // switch below sees the real format instead of taking the default branch.
        if (eChromaFormat == AV_PIX_FMT_NONE) {
            AVPixelFormat recovered = ffmpeg_demuxer_detail::recover_pix_fmt_from_extradata(
                fmtc->streams[iVideoStream]->codecpar);
            if (recovered != AV_PIX_FMT_NONE) {
                const char* name = av_get_pix_fmt_name(recovered);
                LOG(INFO) << "Recovered pix_fmt from SPS extradata: "
                          << (name ? name : "?");
                eChromaFormat = recovered;
            }
        }

        switch (eChromaFormat) {
            case AV_PIX_FMT_YUV420P10LE:
            case AV_PIX_FMT_GRAY10LE:  // monochrome is treated as 420 with chroma filled with 0x0
                nBitDepth = 10;
                nChromaHeight = (nHeight + 1) >> 1;
                nBPP = 2;
                break;
            case AV_PIX_FMT_YUV420P12LE:
                nBitDepth = 12;
                nChromaHeight = (nHeight + 1) >> 1;
                nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P10LE:
                nBitDepth = 10;
                nChromaHeight = nHeight << 1;
                nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P12LE:
                nBitDepth = 12;
                nChromaHeight = nHeight << 1;
                nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P:
                nBitDepth = 8;
                nChromaHeight = nHeight << 1;
                nBPP = 1;
                break;
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUVJ420P:
            case AV_PIX_FMT_YUVJ422P:  // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
            case AV_PIX_FMT_YUVJ444P:  // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
            case AV_PIX_FMT_GRAY8:  // monochrome is treated as 420 with chroma filled with 0x0
                nBitDepth = 8;
                nChromaHeight = (nHeight + 1) >> 1;
                nBPP = 1;
                break;
            default: {
                // Warn on first occurrence only to avoid log spam when the
                // same unsupported pixfmt streams in repeatedly.
                static bool warned = false;
                if (!warned) {
                    LOG(WARNING) << "ChromaFormat not recognized. Assuming 420";
                    warned = true;
                }
                eChromaFormat = AV_PIX_FMT_YUV420P;
                nBitDepth = 8;
                nChromaHeight = (nHeight + 1) >> 1;
                nBPP = 1;
            }
        }

        bMp4H264 =
            eVideoCodec == AV_CODEC_ID_H264 && (!strcmp(fmtc->iformat->long_name, "QuickTime / MOV") ||
                                                !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") ||
                                                !strcmp(fmtc->iformat->long_name, "Matroska / WebM"));
        bMp4HEVC =
            eVideoCodec == AV_CODEC_ID_HEVC && (!strcmp(fmtc->iformat->long_name, "QuickTime / MOV") ||
                                                !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") ||
                                                !strcmp(fmtc->iformat->long_name, "Matroska / WebM"));

        bMp4MPEG4 =
            eVideoCodec == AV_CODEC_ID_MPEG4 && (!strcmp(fmtc->iformat->long_name, "QuickTime / MOV") ||
                                                 !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") ||
                                                 !strcmp(fmtc->iformat->long_name, "Matroska / WebM"));

        // Initialize bitstream filter and its required resources
        if (bMp4H264) {
            const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                           << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }
        if (bMp4HEVC) {
            const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                           << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }

        /* Some inputs doesn't allow seek functionality.
        * Check this ahead of time. */
        is_seekable = fmtc->iformat->read_seek || fmtc->iformat->read_seek2;
    }

    /**
    *   @brief  Private constructor to initialize libavformat resources.
    *   @param  fmtc - Pointer to AVFormatContext allocated inside avformat_open_input()
    */
    FFmpegDemuxer(AVFormatContext* fmtc, int64_t timeScale = 1000 /*Hz*/) : fmtc(fmtc) {
        nvtxRangePushA("init");
        init(fmtc, timeScale);
        nvtxRangePop();
    }

    FFmpegDemuxer(AVFormatContext* fmtc, const FastStreamInfo* fastStreamInfo, int64_t timeScale = 1000 /*Hz*/) : fmtc(fmtc) {
        nvtxRangePushA("fast init");
        init(fmtc, timeScale, fastStreamInfo);
        nvtxRangePop();
    }

    AVFormatContext* CreateFormatContext(DataProvider* pDataProvider) {
        AVFormatContext* ctx = NULL;
        if (!(ctx = avformat_alloc_context())) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }

        uint8_t* avioc_buffer = NULL;
        int avioc_buffer_size = 8 * 1024 * 1024;
        avioc_buffer = (uint8_t*)av_malloc(avioc_buffer_size);
        if (!avioc_buffer) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        avioc =
            avio_alloc_context(avioc_buffer, avioc_buffer_size, 0, pDataProvider, &ReadPacket, NULL, NULL);
        if (!avioc) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        ctx->pb = avioc;

        ck(avformat_open_input(&ctx, NULL, NULL, NULL));
        return ctx;
    }

    /**
    *   @brief  Allocate and return AVFormatContext*.
    *   @param  szFilePath - Filepath pointing to input stream.
    *   @return Pointer to AVFormatContext
    */
    AVFormatContext* CreateFormatContext(const char* szFilePath) {
        av_log_set_level(AV_LOG_QUIET);
        avformat_network_init();

        AVFormatContext* ctx = NULL;
        ck(avformat_open_input(&ctx, szFilePath, NULL, NULL));
        return ctx;
    }

   public:
    FFmpegDemuxer(const char *szFilePath, int64_t timescale = 1000 /*Hz*/) : FFmpegDemuxer(CreateFormatContext(szFilePath), timescale) {}
    FFmpegDemuxer(const char *szFilePath, const FastStreamInfo* fastStreamInfo, int64_t timescale = 1000 /*Hz*/) : FFmpegDemuxer(CreateFormatContext(szFilePath), fastStreamInfo, timescale) {}
    
    FFmpegDemuxer(DataProvider* pDataProvider) : FFmpegDemuxer(CreateFormatContext(pDataProvider)) {
        avioc = fmtc->pb;
    }

    FFmpegDemuxer(const char* szFilePath, int buf_size /*512*1024*/, int64_t timescale = 1000) {
        unsigned char* io_buffer;
        int fd = ::open(szFilePath, O_RDONLY);
        if (fd < 0) {
            LOG(ERROR) << "posix error at line " << __LINE__ << " in file " << __FILE__
                       << ", function open read-only file " << szFilePath
                       << " failed, error message: " << strerror(errno);
            return;
        }
        // prepare buffered IO
        ffbufIO = (FFBufferIO*)malloc(sizeof(FFBufferIO));
        if (ffbufIO == nullptr) {
            LOG(ERROR) << "memory error "
                       << " at line " << __LINE__ << " in file " << __FILE__
                       << ", function malloc(FFBufferIO) returns null";
            goto failed_clean;
        }
        memset(ffbufIO, 0, sizeof(FFBufferIO));
        ffbufIO->fd = fd;
        ffbufIO->file = strdup(szFilePath);
        if (!ffbufIO->file) {
            LOG(WARNING) << "memory error "
                         << " at line " << __LINE__ << " in file " << __FILE__ << ", strdup(" << szFilePath
                         << ") returns null.";
        }
        // prepare ffmpeg
        av_log_set_level(AV_LOG_QUIET);
        avformat_network_init();
        io_buffer = (unsigned char*)av_mallocz(buf_size);
        if (io_buffer == nullptr) {
            LOG(ERROR) << "FFmpeg error "
                       << " at line " << __LINE__ << " in file " << __FILE__ << ", function av_malloc("
                       << buf_size << ") returns null";
            goto failed_clean;
        }
        avioc =
            avio_alloc_context(io_buffer, buf_size, 0, ffbufIO, &ffbufio_read_packet, NULL, &ffbufio_seek);
        if (avioc == nullptr) {
            LOG(ERROR) << "FFmpeg error "
                       << " at line " << __LINE__ << " in file " << __FILE__
                       << ", function avio_alloc_context returns null";
            goto failed_clean;
        }

        //
        fmtc = avformat_alloc_context();
        if (fmtc == nullptr) {
            LOG(ERROR) << "FFmpeg error "
                       << " at line " << __LINE__ << " in file " << __FILE__
                       << ", function avformat_alloc_context() returns null";
            goto failed_clean;
        }
        fmtc->pb = avioc;

        ck(avformat_open_input(&fmtc, NULL, NULL, NULL));
        init(fmtc, timescale);
        return;
    failed_clean:
        // clean fd
        if (fd > 0) {
            ::close(fd);
        }
        if (ffbufIO) {
            if (ffbufIO->file) {
                free(ffbufIO->file);
            }
            ffbufIO->fd = 0;
            free(ffbufIO);
        }
        // clean avio buffer
        if (io_buffer) {
            av_freep(&io_buffer);
        }
        // clean avio
        if (avioc) {
            av_freep(&avioc);
        }
    }

    ~FFmpegDemuxer() {
        if (!fmtc) {
            return;
        }

        if (pkt) {
            av_packet_free(&pkt);
        }
        if (pktFiltered) {
            av_packet_free(&pktFiltered);
        }

        if (bsfc) {
            av_bsf_free(&bsfc);
        }
        if (codecContext) {
            avcodec_free_context(&codecContext);
        }

        avformat_close_input(&fmtc);

        if (avioc) {
            av_freep(&avioc->buffer);
            av_freep(&avioc);
        }

        if (pDataWithHeader) {
            av_free(pDataWithHeader);
        }

        // free buffered IO
        if (ffbufIO) {
            if (ffbufIO->fd > 0) {
                ::close(ffbufIO->fd);
                ffbufIO->fd = 0;
            }
            if (ffbufIO->file) {
                free(ffbufIO->file);
            }
            free(ffbufIO);
        }
    }
    AVCodecID GetVideoCodec() { return eVideoCodec; }
    AVPixelFormat GetChromaFormat() { return eChromaFormat; }
    int GetWidth() { return nWidth; }
    int GetHeight() { return nHeight; }
    int GetBitDepth() { return nBitDepth; }
    int GetFrameSize() { return nWidth * (nHeight + nChromaHeight) * nBPP; }

    double GetFrameRate() { return framerate; }
    double GetAvgFrameRate() { return avg_framerate; }

    AVPixelFormat GetPixelFormat() const { return eChromaFormat; }

    AVColorSpace GetColorSpace() const { return color_space; }

    AVColorRange GetColorRange() const { return color_range; }

    bool IsVFR() const { return framerate != avg_framerate; }
    bool IsVFRV2() const {
        /* Some videos have framerate = 30.000000 and avg_framerate = 30.024213,
         * since some video first frame is not key frame.
         */
        if (abs(framerate - avg_framerate) < 0.1)
            return false;
        else
            return true;
    }
    bool IsValid() const { return fmtc != nullptr; }
    bool HasDemuxError() const { return demux_error; }
    const std::string& GetLastDemuxError() const { return demux_error_message; }
    int64_t TsFromTime(double ts_sec) {
        /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
         * and switch to fixed point precision arithmetics; */
        auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

        // Rescale the timestamp to value represented in stream base units;
        AVRational factor;
        factor.num = 1;
        factor.den = AV_TIME_BASE;
        return av_rescale_q(ts_tbu, factor, fmtc->streams[iVideoStream]->time_base);
    }

    int64_t TsFromFrameNumber(int64_t frame_num) {
        auto const ts_sec = (double)frame_num / framerate;
        return TsFromTime(ts_sec) + fmtc->streams[iVideoStream]->start_time;
    }

    double TimeFromTs(int64_t ts) { return ts * av_q2d(fmtc->streams[iVideoStream]->time_base); }

    int FrameNumFromTs(int64_t ts) {
        /**
         * TRICK:
         * we add 0.5 to the result to avoid precision loss
         * a corner case, w/o +0.5, input ts=79999, output frame_id=1.999975
         */
        return (framerate * (ts - fmtc->streams[iVideoStream]->start_time) /
                (double)fmtc->streams[iVideoStream]->time_base.den) *
                   (double)fmtc->streams[iVideoStream]->time_base.num +
               0.5;
    }

    bool Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts = NULL, int* flag = NULL) {
        NVTX_SCOPED_RANGE("demux")
        ClearDemuxError();
        if (!fmtc) {
            SetDemuxError("No AVFormatContext provided for demux.");
            return false;
        }

        *pnVideoBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        int e = 0;
        while ((e = av_read_frame(fmtc, pkt)) >= 0 && pkt->stream_index != iVideoStream) {
            av_packet_unref(pkt);
        }
        if (e < 0) {
            if (e != AVERROR_EOF) {
                SetDemuxError("FFmpeg read frame failed for codec " +
                              std::string(avcodec_get_name(eVideoCodec)) +
                              ", ret=" + std::to_string(e));
            }
            return false;
        }

        if (bMp4H264 || bMp4HEVC) {
            if (pktFiltered->data) {
                av_packet_unref(pktFiltered);
            }
            if (!bsfc) {
                SetDemuxError("FFmpeg bitstream filter is not initialized for codec " +
                              std::string(avcodec_get_name(eVideoCodec)));
                av_packet_unref(pkt);
                return false;
            }

            const int send_ret = av_bsf_send_packet(bsfc, pkt);
            if (send_ret < 0) {
                SetDemuxError("FFmpeg bitstream filter send failed for codec " +
                              std::string(avcodec_get_name(eVideoCodec)) +
                              ", ret=" + std::to_string(send_ret));
                av_packet_unref(pkt);
                return false;
            }

            const int recv_ret = av_bsf_receive_packet(bsfc, pktFiltered);
            if (recv_ret < 0) {
                SetDemuxError("FFmpeg bitstream filter receive failed for codec " +
                              std::string(avcodec_get_name(eVideoCodec)) +
                              ", ret=" + std::to_string(recv_ret));
                return false;
            }
            if (!pktFiltered->data || pktFiltered->size <= 0) {
                SetDemuxError("FFmpeg bitstream filter produced an empty packet for codec " +
                              std::string(avcodec_get_name(eVideoCodec)));
                return false;
            }
            *ppVideo = pktFiltered->data;
            *pnVideoBytes = pktFiltered->size;
            if (pts) *pts = (int64_t)(pktFiltered->pts);
            if (flag) *flag = pktFiltered->flags;
        } else {
            if (bMp4MPEG4 && (frameCount == 0)) {
                int extraDataSize = fmtc->streams[iVideoStream]->codecpar->extradata_size;

                if (extraDataSize > 0) {
                    // extradata contains start codes 00 00 01. Subtract its size
                    pDataWithHeader = (uint8_t*)av_malloc(extraDataSize + pkt->size - 3 * sizeof(uint8_t));

                    if (!pDataWithHeader) {
                        LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
                        return false;
                    }

                    memcpy(pDataWithHeader, fmtc->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                    memcpy(pDataWithHeader + extraDataSize, pkt->data + 3, pkt->size - 3 * sizeof(uint8_t));

                    *ppVideo = pDataWithHeader;
                    *pnVideoBytes = extraDataSize + pkt->size - 3 * sizeof(uint8_t);
                }

            } else {
                *ppVideo = pkt->data;
                *pnVideoBytes = pkt->size;
            }

            if (pts) *pts = (int64_t)(pkt->pts);
            if (flag) *flag = pkt->flags;
        }

        frameCount++;

        return true;
    }

    bool Seek(SeekContext& seekCtx, uint8_t** ppVideo, int* pnVideoBytes) {
        /* !!! IMPORTANT !!!
         * Across this function packet decode timestamp (DTS) values are used to
         * compare given timestamp against. This is done for reason. DTS values shall
         * monotonically increase during the course of decoding unlike PTS velues
         * which may be affected by frame reordering due to B frames presence.
         */

        if (!is_seekable) {
            cerr << "Seek isn't supported for this input." << endl;
            return false;
        }

        if (IsVFR() && (BY_NUMBER == seekCtx.crit)) {
            cerr << "Can't seek by frame number in VFR sequences. Seek by timestamp "
                    "instead."
                 << endl;
            return false;
        }

        // Seek for single frame;
        auto seek_frame = [&](SeekContext const& seek_ctx, int flags) {
            bool seek_backward = false;
            int64_t timestamp = 0;
            int ret = 0;

            switch (seek_ctx.crit) {
                case BY_NUMBER:
                    timestamp = TsFromFrameNumber(seek_ctx.seek_frame);
                    ret = av_seek_frame(fmtc, iVideoStream, timestamp,
                                        seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
                    break;
                case BY_TIMESTAMP:
                    timestamp = TsFromTime(seek_ctx.seek_frame);
                    ret = av_seek_frame(fmtc, iVideoStream, timestamp,
                                        seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
                    break;
                default:
                    throw runtime_error("Invalid seek mode");
            }

            if (ret < 0) {
                throw runtime_error("Error seeking for frame: ");
            }
        };

        // Check if frame satisfies seek conditions;
        auto is_seek_done = [&](PacketData& pkt_data, SeekContext const& seek_ctx) {
            int64_t target_ts = 0;

            switch (seek_ctx.crit) {
                case BY_NUMBER:
                    target_ts = TsFromFrameNumber(seek_ctx.seek_frame);
                    break;
                case BY_TIMESTAMP:
                    target_ts = TsFromTime(seek_ctx.seek_frame);
                    break;
                default:
                    throw runtime_error("Invalid seek criteria");
                    break;
            }

            if (pkt_data.dts == target_ts) {
                return 0;
            } else if (pkt_data.dts > target_ts) {
                return 1;
            } else {
                return -1;
            };
        };

        auto demux_or_throw = [&](int64_t* out_pts) {
            bool ret = Demux(ppVideo, pnVideoBytes, out_pts);
            if (!ret) {
                throw runtime_error(HasDemuxError() ? GetLastDemuxError()
                                                    : "Demux returned no packet while seeking");
            }
        };

        /* This will seek for exact frame number;
         * Note that decoder may not be able to decode such frame; */
        auto seek_for_exact_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            // Repetititive seek until seek condition is satisfied;
            SeekContext tmp_ctx(seek_ctx.seek_frame, true);
            seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

            int condition = 0;
            do {
                demux_or_throw(&seek_ctx.out_frame_pts);
                condition = is_seek_done(pkt_data, seek_ctx);

                // We've gone too far and need to seek backwards;
                if (condition > 0) {
                    tmp_ctx.seek_frame--;
                    seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
                }
                // Need to read more frames until we reach requested number;
                else if (condition < 0) {
                    continue;
                }
            } while (0 != condition);

            // seek_ctx.out_frame_pts = pkt_data.pts;
            seek_ctx.out_frame_duration = pkt_data.duration;
        };

        /* This will seek to nearest I-frame;
         * Idea is to seek to N'th exact frame and rewing to nearest I-frame; */
        auto seek_for_nearest_iframe = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            int64_t time_in_miliseconds = seek_ctx.seek_frame;
            int64_t target_pts = time_in_miliseconds * (double)fmtc->streams[iVideoStream]->time_base.den /
                                 (fmtc->streams[iVideoStream]->time_base.num * 1000);

            int rv = av_seek_frame(fmtc, iVideoStream, target_pts, 0);
            if (rv < 0) throw std::runtime_error("Failed to seek");

            avcodec_flush_buffers(codecContext);
            demux_or_throw(&seek_ctx.out_frame_pts);
            seek_ctx.out_frame_duration = pktFiltered->duration;
        };

        // Seek for closest key frame in the past;
        auto seek_for_prev_key_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);

            demux_or_throw(&seek_ctx.out_frame_pts);
            // seek_ctx.out_frame_pts = pkt_data.pts;
            seek_ctx.out_frame_duration = pkt_data.duration;
        };

        PacketData pktData;
        pktData.bsl_data = size_t(*ppVideo);
        pktData.bsl = *pnVideoBytes;

        switch (seekCtx.mode) {
            case EXACT_FRAME:
                seek_for_exact_frame(pktData, seekCtx);
                break;
            case PREV_KEY_FRAME:
                seek_for_prev_key_frame(pktData, seekCtx);
                break;
            case NEAREST_FUTURE_KEY_FRAME:
                seek_for_nearest_iframe(pktData, seekCtx);
                break;
            default:
                throw runtime_error("Unsupported seek mode");
                break;
        }

        return true;
    }

    bool SeekWithTS(SeekContext& seekCtx, uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts) {
        /* !!! IMPORTANT !!!
         * Across this function packet decode timestamp (DTS) values are used to
         * compare given timestamp against. This is done for reason. DTS values
         * shall monotonically increase during the course of decoding unlike PTS
         * velues which may be affected by frame reordering due to B frames
         * presence.
         */

        if (!is_seekable) {
            cerr << "Seek isn't supported for this input." << endl;
            return false;
        }

        if (IsVFR() && (BY_NUMBER == seekCtx.crit)) {
            cerr << "Can't seek by frame number in VFR sequences. Seek by timestamp "
                    "instead."
                 << endl;
            return false;
        }
        // Seek for single frame;
        auto seek_frame = [&](SeekContext const& seek_ctx, int flags) {
            bool seek_backward = false;
            int64_t timestamp = 0;
            int ret = 0;

            switch (seek_ctx.crit) {
                case BY_TIMESTAMP:
                    timestamp = seek_ctx.seek_frame;
                    ret = av_seek_frame(fmtc, iVideoStream, timestamp,
                                        seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
                    break;
                default:
                    throw runtime_error("Invalid seek mode");
            }

            if (ret < 0) {
                throw runtime_error("Error seeking for frame: ");
            }
        };

        // Check if frame satisfies seek conditions;
        auto is_seek_done = [&](PacketData& pkt_data, SeekContext const& seek_ctx) {
            int64_t target_ts = 0;

            switch (seek_ctx.crit) {
                target_ts = TsFromTime(seek_ctx.seek_frame);
                break;
                default:
                    throw runtime_error("Invalid seek criteria");
                    break;
            }

            if (pkt_data.dts == target_ts) {
                return 0;
            } else if (pkt_data.dts > target_ts) {
                return 1;
            } else {
                return -1;
            };
        };

        auto demux_or_throw = [&](int64_t* out_pts) {
            bool ret = Demux(ppVideo, pnVideoBytes, out_pts);
            if (!ret) {
                throw runtime_error(HasDemuxError() ? GetLastDemuxError()
                                                    : "Demux returned no packet while seeking");
            }
        };

        /* This will seek to nearest I-frame;
     * Idea is to seek to N'th exact frame and rewing to nearest I-frame; */
        auto seek_for_nearest_iframe = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            int64_t target_pts = seek_ctx.seek_frame;

            int rv = av_seek_frame(fmtc, iVideoStream, target_pts, 0);
            if (rv < 0) throw std::runtime_error("Failed to seek");

            avcodec_flush_buffers(codecContext);
            demux_or_throw(&seek_ctx.out_frame_pts);
            seek_ctx.out_frame_duration = pktFiltered->duration;
        };

        // Seek for closest key frame in the past;
        auto seek_for_prev_key_frame = [&](PacketData& pkt_data, SeekContext& seek_ctx) {
            seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);

            demux_or_throw(&seek_ctx.out_frame_pts);
            // seek_ctx.out_frame_pts = pkt_data.pts;
            seek_ctx.out_frame_duration = pkt_data.duration;
        };

        PacketData pktData;
        pktData.bsl_data = size_t(*ppVideo);
        pktData.bsl = *pnVideoBytes;

        switch (seekCtx.mode) {
            case EXACT_FRAME:
                throw runtime_error("Current seeking with timestamp does not support EXACT_FRAME");
                break;
            case PREV_KEY_FRAME:
                seek_for_prev_key_frame(pktData, seekCtx);
                break;
            case NEAREST_FUTURE_KEY_FRAME:
                seek_for_nearest_iframe(pktData, seekCtx);
                break;
            default:
                throw runtime_error("Unsupported seek mode");
                break;
        }
        *pts = seekCtx.out_frame_pts;

        return true;
    }

    static int ReadPacket(void* opaque, uint8_t* pBuf, int nBuf) {
        return ((DataProvider*)opaque)->GetData(pBuf, nBuf);
    }
};

// #ifndef DEMUX_ONLY
inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
        case AV_CODEC_ID_MPEG1VIDEO:
            return cudaVideoCodec_MPEG1;
        case AV_CODEC_ID_MPEG2VIDEO:
            return cudaVideoCodec_MPEG2;
        case AV_CODEC_ID_MPEG4:
            return cudaVideoCodec_MPEG4;
        case AV_CODEC_ID_WMV3:
        case AV_CODEC_ID_VC1:
            return cudaVideoCodec_VC1;
        case AV_CODEC_ID_H264:
            return cudaVideoCodec_H264;
        case AV_CODEC_ID_HEVC:
            return cudaVideoCodec_HEVC;
        case AV_CODEC_ID_VP8:
            return cudaVideoCodec_VP8;
        case AV_CODEC_ID_VP9:
            return cudaVideoCodec_VP9;
        case AV_CODEC_ID_MJPEG:
            return cudaVideoCodec_JPEG;
        case AV_CODEC_ID_AV1:
            return cudaVideoCodec_AV1;
        default:
            return cudaVideoCodec_NumCodecs;
    }
}
// #endif
