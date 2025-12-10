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
#ifndef GOPDECODERUTILS_H
#define GOPDECODERUTILS_H
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include "FFmpegDemuxer.h"
namespace {
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    for (const auto& elem : v) {
        os << elem << " ";
    }
    return os;
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::pair<K, V>& p) {
    return os << '(' << p.first << ", " << p.second << ')';
}

void writeVectorToFile(const std::vector<int>& vec, std::ofstream& outFile) {
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for writing");
    }

    std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(outFile, "\t"));
    outFile << "\n";
}

template <typename K, typename V>
void writeMapToFile(const std::map<K, V>& map, std::ofstream& outFile) {
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for writing");
    }

    /*std::copy(map.begin(), map.end(),
      std::ostream_iterator<std::pair<K, V>>(outFile, "\t"));*/
    std::for_each(std::begin(map), std::end(map), [&outFile](const std::pair<K, V>& element) {
        outFile << element.first << "\t" << element.second << "\t";
    });
    outFile << "\n";
}

template <typename K, typename V>
std::map<K, V> readMapFromFile(std::ifstream& inFile) {
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for reading");
    }

    std::map<K, V> result;
    std::string line;
    std::getline(inFile, line);
    std::istringstream iss(line);

    std::pair<K, V> item;
    while (iss >> item.first >> item.second) {
        result.insert(item);
    }
    return result;
}

std::vector<int> readIntegersFromFile(std::ifstream& inFile) {
    std::vector<int> numbers;

    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << std::endl;
        return numbers;
    }

    std::string line;
    std::getline(inFile, line);
    std::istringstream iss(line);
    int number;
    while (iss >> number) {
        numbers.push_back(number);
        iss.ignore(1, '\t');
    }

    return numbers;
}

bool hasDuplicates(const std::vector<int>& vec) {
    std::unordered_set<int> seen;
    for (int num : vec) {
        if (seen.find(num) != seen.end()) {
            return true;
        }
        seen.insert(num);
    }
    return false;
}

template <typename Key, typename Value>
class LFUCache {
    int capacity;
    int minFreq;
    // stores the key-value-freq. the key as the first element of the map.
    // second element of the map is a pair, the first element of the pairing is
    // actually value, second refers to the frequency of the key-value pairing;
    std::unordered_map<Key, pair<Value, int>> keyVal;
    // stores the keys in a list, with the key's frequency as the index of the
    // list.
    std::unordered_map<int, std::list<Key>> freqLists;
    // stores the position of each key in the freqList.
    std::unordered_map<Key, typename std::list<Key>::iterator> pos;

   public:
    LFUCache(int capacity) {
        this->capacity = capacity;
        minFreq = 0;
    }

    Value get(const Key& key) {
        // If the key is not found in the keyVal map, return nullptr
        if (keyVal.find(key) == keyVal.end()) {
            return nullptr;
        }

        // The key is removed from the list at index equal to its current frequency
        // in the freqLists map.
        freqLists[keyVal[key].second].erase(pos[key]);

        // The key's frequency in keyVal is incremented by 1.
        keyVal[key].second++;

        // The key is added to the list at index equal to its new frequency in the
        // freqLists map.
        freqLists[keyVal[key].second].push_back(key);

        // The key's position in the freqList map is updated in the pos map.
        pos[key] = --freqLists[keyVal[key].second].end();

        // If the list at index equal to the current minimum frequency is empty, the
        // minimum frequency is incremented by 1.
        if (freqLists[minFreq].empty()) minFreq++;

        // The value associated with the key is returned.
        return keyVal[key].first;
    }

    Value put(const Key& key, const Value& value) {
        // If the capacity is 0, the function returns immediately.
        if (!capacity) return nullptr;

        // If the key already exists in the keyVal map
        if (keyVal.find(key) != keyVal.end()) {
            // The key's value is updated in the keyVal map.
            keyVal[key].first = value;

            // The key is removed from the list at index equal to its current
            // frequency in the freqList map.
            freqLists[keyVal[key].second].erase(pos[key]);

            // The key's frequency in keyVal is incremented by 1.
            keyVal[key].second++;

            // The key is added to the list at index equal to its new frequency in the
            // freqList map.
            freqLists[keyVal[key].second].push_back(key);

            // The key's position in the freqList map is updated in the pos map.
            pos[key] = --freqLists[keyVal[key].second].end();

            // If the list at index equal to the current minimum frequency is empty,
            // the minimum frequency is incremented by 1.
            if (freqLists[minFreq].empty()) minFreq++;
            return nullptr;
        }

        // If the key does not already exist in the keyVal map and the size of
        // keyVal is equal to the capacity, the following steps are taken:
        if (keyVal.size() == capacity) {
            // The key at the front of the list at index equal to the current minimum
            // frequency in the freqList map is removed from all three maps
            Key delKey = freqLists[minFreq].front();
            keyVal.erase(delKey);
            pos.erase(delKey);
            return freqLists[minFreq].pop_front();
        }
        // A new key-value pair is added to keyVal, with a frequency of 1
        keyVal[key] = {value, 1};
        // The key is added to the list at index 1 in the freqList map.
        freqLists[1].push_back(key);
        // The key's position in the freqList map is updated in the pos map.
        pos[key] = --freqLists[1].end();

        // The minimum frequency is set to 1.
        minFreq = 1;
        return nullptr;
    }

    int getSize() { return keyVal.size(); }
};

static void SavePacketBufferToFile(const uint8_t* packet_buffer, int nVideoBytes, int frame_id) {
    std::string filename = "packet_" + std::to_string(frame_id) + ".bin";
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(packet_buffer), nVideoBytes);
        outfile.close();
    }
}

/**
 * @brief Check if a video packet represents a key frame
 * @param codec_id FFmpeg codec ID (AVCodecID enum value)
 * @param pVideo Pointer to video packet data
 * @param demux_flags Demuxer flags (should contain AV_PKT_FLAG_KEY for key frames)
 * @return true if the packet is a key frame, false otherwise
 */
inline bool iskeyFrame(AVCodecID codec_id, const uint8_t* pVideo, int demux_flags) {
    if (!pVideo) {
        return false;
    }

    bool bPS = false;
    if (codec_id == AV_CODEC_ID_HEVC) {
        uint8_t b = pVideo[2] == 1 ? pVideo[3] : pVideo[4];
        int nal_unit_type = b >> 1;
        if (nal_unit_type == 32 || nal_unit_type == 33 || nal_unit_type == 34 || nal_unit_type == 39 ||
            nal_unit_type == 40) {
            bPS = true;
        }
    } else if (codec_id == AV_CODEC_ID_H264) {
        uint8_t b = pVideo[2] == 1 ? pVideo[3] : pVideo[4];
        int nal_ref_idc = b >> 5;
        int nal_unit_type = b & 0x1f;
        if (nal_unit_type == 6 || nal_unit_type == 7 || nal_unit_type == 8 || nal_unit_type == 9) {
            bPS = true;
        }
    } else {
        throw std::domain_error("[ERROR] Unsupported video codec: " + std::to_string(codec_id));
    }

    return (demux_flags & AV_PKT_FLAG_KEY) && bPS;
}

/**
 * @brief Check if a video packet has key frame NAL unit type (without checking flags)
 * @param codec_id FFmpeg codec ID (AVCodecID enum value)
 * @param pVideo Pointer to video packet data
 * @return true if the packet has key frame NAL unit type, false otherwise
 */
inline bool hasKeyFrameNalType(AVCodecID codec_id, const uint8_t* pVideo) {
    if (!pVideo) {
        return false;
    }

    if (codec_id == AV_CODEC_ID_HEVC) {
        uint8_t b = pVideo[2] == 1 ? pVideo[3] : pVideo[4];
        int nal_unit_type = b >> 1;
        return (nal_unit_type == 32 || nal_unit_type == 33 || nal_unit_type == 34 || nal_unit_type == 39 ||
                nal_unit_type == 40);
    } else if (codec_id == AV_CODEC_ID_H264) {
        uint8_t b = pVideo[2] == 1 ? pVideo[3] : pVideo[4];
        int nal_unit_type = b & 0x1f;
        return (nal_unit_type == 6 || nal_unit_type == 7 || nal_unit_type == 8 || nal_unit_type == 9);
    } else {
        throw std::domain_error("[ERROR] Unsupported video codec: " + std::to_string(codec_id));
    }
}

/**
 * @brief Parse GOP start indices from a video demuxer
    If a frame is a I-frame and key_frame in the same time, then the frame is the
    start of a new GOP The keyframe is the frame which has flag AV_FRAME_FLAG_KEY
    (1 << 1), if open_gop is false, each keyframe is a IDR picture
    For video with open_gop is true, to parse NAL unit to get IDR picture id, The
    keyframe is the frame which has flag AV_FRAME_FLAG_KEY is a Recovery Point. The
    recovery point SEI message assists a decoder in determining when the decoding
    process will produce acceptable pictures for display after the decoder initiates
    random access or after the encoder indicates a broken link in the coded video
    sequence.
 * @param demuxer Pointer to FFmpegDemuxer instance
 * @param frame2pts Reference to map storing frame index to PTS mapping
 * @param pts2frame Reference to map storing PTS to frame index mapping
 * @param isVFR Whether the video is Variable Frame Rate
 * @return Vector of GOP start frame indices
 */
inline std::vector<int> parse_gop_start_idx(FFmpegDemuxer* demuxer, std::map<int, int64_t>& frame2pts,
                                            std::map<int64_t, int>& pts2frame, bool isVFR) {
    std::vector<int> gop_start_idx;
    std::vector<std::pair<int64_t, bool>> pts_keyFrame_pair;
    int nVideoBytes = 0, flags = 0;
    uint8_t* pVideo = NULL;
    int64_t timestamp;
    int frame_cnt = 0;

    do {
        auto ret = demuxer->Demux(&pVideo, &nVideoBytes, &timestamp, &flags);
        ++frame_cnt;

        if (nVideoBytes) {
            if (!ret) {
                throw std::invalid_argument("[ERROR] Demux error");
            }

            bool is_key_frame = iskeyFrame(demuxer->GetVideoCodec(), pVideo, flags);

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

        if (isVFR) {
            frame2pts[current_frame_idx] = current_timestamp;
            pts2frame[current_timestamp] = current_frame_idx;
        }

        if (isKeyFrame) {
            gop_start_idx.push_back(current_frame_idx);
        }
    }
    gop_start_idx.push_back(current_frame_idx);

    if (gop_start_idx.size() <= 0) {
        throw std::out_of_range("[ERROR] The video must have at least one GOP");
    }

    return gop_start_idx;
}

/**
 * @brief Parse GOP lengths from GOP start indices and sorted frame IDs
 * @param gop_start_id_list Vector of GOP start frame indices
 * @param sorted_frame_ids Vector of sorted frame IDs
 * @param first_frame_ids Reference to vector to store first frame IDs of each GOP
 * @return Vector of GOP lengths
 */
inline std::vector<int> parse_gop_length(const std::vector<int>& gop_start_id_list,
                                         const std::vector<int>& sorted_frame_ids,
                                         std::vector<int>& first_frame_ids) {
    if (sorted_frame_ids.back() >= gop_start_id_list.back()) {
        throw std::out_of_range(
            "[ERROR] End frame of last GOP : " + std::to_string(gop_start_id_list.back()) +
            " must be behind the frame_ids : " + std::to_string(sorted_frame_ids.back()));
    }

    std::vector<int> gop_length;
    for (auto iter = sorted_frame_ids.begin(); iter != sorted_frame_ids.end();) {
        auto next_key_it = std::upper_bound(gop_start_id_list.begin(), gop_start_id_list.end(), *iter);
        if (next_key_it == gop_start_id_list.begin()) {
            throw std::out_of_range("[ERROR] Can not find a gop for frame: " + std::to_string(*iter) +
                                    " only with next gop_start_id: " + std::to_string(*next_key_it));
        }
        gop_length.push_back(*(next_key_it) - *(next_key_it - 1));
        // first_frame_ids.push_back(*iter);
        first_frame_ids.push_back(*(next_key_it - 1));

        iter = std::lower_bound(sorted_frame_ids.begin(), sorted_frame_ids.end(), *next_key_it);
    }
    return gop_length;
}

}  // namespace
#endif
