#include <afex/afex.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace afex
{

namespace
{

std::uint16_t read_u16(std::istream& stream) {
    auto bytes = std::array<unsigned char, 2>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 16-bit value.");
    }

    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t read_u32(std::istream& stream) {
    auto bytes = std::array<unsigned char, 4>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 32-bit value.");
    }

    return static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

std::string read_tag(std::istream& stream) {
    auto tag = std::array<char, 4>{};
    stream.read(tag.data(), tag.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a WAV chunk tag.");
    }

    return std::string{tag.data(), tag.size()};
}

void skip_bytes(std::istream& stream, std::uint32_t count) {
    stream.seekg(count, std::ios::cur);
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while skipping WAV data.");
    }
}

void skip_chunk_payload(std::istream& stream, std::uint32_t count) {
    skip_bytes(stream, count + (count % 2));
}

float pcm_sample_to_float(const std::vector<unsigned char>& bytes, std::size_t offset, std::uint16_t bits_per_sample) {
    if (bits_per_sample == 8) {
        return (static_cast<float>(bytes[offset]) - 128.0f) / 128.0f;
    }

    if (bits_per_sample == 16) {
        const auto value = static_cast<std::int16_t>(bytes[offset] | (bytes[offset + 1] << 8));
        return static_cast<float>(value) / 32768.0f;
    }

    if (bits_per_sample == 24) {
        auto value = static_cast<std::int32_t>(bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16));
        if ((value & 0x00800000) != 0) {
            value |= static_cast<std::int32_t>(0xff000000);
        }

        return static_cast<float>(value) / 8388608.0f;
    }

    if (bits_per_sample == 32) {
        const auto value = static_cast<std::int32_t>(bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24));
        return static_cast<float>(value) / 2147483648.0f;
    }

    throw std::runtime_error("Unsupported PCM WAV bit depth: " + std::to_string(bits_per_sample));
}

float float_sample_to_float(const std::vector<unsigned char>& bytes, std::size_t offset, std::uint16_t bits_per_sample) {
    if (bits_per_sample != 32) {
        throw std::runtime_error("Only 32-bit IEEE float WAV files are supported.");
    }

    auto value = float{};
    static_assert(sizeof(value) == 4);
    std::copy(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.begin() + static_cast<std::ptrdiff_t>(offset + 4), reinterpret_cast<unsigned char*>(&value));
    return value;
}

}

LoadedAudioData load_audio_file(std::string_view audio_file_path) {
    auto stream = std::ifstream{std::string{audio_file_path}, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("Unable to open audio file: " + std::string{audio_file_path});
    }

    if (read_tag(stream) != "RIFF") {
        throw std::runtime_error("Unsupported audio file: expected a RIFF WAV file.");
    }

    static_cast<void>(read_u32(stream));
    if (read_tag(stream) != "WAVE") {
        throw std::runtime_error("Unsupported audio file: expected a WAVE file.");
    }

    auto audio_format = std::uint16_t{0};
    auto channel_count = std::uint16_t{0};
    auto sample_rate_hz = std::uint32_t{0};
    auto bits_per_sample = std::uint16_t{0};
    auto data_bytes = std::vector<unsigned char>{};

    while (stream.peek() != std::ifstream::traits_type::eof()) {
        const auto chunk_id = read_tag(stream);
        const auto chunk_size = read_u32(stream);

        if (chunk_id == "fmt ") {
            audio_format = read_u16(stream);
            channel_count = read_u16(stream);
            sample_rate_hz = read_u32(stream);
            static_cast<void>(read_u32(stream));
            static_cast<void>(read_u16(stream));
            bits_per_sample = read_u16(stream);
            if (chunk_size > 16) {
                skip_chunk_payload(stream, chunk_size - 16);
            }
        } else if (chunk_id == "data") {
            data_bytes.resize(chunk_size);
            stream.read(reinterpret_cast<char*>(data_bytes.data()), data_bytes.size());
            if (!stream) {
                throw std::runtime_error("Unexpected end of audio file while reading sample data.");
            }
            if (chunk_size % 2 != 0) {
                skip_bytes(stream, 1);
            }
        } else {
            skip_chunk_payload(stream, chunk_size);
        }
    }

    if (audio_format != 1 && audio_format != 3) {
        throw std::runtime_error("Unsupported WAV encoding. Only PCM and 32-bit IEEE float WAV files are supported.");
    }

    if (channel_count == 0 || sample_rate_hz == 0 || bits_per_sample == 0 || data_bytes.empty()) {
        throw std::runtime_error("Invalid or incomplete WAV file.");
    }

    const auto bytes_per_sample = static_cast<std::size_t>(bits_per_sample / 8);
    if (bytes_per_sample == 0 || data_bytes.size() % bytes_per_sample != 0) {
        throw std::runtime_error("Invalid WAV sample data size.");
    }

    auto samples = std::vector<float>{};
    samples.reserve(data_bytes.size() / bytes_per_sample);
    for (auto offset = std::size_t{0}; offset < data_bytes.size(); offset += bytes_per_sample) {
        samples.push_back(audio_format == 3 ? float_sample_to_float(data_bytes, offset, bits_per_sample) : pcm_sample_to_float(data_bytes, offset, bits_per_sample));
    }

    return LoadedAudioData{
        .samples = std::move(samples),
        .sample_rate_hz = sample_rate_hz,
        .channel_count = channel_count,
    };
}

}
