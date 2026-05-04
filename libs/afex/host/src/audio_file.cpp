#include <afex/host.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <cmath>
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

std::uint16_t read_u16_le(std::istream& stream) {
    auto bytes = std::array<unsigned char, 2>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 16-bit value.");
    }

    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

std::uint32_t read_u32_le(std::istream& stream) {
    auto bytes = std::array<unsigned char, 4>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 32-bit value.");
    }

    return static_cast<std::uint32_t>(
        static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24)
    );
}

std::uint16_t read_u16_be(std::istream& stream) {
    auto bytes = std::array<unsigned char, 2>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 16-bit value.");
    }

    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(bytes[0]) << 8) | static_cast<std::uint16_t>(bytes[1]));
}

std::uint32_t read_u32_be(std::istream& stream) {
    auto bytes = std::array<unsigned char, 4>{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading a 32-bit value.");
    }

    return static_cast<std::uint32_t>(
        (static_cast<std::uint32_t>(bytes[0]) << 24)
        | (static_cast<std::uint32_t>(bytes[1]) << 16)
        | (static_cast<std::uint32_t>(bytes[2]) << 8)
        | static_cast<std::uint32_t>(bytes[3])
    );
}

std::string read_tag(std::istream& stream) {
    auto tag = std::array<char, 4>{};
    stream.read(tag.data(), tag.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while reading an audio chunk tag.");
    }

    return std::string{tag.data(), tag.size()};
}

void skip_bytes(std::istream& stream, std::uint64_t count) {
    stream.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    if (!stream) {
        throw std::runtime_error("Unexpected end of audio file while skipping chunk data.");
    }
}

void skip_chunk_payload(std::istream& stream, std::uint32_t count) {
    skip_bytes(stream, count + (count % 2));
}

std::uint32_t read_aiff_extended_sample_rate(std::istream& stream) {
    const auto sign_and_exponent = read_u16_be(stream);
    const auto exponent = static_cast<int>(sign_and_exponent & 0x7fff);
    const auto is_negative = (sign_and_exponent & 0x8000) != 0;
    const auto high_mantissa = read_u32_be(stream);
    const auto low_mantissa = read_u32_be(stream);

    if (exponent == 0 && high_mantissa == 0 && low_mantissa == 0) {
        return 0;
    }
    if (is_negative) {
        throw std::runtime_error("Invalid AIFF sample rate.");
    }

    const auto mantissa = static_cast<long double>(high_mantissa) * 4294967296.0L + static_cast<long double>(low_mantissa);
    const auto sample_rate = std::ldexp(mantissa, exponent - 16383 - 63);
    if (sample_rate <= 0.0L || sample_rate > static_cast<long double>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("Invalid AIFF sample rate.");
    }

    return static_cast<std::uint32_t>(std::llround(sample_rate));
}

float pcm_sample_to_float_le(const std::vector<unsigned char>& bytes, std::size_t offset, std::uint16_t bits_per_sample) {
    if (bits_per_sample == 8) {
        return (static_cast<float>(bytes[offset]) - 128.0f) / 128.0f;
    }

    if (bits_per_sample == 16) {
        const auto value = static_cast<std::int16_t>(static_cast<std::uint16_t>(bytes[offset]) | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8));
        return static_cast<float>(value) / 32768.0f;
    }

    if (bits_per_sample == 24) {
        auto value = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(bytes[offset])
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
        );
        if ((value & 0x00800000) != 0) {
            value |= static_cast<std::int32_t>(0xff000000);
        }

        return static_cast<float>(value) / 8388608.0f;
    }

    if (bits_per_sample == 32) {
        const auto value = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(bytes[offset])
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
            | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24)
        );
        return static_cast<float>(value) / 2147483648.0f;
    }

    throw std::runtime_error("Unsupported PCM WAV bit depth: " + std::to_string(bits_per_sample));
}

float pcm_sample_to_float_be(const std::vector<unsigned char>& bytes, std::size_t offset, std::uint16_t bits_per_sample) {
    if (bits_per_sample == 8) {
        const auto value = bytes[offset] < 128 ? static_cast<int>(bytes[offset]) : static_cast<int>(bytes[offset]) - 256;
        return static_cast<float>(value) / 128.0f;
    }

    if (bits_per_sample == 16) {
        const auto value = static_cast<std::int16_t>((static_cast<std::uint16_t>(bytes[offset]) << 8) | static_cast<std::uint16_t>(bytes[offset + 1]));
        return static_cast<float>(value) / 32768.0f;
    }

    if (bits_per_sample == 24) {
        auto value = static_cast<std::int32_t>(
            (static_cast<std::uint32_t>(bytes[offset]) << 16)
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
            | static_cast<std::uint32_t>(bytes[offset + 2])
        );
        if ((value & 0x00800000) != 0) {
            value |= static_cast<std::int32_t>(0xff000000);
        }

        return static_cast<float>(value) / 8388608.0f;
    }

    if (bits_per_sample == 32) {
        const auto value = static_cast<std::int32_t>(
            (static_cast<std::uint32_t>(bytes[offset]) << 24)
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 16)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 8)
            | static_cast<std::uint32_t>(bytes[offset + 3])
        );
        return static_cast<float>(value) / 2147483648.0f;
    }

    throw std::runtime_error("Unsupported AIFF PCM bit depth: " + std::to_string(bits_per_sample));
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

LoadedAudioData load_riff_wave_file(std::istream& stream) {
    static_cast<void>(read_u32_le(stream));
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
        const auto chunk_size = read_u32_le(stream);

        if (chunk_id == "fmt ") {
            audio_format = read_u16_le(stream);
            channel_count = read_u16_le(stream);
            sample_rate_hz = read_u32_le(stream);
            static_cast<void>(read_u32_le(stream));
            static_cast<void>(read_u16_le(stream));
            bits_per_sample = read_u16_le(stream);
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
        samples.push_back(
            audio_format == 3
                ? float_sample_to_float(data_bytes, offset, bits_per_sample)
                : pcm_sample_to_float_le(data_bytes, offset, bits_per_sample)
        );
    }

    return LoadedAudioData{
        .samples = std::move(samples),
        .sample_rate_hz = sample_rate_hz,
        .channel_count = channel_count,
    };
}

LoadedAudioData load_aiff_file(std::istream& stream) {
    static_cast<void>(read_u32_be(stream));
    const auto form_type = read_tag(stream);
    if (form_type != "AIFF" && form_type != "AIFC") {
        throw std::runtime_error("Unsupported audio file: expected an AIFF file.");
    }

    auto channel_count = std::uint16_t{0};
    auto sample_frame_count = std::uint32_t{0};
    auto sample_rate_hz = std::uint32_t{0};
    auto bits_per_sample = std::uint16_t{0};
    auto is_little_endian_pcm = false;
    auto data_bytes = std::vector<unsigned char>{};

    while (stream.peek() != std::ifstream::traits_type::eof()) {
        const auto chunk_id = read_tag(stream);
        const auto chunk_size = read_u32_be(stream);

        if (chunk_id == "COMM") {
            if (chunk_size < 18) {
                throw std::runtime_error("Invalid or incomplete AIFF COMM chunk.");
            }

            channel_count = read_u16_be(stream);
            sample_frame_count = read_u32_be(stream);
            bits_per_sample = read_u16_be(stream);
            sample_rate_hz = read_aiff_extended_sample_rate(stream);

            auto consumed_bytes = std::uint32_t{18};
            if (form_type == "AIFC") {
                if (chunk_size < 22) {
                    throw std::runtime_error("Invalid or incomplete AIFC COMM chunk.");
                }

                const auto compression_type = read_tag(stream);
                consumed_bytes += 4;
                if (compression_type == "NONE" || compression_type == "twos") {
                    is_little_endian_pcm = false;
                } else if (compression_type == "sowt") {
                    is_little_endian_pcm = true;
                } else {
                    throw std::runtime_error("Unsupported AIFC encoding. Only uncompressed PCM AIFF/AIFC files are supported.");
                }
            }

            if (chunk_size > consumed_bytes) {
                skip_chunk_payload(stream, chunk_size - consumed_bytes);
            }
        } else if (chunk_id == "SSND") {
            if (chunk_size < 8) {
                throw std::runtime_error("Invalid or incomplete AIFF SSND chunk.");
            }

            const auto offset = read_u32_be(stream);
            static_cast<void>(read_u32_be(stream));
            const auto payload_size = chunk_size - 8;
            if (offset > payload_size) {
                throw std::runtime_error("Invalid AIFF SSND offset.");
            }

            skip_bytes(stream, offset);
            data_bytes.resize(payload_size - offset);
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

    if (channel_count == 0 || sample_rate_hz == 0 || bits_per_sample == 0 || data_bytes.empty()) {
        throw std::runtime_error("Invalid or incomplete AIFF file.");
    }

    const auto bytes_per_sample = static_cast<std::size_t>(bits_per_sample / 8);
    const auto expected_byte_count = static_cast<std::size_t>(sample_frame_count) * channel_count * bytes_per_sample;
    if (bytes_per_sample == 0 || data_bytes.size() % bytes_per_sample != 0 || (expected_byte_count > 0 && data_bytes.size() < expected_byte_count)) {
        throw std::runtime_error("Invalid AIFF sample data size.");
    }
    if (expected_byte_count > 0 && data_bytes.size() > expected_byte_count) {
        data_bytes.resize(expected_byte_count);
    }

    auto samples = std::vector<float>{};
    samples.reserve(data_bytes.size() / bytes_per_sample);
    for (auto offset = std::size_t{0}; offset < data_bytes.size(); offset += bytes_per_sample) {
        samples.push_back(
            is_little_endian_pcm
                ? pcm_sample_to_float_le(data_bytes, offset, bits_per_sample)
                : pcm_sample_to_float_be(data_bytes, offset, bits_per_sample)
        );
    }

    return LoadedAudioData{
        .samples = std::move(samples),
        .sample_rate_hz = sample_rate_hz,
        .channel_count = channel_count,
    };
}

}

AudioData LoadedAudioData::view() const {
    return AudioData{
        .samples = samples,
        .sample_rate_hz = sample_rate_hz,
        .channel_count = channel_count,
    };
}

LoadedAudioData load_audio_file(std::string_view audio_file_path) {
    auto stream = std::ifstream{std::string{audio_file_path}, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("Unable to open audio file: " + std::string{audio_file_path});
    }

    const auto container_tag = read_tag(stream);
    if (container_tag == "RIFF") {
        return load_riff_wave_file(stream);
    }
    if (container_tag == "FORM") {
        return load_aiff_file(stream);
    }

    throw std::runtime_error("Unsupported audio file: expected a RIFF/WAVE or AIFF file.");
}

Status load_audio_file(std::string_view audio_file_path, LoadedAudioData& audio) {
#if AFEX_ENABLE_EXCEPTIONS
    try {
        audio = load_audio_file(audio_file_path);
        return {};
    } catch (const std::invalid_argument& error) {
        return Status{.error = AfexError::InvalidArgument, .message = error.what()};
    } catch (const std::runtime_error& error) {
        return Status{.error = AfexError::UnsupportedAudioFile, .message = error.what()};
    }
#else
    static_cast<void>(audio_file_path);
    static_cast<void>(audio);
    return Status{.error = AfexError::UnsupportedAudioFile, .message = "Host audio file loading requires AFEX_ENABLE_EXCEPTIONS in this build."};
#endif
}

}
