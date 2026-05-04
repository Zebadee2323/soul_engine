#include "extractor_helpers.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace afex
{

constexpr double pi = 3.14159265358979323846;


thread_local AnalysisWorkspace* active_analysis_workspace = nullptr;

bool same_audio(const AudioData& left, const AudioData& right) {
    return left.samples.data() == right.samples.data() && left.samples.size() == right.samples.size() &&
           left.sample_rate_hz == right.sample_rate_hz && left.channel_count == right.channel_count;
}

bool same_pitch_settings(const PitchExtractorSettings& left, const PitchExtractorSettings& right) {
    return left.min_frequency_hz == right.min_frequency_hz && left.max_frequency_hz == right.max_frequency_hz &&
           left.confidence_threshold == right.confidence_threshold && left.max_analysis_seconds == right.max_analysis_seconds &&
           left.max_sample_rate_hz == right.max_sample_rate_hz && left.normalization.enabled == right.normalization.enabled &&
           left.normalization.minimum == right.normalization.minimum && left.normalization.maximum == right.normalization.maximum;
}

std::vector<double> compute_downmix_mono(const AudioData& audio) {
    const auto channel_count = audio.channel_count == 0 ? std::uint32_t{1} : audio.channel_count;
    const auto frame_count = audio.samples.size() / channel_count;
    auto mono = std::vector<double>(frame_count, 0.0);

    for (auto frame = std::size_t{0}; frame < frame_count; ++frame) {
        auto sum = 0.0;
        for (auto channel = std::uint32_t{0}; channel < channel_count; ++channel) {
            sum += static_cast<double>(audio.samples[frame * channel_count + channel]);
        }
        mono[frame] = sum / static_cast<double>(channel_count);
    }

    return mono;
}

std::vector<std::size_t> compute_frame_offsets(std::size_t sample_count, std::size_t frame_size, std::size_t hop_size) {
    auto offsets = std::vector<std::size_t>{};
    if (sample_count == 0) {
        return offsets;
    }

    if (sample_count <= frame_size) {
        offsets.push_back(0);
        return offsets;
    }

    for (auto offset = std::size_t{0}; offset + frame_size <= sample_count; offset += hop_size) {
        offsets.push_back(offset);
    }

    return offsets;
}

double parameter_or(const ExtractorParameters& parameters, std::string_view name, double fallback) {
    const auto found = parameters.find(std::string{name});
    if (found == parameters.end()) {
        return fallback;
    }

    return found->second;
}

std::size_t positive_size_parameter(const ExtractorParameters& parameters, std::string_view name, std::size_t fallback) {
    const auto value = parameter_or(parameters, name, static_cast<double>(fallback));
    if (value < 1.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument(std::string{name} + " must be greater than or equal to 1.");
#endif
        return fallback;
    }

    return static_cast<std::size_t>(std::llround(value));
}

double non_negative_parameter(const ExtractorParameters& parameters, std::string_view name, double fallback) {
    const auto value = parameter_or(parameters, name, fallback);
    if (value < 0.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument(std::string{name} + " must be greater than or equal to 0.");
#endif
        return fallback;
    }

    return value;
}

NormalizationSettings normalization_settings(const ExtractorParameters& parameters, double default_minimum, double default_maximum) {
    return NormalizationSettings{
        .enabled = parameter_or(parameters, "normalize", 0.0) != 0.0,
        .minimum = parameter_or(parameters, "normalization_min_hz", default_minimum),
        .maximum = parameter_or(parameters, "normalization_max_hz", default_maximum),
    };
}

NormalizationSettings normalization_settings(const ExtractorParameters& parameters, std::string_view min_name, std::string_view max_name,
                                             double default_minimum, double default_maximum) {
    return NormalizationSettings{
        .enabled = parameter_or(parameters, "normalize", 0.0) != 0.0,
        .minimum = parameter_or(parameters, min_name, default_minimum),
        .maximum = parameter_or(parameters, max_name, default_maximum),
    };
}

FrameParameterSettings frame_parameter_settings(const ExtractorParameters& parameters, std::size_t default_size, std::size_t default_hop) {
    return FrameParameterSettings{
        .size = positive_size_parameter(parameters, "frame_size", default_size),
        .hop = positive_size_parameter(parameters, "hop_size", default_hop),
        .size_ms = parameter_or(parameters, "frame_size_ms", 0.0),
        .hop_ms = parameter_or(parameters, "hop_size_ms", 0.0),
    };
}

bool normalization_range_valid(double minimum, double maximum) {
    return minimum < maximum;
}

double normalize_to_unit_interval(double value, double minimum, double maximum) {
    return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
}

void normalize_values_to_unit_interval(std::vector<double>& values, double minimum, double maximum) {
    for (auto& value : values) {
        value = normalize_to_unit_interval(value, minimum, maximum);
    }
}

AnalysisWorkspace::AnalysisWorkspace(const AudioData& audio) :
    m_audio                  (&audio)
{
}

bool AnalysisWorkspace::owns(const AudioData& audio) const {
    return m_audio != nullptr && same_audio(*m_audio, audio);
}

const std::vector<double>& AnalysisWorkspace::mono() {
    if (!m_mono_ready) {
        m_mono = compute_downmix_mono(*m_audio);
        m_mono_ready = true;
    }

    return m_mono;
}

const std::vector<std::size_t>& AnalysisWorkspace::offsets(std::size_t sample_count, std::size_t frame_size, std::size_t hop_size) {
    const auto found = std::find_if(m_offsets_caches.begin(), m_offsets_caches.end(), [sample_count, frame_size, hop_size](const OffsetsCache& cache) {
        return cache.sample_count == sample_count && cache.frame_size == frame_size && cache.hop_size == hop_size;
    });

    if (found != m_offsets_caches.end()) {
        return found->values;
    }

    auto cache = OffsetsCache{
        .sample_count = sample_count,
        .frame_size = frame_size,
        .hop_size = hop_size,
        .values = compute_frame_offsets(sample_count, frame_size, hop_size),
    };
    m_offsets_caches.push_back(std::move(cache));
    return m_offsets_caches.back().values;
}

const std::vector<double>& AnalysisWorkspace::spectrum(std::size_t offset, std::size_t frame_size) {
    const auto found = std::find_if(m_spectrum_caches.begin(), m_spectrum_caches.end(), [offset, frame_size](const SpectrumCache& cache) {
        return cache.offset == offset && cache.frame_size == frame_size;
    });

    if (found != m_spectrum_caches.end()) {
        return found->values;
    }

    auto cache = SpectrumCache{
        .offset = offset,
        .frame_size = frame_size,
        .values = magnitude_spectrum(mono(), offset, frame_size),
    };
    m_spectrum_caches.push_back(std::move(cache));
    return m_spectrum_caches.back().values;
}

const PitchSummary* AnalysisWorkspace::pitch(const PitchExtractorSettings& settings) const {
    if (!m_pitch_settings.has_value() || !m_pitch_summary.has_value() || !same_pitch_settings(*m_pitch_settings, settings)) {
        return nullptr;
    }

    return &(*m_pitch_summary);
}

void AnalysisWorkspace::set_pitch(const PitchExtractorSettings& settings, PitchSummary summary) {
    m_pitch_settings = settings;
    m_pitch_summary = std::move(summary);
}

ActiveAnalysisWorkspaceScope::ActiveAnalysisWorkspaceScope(AnalysisWorkspace& workspace) :
    m_previous              (active_analysis_workspace)
{
    active_analysis_workspace = &workspace;
}

ActiveAnalysisWorkspaceScope::~ActiveAnalysisWorkspaceScope() {
    active_analysis_workspace = m_previous;
}

std::vector<double> downmix_mono(const AudioData& audio) {
    return compute_downmix_mono(audio);
}

const std::vector<double>& mono_audio(const AudioData& audio) {
    if (active_analysis_workspace != nullptr && active_analysis_workspace->owns(audio)) {
        return active_analysis_workspace->mono();
    }

    static thread_local auto fallback = std::vector<double>{};
    fallback = compute_downmix_mono(audio);
    return fallback;
}

std::size_t samples_from_ms(std::uint32_t sample_rate_hz, double milliseconds, std::size_t fallback) {
    if (sample_rate_hz == 0 || milliseconds <= 0.0) {
        return fallback;
    }

    return std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(static_cast<double>(sample_rate_hz) * milliseconds / 1000.0)));
}

FrameSettings frame_settings(const AudioData& audio, const FrameParameterSettings& parameters) {
    auto size = parameters.size;
    auto hop = parameters.hop;

    if (parameters.size_ms > 0.0) {
        size = samples_from_ms(audio.sample_rate_hz, parameters.size_ms, size);
    }

    if (parameters.hop_ms > 0.0) {
        hop = samples_from_ms(audio.sample_rate_hz, parameters.hop_ms, hop);
    }

    size = std::min(size, current_analyze_settings().max_frame_size);
    hop = std::min(hop, size);
    return FrameSettings{.size = size, .hop = hop};
}

FrameSettings frame_settings(const AudioData& audio, const ExtractorParameters& parameters, std::size_t default_size, std::size_t default_hop) {
    return frame_settings(audio, frame_parameter_settings(parameters, default_size, default_hop));
}

bool frame_settings_valid(const FrameSettings& settings) {
    return settings.size >= 1 && settings.hop >= 1;
}

double frame_rms(const std::vector<double>& mono, std::size_t offset, std::size_t size) {
    auto square_sum = 0.0;
    for (auto i = std::size_t{0}; i < size && offset + i < mono.size(); ++i) {
        square_sum += mono[offset + i] * mono[offset + i];
    }

    return std::sqrt(square_sum / static_cast<double>(size));
}

std::vector<std::size_t> frame_offsets(std::size_t sample_count, std::size_t frame_size, std::size_t hop_size) {
    return compute_frame_offsets(sample_count, frame_size, hop_size);
}

const std::vector<std::size_t>& cached_frame_offsets(const AudioData& audio, std::size_t sample_count, std::size_t frame_size, std::size_t hop_size) {
    if (active_analysis_workspace != nullptr && active_analysis_workspace->owns(audio)) {
        return active_analysis_workspace->offsets(sample_count, frame_size, hop_size);
    }

    static thread_local auto fallback = std::vector<std::size_t>{};
    fallback = compute_frame_offsets(sample_count, frame_size, hop_size);
    return fallback;
}


std::vector<double> magnitude_spectrum(const std::vector<double>& mono, std::size_t offset, std::size_t frame_size) {
    if ((frame_size & (frame_size - 1)) == 0) {
        auto spectrum = std::vector<std::complex<double>>(frame_size);
        for (auto n = std::size_t{0}; n < frame_size; ++n) {
            const auto sample = offset + n < mono.size() ? mono[offset + n] : 0.0;
            const auto window = frame_size <= 1 ? 1.0 : 0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(n) / static_cast<double>(frame_size - 1));
            spectrum[n] = sample * window;
        }

        for (auto i = std::size_t{1}, j = std::size_t{0}; i < frame_size; ++i) {
            auto bit = frame_size >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;

            if (i < j) {
                std::swap(spectrum[i], spectrum[j]);
            }
        }

        for (auto length = std::size_t{2}; length <= frame_size; length <<= 1) {
            const auto angle = -2.0 * pi / static_cast<double>(length);
            const auto length_twiddle = std::complex<double>{std::cos(angle), std::sin(angle)};
            for (auto i = std::size_t{0}; i < frame_size; i += length) {
                auto twiddle = std::complex<double>{1.0, 0.0};
                for (auto j = std::size_t{0}; j < length / 2; ++j) {
                    const auto even = spectrum[i + j];
                    const auto odd = spectrum[i + j + length / 2] * twiddle;
                    spectrum[i + j] = even + odd;
                    spectrum[i + j + length / 2] = even - odd;
                    twiddle *= length_twiddle;
                }
            }
        }

        auto magnitudes = std::vector<double>(frame_size / 2 + 1, 0.0);
        for (auto bin = std::size_t{0}; bin < magnitudes.size(); ++bin) {
            magnitudes[bin] = std::abs(spectrum[bin]);
        }

        return magnitudes;
    }

    const auto bin_count = frame_size / 2 + 1;
    auto magnitudes = std::vector<double>(bin_count, 0.0);

    for (auto bin = std::size_t{0}; bin < bin_count; ++bin) {
        auto real = 0.0;
        auto imaginary = 0.0;
        for (auto n = std::size_t{0}; n < frame_size; ++n) {
            const auto sample = offset + n < mono.size() ? mono[offset + n] : 0.0;
            const auto window = frame_size <= 1 ? 1.0 : 0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(n) / static_cast<double>(frame_size - 1));
            const auto phase = 2.0 * pi * static_cast<double>(bin) * static_cast<double>(n) / static_cast<double>(frame_size);
            real += sample * window * std::cos(phase);
            imaginary -= sample * window * std::sin(phase);
        }
        magnitudes[bin] = std::sqrt(real * real + imaginary * imaginary);
    }

    return magnitudes;
}

const std::vector<double>& cached_magnitude_spectrum(const AudioData& audio, std::size_t offset, std::size_t frame_size) {
    if (active_analysis_workspace != nullptr && active_analysis_workspace->owns(audio)) {
        return active_analysis_workspace->spectrum(offset, frame_size);
    }

    static thread_local auto fallback = std::vector<double>{};
    fallback = magnitude_spectrum(mono_audio(audio), offset, frame_size);
    return fallback;
}

const PitchSummary* cached_pitch_summary(const AudioData& audio, const PitchExtractorSettings& settings) {
    if (active_analysis_workspace == nullptr || !active_analysis_workspace->owns(audio)) {
        return nullptr;
    }

    return active_analysis_workspace->pitch(settings);
}

void cache_pitch_summary(const AudioData& audio, const PitchExtractorSettings& settings, PitchSummary summary) {
    if (active_analysis_workspace != nullptr && active_analysis_workspace->owns(audio)) {
        active_analysis_workspace->set_pitch(settings, std::move(summary));
    }
}


}