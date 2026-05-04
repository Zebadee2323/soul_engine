#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace afex
{

namespace
{

constexpr double pi = 3.14159265358979323846;

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

std::vector<double> downmix_mono(const AudioData& audio) {
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

std::size_t samples_from_ms(std::uint32_t sample_rate_hz, double milliseconds, std::size_t fallback) {
    if (sample_rate_hz == 0 || milliseconds <= 0.0) {
        return fallback;
    }

    return std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(static_cast<double>(sample_rate_hz) * milliseconds / 1000.0)));
}

struct FrameSettings
{
    std::size_t                      size;
    std::size_t                      hop;
};

FrameSettings frame_settings(const AudioData& audio, const ExtractorParameters& parameters, std::size_t default_size, std::size_t default_hop) {
    auto size = positive_size_parameter(parameters, "frame_size", default_size);
    auto hop = positive_size_parameter(parameters, "hop_size", default_hop);

    const auto size_ms = parameter_or(parameters, "frame_size_ms", 0.0);
    if (size_ms > 0.0) {
        size = samples_from_ms(audio.sample_rate_hz, size_ms, size);
    }

    const auto hop_ms = parameter_or(parameters, "hop_size_ms", 0.0);
    if (hop_ms > 0.0) {
        hop = samples_from_ms(audio.sample_rate_hz, hop_ms, hop);
    }

    size = std::min(size, current_analyze_settings().max_frame_size);
    hop = std::min(hop, size);
    return FrameSettings{.size = size, .hop = hop};
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

double mean_or_zero(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

struct PitchSummary
{
    double                           frequency_hz                = 0.0;
    double                           confidence                  = 0.0;
#if !AFEX_EMBEDDED
    std::string                      note;
#endif
};

PitchSummary pitch_summary(double frequency_hz, double confidence, std::string_view note) {
    return PitchSummary{
        .frequency_hz = frequency_hz,
        .confidence = confidence,
#if !AFEX_EMBEDDED
        .note = std::string{note},
#endif
    };
}

PitchSummary estimate_pitch(const AudioData& audio, const ExtractorParameters& parameters) {
    if (audio.sample_rate_hz == 0) {
        return pitch_summary(0.0, 0.0, "Pitch requires a non-zero sample rate.");
    }

    const auto mono = downmix_mono(audio);
    if (mono.size() < 3) {
        return pitch_summary(0.0, 0.0, "At least three frames are needed for pitch estimation.");
    }

    const auto min_frequency_hz = parameter_or(parameters, "min_frequency_hz", 50.0);
    const auto max_frequency_hz = parameter_or(parameters, "max_frequency_hz", 500.0);
    const auto confidence_threshold = parameter_or(parameters, "confidence_threshold", 0.3);
    if (min_frequency_hz <= 0.0 || max_frequency_hz <= 0.0 || min_frequency_hz >= max_frequency_hz) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("pitch min_frequency_hz and max_frequency_hz must be positive and increasing.");
#endif
        return pitch_summary(0.0, 0.0, "pitch min_frequency_hz and max_frequency_hz must be positive and increasing.");
    }
    if (confidence_threshold < 0.0 || confidence_threshold > 1.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("pitch confidence_threshold must be between 0 and 1.");
#endif
        return pitch_summary(0.0, 0.0, "pitch confidence_threshold must be between 0 and 1.");
    }

    auto min_lag = static_cast<std::size_t>(std::floor(static_cast<double>(audio.sample_rate_hz) / max_frequency_hz));
    auto max_lag = static_cast<std::size_t>(std::ceil(static_cast<double>(audio.sample_rate_hz) / min_frequency_hz));
    min_lag = std::max<std::size_t>(1, min_lag);
    max_lag = std::min<std::size_t>(max_lag, mono.size() - 1);
    if (min_lag > max_lag) {
        return pitch_summary(0.0, 0.0, "Audio is too short for the configured pitch frequency range.");
    }

    auto best_lag = min_lag;
    auto best_correlation = -std::numeric_limits<double>::infinity();
    for (auto lag = min_lag; lag <= max_lag; ++lag) {
        auto cross = 0.0;
        auto left_energy = 0.0;
        auto right_energy = 0.0;
        for (auto i = std::size_t{0}; i + lag < mono.size(); ++i) {
            cross += mono[i] * mono[i + lag];
            left_energy += mono[i] * mono[i];
            right_energy += mono[i + lag] * mono[i + lag];
        }

        const auto denominator = std::sqrt(left_energy * right_energy);
        const auto correlation = denominator == 0.0 ? 0.0 : cross / denominator;
        if (correlation > best_correlation) {
            best_correlation = correlation;
            best_lag = lag;
        }
    }

    const auto confidence = std::clamp(best_correlation, 0.0, 1.0);
    if (confidence < confidence_threshold) {
        return pitch_summary(0.0, confidence, "No pitch exceeded confidence_threshold.");
    }

    return pitch_summary(static_cast<double>(audio.sample_rate_hz) / static_cast<double>(best_lag), confidence,
                         "Autocorrelation estimate over downmixed mono audio.");
}

}

FeatureResult extract_pitch(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto pitch = estimate_pitch(audio, parameters);
#if AFEX_EMBEDDED
    return complete_feature(feature_names::pitch, pitch.frequency_hz, {}, "Hz");
#else
    return complete_feature(feature_names::pitch, pitch.frequency_hz, {}, "Hz", pitch.note);
#endif
}

FeatureResult extract_pitch_confidence(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto pitch = estimate_pitch(audio, parameters);
#if AFEX_EMBEDDED
    return complete_feature(feature_names::pitch_confidence, pitch.confidence, {}, "ratio");
#else
    return complete_feature(feature_names::pitch_confidence, pitch.confidence, {}, "ratio", pitch.note);
#endif
}

FeatureResult extract_spectral_centroid(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto mono = downmix_mono(audio);
    if (mono.empty() || audio.sample_rate_hz == 0) {
        return complete_feature(feature_names::spectral_centroid, 0.0, {}, "Hz", "Spectral centroid requires samples and a non-zero sample rate.");
    }

    const auto settings = frame_settings(audio, parameters, 2048, 512);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::spectral_centroid, "Frame settings are invalid.");
    }
    auto values = std::vector<double>{};
    auto sum_centroid = 0.0;
    auto centroid_count = std::size_t{0};
    for (const auto offset : frame_offsets(mono.size(), settings.size, settings.hop)) {
        const auto magnitudes = magnitude_spectrum(mono, offset, settings.size);
        auto weighted_sum = 0.0;
        auto magnitude_sum = 0.0;
        for (auto bin = std::size_t{0}; bin < magnitudes.size(); ++bin) {
            const auto frequency_hz = static_cast<double>(bin) * static_cast<double>(audio.sample_rate_hz) / static_cast<double>(settings.size);
            weighted_sum += frequency_hz * magnitudes[bin];
            magnitude_sum += magnitudes[bin];
        }
        const auto centroid = magnitude_sum == 0.0 ? 0.0 : weighted_sum / magnitude_sum;
        sum_centroid += centroid;
        ++centroid_count;
        if (current_analyze_settings().keep_intermediate_values) {
            values.push_back(centroid);
        }
    }

    const auto value = centroid_count == 0 ? 0.0 : sum_centroid / static_cast<double>(centroid_count);
    return complete_feature(feature_names::spectral_centroid, value, std::move(values), "Hz", "Mean spectral centroid across Hann-windowed frames.");
}

FeatureResult extract_spectral_flux(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto mono = downmix_mono(audio);
    if (mono.empty()) {
        return complete_feature(feature_names::spectral_flux, 0.0, {}, "magnitude", "No samples were supplied.");
    }

    const auto settings = frame_settings(audio, parameters, 2048, 512);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::spectral_flux, "Frame settings are invalid.");
    }
    auto values = std::vector<double>{};
    auto sum_flux = 0.0;
    auto flux_count = std::size_t{0};
    auto previous = std::vector<double>{};
    for (const auto offset : frame_offsets(mono.size(), settings.size, settings.hop)) {
        auto current = magnitude_spectrum(mono, offset, settings.size);
        const auto norm = std::accumulate(current.begin(), current.end(), 0.0);
        if (norm > 0.0) {
            for (auto& magnitude : current) {
                magnitude /= norm;
            }
        }

        if (!previous.empty()) {
            auto flux = 0.0;
            for (auto i = std::size_t{0}; i < current.size(); ++i) {
                const auto increase = current[i] - previous[i];
                if (increase > 0.0) {
                    flux += increase * increase;
                }
            }
            const auto flux_value = std::sqrt(flux);
            sum_flux += flux_value;
            ++flux_count;
            if (current_analyze_settings().keep_intermediate_values) {
                values.push_back(flux_value);
            }
        }
        previous = std::move(current);
    }

    const auto value = flux_count == 0 ? 0.0 : sum_flux / static_cast<double>(flux_count);
    return complete_feature(feature_names::spectral_flux, value, std::move(values), "ratio", "Mean positive spectral change between normalized frames.");
}

FeatureResult extract_onset_density(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto mono = downmix_mono(audio);
    if (mono.empty() || audio.duration_seconds() == 0.0) {
        return complete_feature(feature_names::onset_density, 0.0, {}, "onsets/s", "Onset density requires samples and duration.");
    }

    const auto settings = frame_settings(audio, parameters, 1024, 512);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::onset_density, "Frame settings are invalid.");
    }
    const auto rms_threshold = non_negative_parameter(parameters, "rms_threshold", 0.02);
    const auto rise_threshold = non_negative_parameter(parameters, "rise_threshold", 1.5);
    auto onsets = std::size_t{0};
    auto previous_rms = 0.0;
    auto first = true;

    for (const auto offset : frame_offsets(mono.size(), settings.size, settings.hop)) {
        const auto current_rms = frame_rms(mono, offset, settings.size);
        if (!first && current_rms >= rms_threshold && current_rms >= previous_rms * rise_threshold) {
            ++onsets;
        }
        previous_rms = current_rms;
        first = false;
    }

    return complete_feature(feature_names::onset_density, static_cast<double>(onsets) / audio.duration_seconds(), {static_cast<double>(onsets)}, "onsets/s",
                            "Energy-rise onset estimate; values contains the raw onset count.");
}

FeatureResult extract_voice_activity_ratio(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto mono = downmix_mono(audio);
    if (mono.empty()) {
        return complete_feature(feature_names::voice_activity_ratio, 0.0, {}, "ratio", "No samples were supplied.");
    }

    const auto settings = frame_settings(audio, parameters, 1024, 512);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::voice_activity_ratio, "Frame settings are invalid.");
    }
    const auto rms_threshold = non_negative_parameter(parameters, "rms_threshold", 0.02);
    const auto zcr_min = non_negative_parameter(parameters, "zcr_min", 0.01);
    const auto zcr_max = non_negative_parameter(parameters, "zcr_max", 0.35);
    if (zcr_min > zcr_max) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("voice_activity_ratio zcr_min must be less than or equal to zcr_max.");
#endif
        return failed_feature(feature_names::voice_activity_ratio, "voice_activity_ratio zcr_min must be less than or equal to zcr_max.");
    }

    auto voiced = std::size_t{0};
    auto total = std::size_t{0};
    for (const auto offset : frame_offsets(mono.size(), settings.size, settings.hop)) {
        ++total;
        const auto current_rms = frame_rms(mono, offset, settings.size);
        auto crossings = std::size_t{0};
        for (auto i = std::size_t{1}; i < settings.size && offset + i < mono.size(); ++i) {
            const auto previous = mono[offset + i - 1];
            const auto current = mono[offset + i];
            if ((previous < 0.0 && current >= 0.0) || (previous >= 0.0 && current < 0.0)) {
                ++crossings;
            }
        }
        const auto zcr = settings.size <= 1 ? 0.0 : static_cast<double>(crossings) / static_cast<double>(settings.size - 1);
        if (current_rms >= rms_threshold && zcr >= zcr_min && zcr <= zcr_max) {
            ++voiced;
        }
    }

    return complete_feature(feature_names::voice_activity_ratio, total == 0 ? 0.0 : static_cast<double>(voiced) / static_cast<double>(total),
                            {static_cast<double>(voiced), static_cast<double>(total)}, "ratio",
                            "Simple RMS/ZCR voice activity estimate; values contains voiced frame count and total frame count.");
}

}
