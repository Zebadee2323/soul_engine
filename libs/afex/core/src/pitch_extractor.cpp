#include "pitch_extractor.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include "extractor_helpers.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace afex
{

namespace
{

PitchSummary pitch_summary(double frequency_hz, double confidence, std::string_view note) {
    return PitchSummary{
        .frequency_hz = frequency_hz,
        .confidence = confidence,
#if !AFEX_EMBEDDED
        .note = std::string{note},
#endif
    };
}

std::vector<double> pitch_analysis_samples(const std::vector<double>& mono, std::uint32_t source_sample_rate_hz, const PitchExtractorSettings& settings,
                                           double& analysis_sample_rate_hz, bool& duration_capped) {
    auto source_count = mono.size();
    if (settings.max_analysis_seconds > 0.0 && source_sample_rate_hz != 0) {
        const auto capped_source_count = static_cast<std::size_t>(std::ceil(settings.max_analysis_seconds * static_cast<double>(source_sample_rate_hz)));
        source_count = std::min(source_count, capped_source_count);
        duration_capped = source_count < mono.size();
    }

    analysis_sample_rate_hz = static_cast<double>(source_sample_rate_hz);
    if (settings.max_sample_rate_hz <= 0.0 || analysis_sample_rate_hz <= settings.max_sample_rate_hz) {
        return std::vector<double>{mono.begin(), mono.begin() + static_cast<std::ptrdiff_t>(source_count)};
    }

    const auto sample_rate_ratio = analysis_sample_rate_hz / settings.max_sample_rate_hz;
    const auto target_count = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(static_cast<double>(source_count) / sample_rate_ratio)));
    auto samples = std::vector<double>{};
    samples.reserve(target_count);

    for (auto i = std::size_t{0}; i < target_count; ++i) {
        const auto source_index = std::min(source_count - 1, static_cast<std::size_t>(std::floor(static_cast<double>(i) * sample_rate_ratio)));
        samples.push_back(mono[source_index]);
    }

    analysis_sample_rate_hz = static_cast<double>(source_sample_rate_hz) / sample_rate_ratio;
    return samples;
}

}

PitchExtractorSettings make_pitch_extractor_settings(const ExtractorParameters& parameters) {
    const auto min_frequency_hz = parameter_or(parameters, "min_frequency_hz", 50.0);
    const auto max_frequency_hz = parameter_or(parameters, "max_frequency_hz", 500.0);
    auto settings = PitchExtractorSettings{
        .min_frequency_hz = min_frequency_hz,
        .max_frequency_hz = max_frequency_hz,
        .confidence_threshold = parameter_or(parameters, "confidence_threshold", 0.3),
        .max_analysis_seconds = parameter_or(parameters, "max_analysis_seconds", 3.0),
        .max_sample_rate_hz = parameter_or(parameters, "max_sample_rate_hz", 8000.0),
        .normalization = normalization_settings(parameters, "normalization_min_hz", "normalization_max_hz", min_frequency_hz, max_frequency_hz),
    };
    return settings;
}

PitchSummary estimate_pitch(const AudioData& audio, const PitchExtractorSettings& settings) {
    if (const auto* cached = cached_pitch_summary(audio, settings)) {
        return *cached;
    }

    if (audio.sample_rate_hz == 0) {
        auto summary = pitch_summary(0.0, 0.0, "Pitch requires a non-zero sample rate.");
        cache_pitch_summary(audio, settings, summary);
        return summary;
    }

    const auto& mono = mono_audio(audio);
    if (mono.size() < 3) {
        auto summary = pitch_summary(0.0, 0.0, "At least three frames are needed for pitch estimation.");
        cache_pitch_summary(audio, settings, summary);
        return summary;
    }

    const auto min_frequency_hz = settings.min_frequency_hz;
    const auto max_frequency_hz = settings.max_frequency_hz;
    const auto confidence_threshold = settings.confidence_threshold;
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
    if (settings.max_analysis_seconds < 0.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("pitch max_analysis_seconds must be greater than or equal to 0.");
#endif
        return pitch_summary(0.0, 0.0, "pitch max_analysis_seconds must be greater than or equal to 0.");
    }
    if (settings.max_sample_rate_hz < 0.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("pitch max_sample_rate_hz must be greater than or equal to 0.");
#endif
        return pitch_summary(0.0, 0.0, "pitch max_sample_rate_hz must be greater than or equal to 0.");
    }

    auto analysis_sample_rate_hz = static_cast<double>(audio.sample_rate_hz);
    auto duration_capped = false;
    const auto samples = pitch_analysis_samples(mono, audio.sample_rate_hz, settings, analysis_sample_rate_hz, duration_capped);
    if (samples.size() < 3) {
        auto summary = pitch_summary(0.0, 0.0, "At least three analysis samples are needed for pitch estimation.");
        cache_pitch_summary(audio, settings, summary);
        return summary;
    }

    auto min_lag = static_cast<std::size_t>(std::floor(analysis_sample_rate_hz / max_frequency_hz));
    auto max_lag = static_cast<std::size_t>(std::ceil(analysis_sample_rate_hz / min_frequency_hz));
    min_lag = std::max<std::size_t>(1, min_lag);
    max_lag = std::min<std::size_t>(max_lag, samples.size() - 1);
    if (min_lag > max_lag) {
        auto summary = pitch_summary(0.0, 0.0, "Audio is too short for the configured pitch frequency range.");
        cache_pitch_summary(audio, settings, summary);
        return summary;
    }

    auto best_lag = min_lag;
    auto best_correlation = -std::numeric_limits<double>::infinity();
    for (auto lag = min_lag; lag <= max_lag; ++lag) {
        auto cross = 0.0;
        auto left_energy = 0.0;
        auto right_energy = 0.0;
        for (auto i = std::size_t{0}; i + lag < samples.size(); ++i) {
            cross += samples[i] * samples[i + lag];
            left_energy += samples[i] * samples[i];
            right_energy += samples[i + lag] * samples[i + lag];
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
        auto summary = pitch_summary(0.0, confidence, "No pitch exceeded confidence_threshold.");
        cache_pitch_summary(audio, settings, summary);
        return summary;
    }

    auto note = std::string{"Autocorrelation estimate over capped, downmixed mono audio."};
#if !AFEX_EMBEDDED
    if (duration_capped) {
        note += " Analysis duration was capped with max_analysis_seconds.";
    }
    if (analysis_sample_rate_hz < static_cast<double>(audio.sample_rate_hz)) {
        note += " Analysis sample rate was capped with max_sample_rate_hz.";
    }
#endif
    auto summary = pitch_summary(analysis_sample_rate_hz / static_cast<double>(best_lag), confidence, note);
    cache_pitch_summary(audio, settings, summary);
    return summary;
}

FeatureResult extract_pitch(const AudioData& audio, const ExtractorParameters& parameters) {
    return extract_pitch(audio, make_pitch_extractor_settings(parameters));
}

FeatureResult extract_pitch(const AudioData& audio, const PitchExtractorSettings& settings) {
    const auto pitch = estimate_pitch(audio, settings);
    auto value = pitch.frequency_hz;
    auto unit = std::string_view{"Hz"};
#if !AFEX_EMBEDDED
    auto note = pitch.note;
#endif

    if (settings.normalization.enabled) {
        if (!normalization_range_valid(settings.normalization.minimum, settings.normalization.maximum)) {
#if AFEX_ENABLE_EXCEPTIONS
            throw std::invalid_argument("pitch normalization_min_hz must be less than normalization_max_hz.");
#endif
            return failed_feature(feature_names::pitch, "pitch normalization_min_hz must be less than normalization_max_hz.");
        }

        value = normalize_to_unit_interval(value, settings.normalization.minimum, settings.normalization.maximum);
        unit = "ratio";
#if !AFEX_EMBEDDED
        note += " Pitch was normalized with normalization_min_hz and normalization_max_hz.";
#endif
    }

#if AFEX_EMBEDDED
    return complete_feature(feature_names::pitch, value, {}, unit);
#else
    return complete_feature(feature_names::pitch, value, {}, unit, note);
#endif
}

FeatureResult extract_pitch_confidence(const AudioData& audio, const ExtractorParameters& parameters) {
    return extract_pitch_confidence(audio, make_pitch_extractor_settings(parameters));
}

FeatureResult extract_pitch_confidence(const AudioData& audio, const PitchExtractorSettings& settings) {
    const auto pitch = estimate_pitch(audio, settings);
#if AFEX_EMBEDDED
    return complete_feature(feature_names::pitch_confidence, pitch.confidence, {}, "ratio");
#else
    return complete_feature(feature_names::pitch_confidence, pitch.confidence, {}, "ratio", pitch.note);
#endif
}

}
