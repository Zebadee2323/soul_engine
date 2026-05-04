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

namespace afex
{

namespace
{

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

PitchSummary estimate_pitch(const AudioData& audio, const PitchExtractorSettings& settings) {
    if (audio.sample_rate_hz == 0) {
        return pitch_summary(0.0, 0.0, "Pitch requires a non-zero sample rate.");
    }

    const auto mono = downmix_mono(audio);
    if (mono.size() < 3) {
        return pitch_summary(0.0, 0.0, "At least three frames are needed for pitch estimation.");
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

PitchExtractorSettings make_pitch_extractor_settings(const ExtractorParameters& parameters) {
    const auto min_frequency_hz = parameter_or(parameters, "min_frequency_hz", 50.0);
    const auto max_frequency_hz = parameter_or(parameters, "max_frequency_hz", 500.0);
    auto settings = PitchExtractorSettings{
        .min_frequency_hz = min_frequency_hz,
        .max_frequency_hz = max_frequency_hz,
        .confidence_threshold = parameter_or(parameters, "confidence_threshold", 0.3),
        .normalization = normalization_settings(parameters, "normalization_min_hz", "normalization_max_hz", min_frequency_hz, max_frequency_hz),
    };
    return settings;
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
