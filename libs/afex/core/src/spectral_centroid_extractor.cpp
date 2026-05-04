#include "spectral_centroid_extractor.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include "extractor_helpers.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace afex
{

SpectralCentroidExtractorSettings make_spectral_centroid_extractor_settings(const ExtractorParameters& parameters) {
    return SpectralCentroidExtractorSettings{
        .frame = frame_parameter_settings(parameters, 2048, 512),
        .normalization = normalization_settings(parameters, 0.0, 0.0),
    };
}

FeatureResult extract_spectral_centroid(const AudioData& audio, const ExtractorParameters& parameters) {
    return extract_spectral_centroid(audio, make_spectral_centroid_extractor_settings(parameters));
}

FeatureResult extract_spectral_centroid(const AudioData& audio, const SpectralCentroidExtractorSettings& extractor_settings) {
    const auto& mono = mono_audio(audio);
    if (mono.empty() || audio.sample_rate_hz == 0) {
        return complete_feature(feature_names::spectral_centroid, 0.0, {}, "Hz", "Spectral centroid requires samples and a non-zero sample rate.");
    }

    const auto settings = frame_settings(audio, extractor_settings.frame);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::spectral_centroid, "Frame settings are invalid.");
    }
    auto values = std::vector<double>{};
    auto sum_centroid = 0.0;
    auto centroid_count = std::size_t{0};
    for (const auto offset : cached_frame_offsets(audio, mono.size(), settings.size, settings.hop)) {
        const auto& magnitudes = cached_magnitude_spectrum(audio, offset, settings.size);
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
        if (current_analyze_settings().include_feature_values) {
            values.push_back(centroid);
        }
    }

    auto value = centroid_count == 0 ? 0.0 : sum_centroid / static_cast<double>(centroid_count);
    auto unit = std::string_view{"Hz"};
    auto note = std::string{"Mean spectral centroid across Hann-windowed frames."};
    if (extractor_settings.normalization.enabled) {
        const auto maximum = extractor_settings.normalization.maximum == 0.0 ? static_cast<double>(audio.sample_rate_hz) * 0.5 :
                             extractor_settings.normalization.maximum;
        if (!normalization_range_valid(extractor_settings.normalization.minimum, maximum)) {
#if AFEX_ENABLE_EXCEPTIONS
            throw std::invalid_argument("spectral_centroid normalization_min_hz must be less than normalization_max_hz.");
#endif
            return failed_feature(feature_names::spectral_centroid, "spectral_centroid normalization_min_hz must be less than normalization_max_hz.");
        }

        value = normalize_to_unit_interval(value, extractor_settings.normalization.minimum, maximum);
        normalize_values_to_unit_interval(values, extractor_settings.normalization.minimum, maximum);
        unit = "ratio";
        note += " Values were normalized with normalization_min_hz and normalization_max_hz.";
    }

    return complete_feature(feature_names::spectral_centroid, value, std::move(values), unit, note);
}

}
