#include "onset_density_extractor.hpp"
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

OnsetDensityExtractorSettings make_onset_density_extractor_settings(const ExtractorParameters& parameters) {
    return OnsetDensityExtractorSettings{
        .frame = frame_parameter_settings(parameters, 1024, 512),
        .rms_threshold = non_negative_parameter(parameters, "rms_threshold", 0.02),
        .rise_threshold = non_negative_parameter(parameters, "rise_threshold", 1.5),
        .normalization = normalization_settings(parameters, "normalization_min_onsets_per_second", "normalization_max_onsets_per_second", 0.0, 0.0),
    };
}

FeatureResult extract_onset_density(const AudioData& audio, const ExtractorParameters& parameters) {
    return extract_onset_density(audio, make_onset_density_extractor_settings(parameters));
}

FeatureResult extract_onset_density(const AudioData& audio, const OnsetDensityExtractorSettings& extractor_settings) {
    const auto mono = downmix_mono(audio);
    if (mono.empty() || audio.duration_seconds() == 0.0) {
        return complete_feature(feature_names::onset_density, 0.0, {}, "onsets/s", "Onset density requires samples and duration.");
    }

    const auto settings = frame_settings(audio, extractor_settings.frame);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::onset_density, "Frame settings are invalid.");
    }
    const auto rms_threshold = extractor_settings.rms_threshold;
    const auto rise_threshold = extractor_settings.rise_threshold;
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

    auto values = std::vector<double>{};
    auto note = std::string{"Energy-rise onset estimate."};
    if (current_analyze_settings().include_feature_values) {
        values.push_back(static_cast<double>(onsets));
        note += " Values field contains the raw onset count.";
    }

    auto value = static_cast<double>(onsets) / audio.duration_seconds();
    auto unit = std::string_view{"onsets/s"};
    if (extractor_settings.normalization.enabled) {
        const auto maximum = extractor_settings.normalization.maximum == 0.0 ?
                             static_cast<double>(frame_offsets(mono.size(), settings.size, settings.hop).size()) / audio.duration_seconds() :
                             extractor_settings.normalization.maximum;
        if (!normalization_range_valid(extractor_settings.normalization.minimum, maximum)) {
#if AFEX_ENABLE_EXCEPTIONS
            throw std::invalid_argument("onset_density normalization_min_onsets_per_second must be less than normalization_max_onsets_per_second.");
#endif
            return failed_feature(feature_names::onset_density,
                                  "onset_density normalization_min_onsets_per_second must be less than normalization_max_onsets_per_second.");
        }

        value = normalize_to_unit_interval(value, extractor_settings.normalization.minimum, maximum);
        unit = "ratio";
        note += " Density was normalized with normalization_min_onsets_per_second and normalization_max_onsets_per_second.";
    }

    return complete_feature(feature_names::onset_density, value, std::move(values), unit, note);
}

}
