#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <string>

namespace afex
{

FeatureResult extract_zcr(const AudioData& audio) {
    if (audio.samples.size() < 2) {
        return FeatureResult{
            .name = std::string{feature_names::zcr},
            .status = FeatureStatus::Complete,
            .value = 0.0,
            .values = {},
            .unit = "ratio",
            .note = "At least two samples are needed for zero crossing rate.",
        };
    }

    auto crossings = std::size_t{0};
    for (auto i = std::size_t{1}; i < audio.samples.size(); ++i) {
        const auto previous = audio.samples[i - 1];
        const auto current = audio.samples[i];
        if ((previous < 0.0f && current >= 0.0f) || (previous >= 0.0f && current < 0.0f)) {
            ++crossings;
        }
    }

    return FeatureResult{
        .name = std::string{feature_names::zcr},
        .status = FeatureStatus::Complete,
        .value = static_cast<double>(crossings) / static_cast<double>(audio.samples.size() - 1),
        .values = {},
        .unit = "ratio",
        .note = {},
    };
}

}
