#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cmath>
#include <numeric>
#include <string>

namespace afex
{

FeatureResult extract_rms(const AudioData& audio) {
    if (audio.samples.empty()) {
        return FeatureResult{
            .name = std::string{feature_names::rms},
            .status = FeatureStatus::Complete,
            .value = 0.0,
            .values = {},
            .unit = "amplitude",
            .note = "No samples were supplied.",
        };
    }

    const auto square_sum = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [](double sum, float sample) {
            return sum + static_cast<double>(sample) * static_cast<double>(sample);
        }
    );

    return FeatureResult{
        .name = std::string{feature_names::rms},
        .status = FeatureStatus::Complete,
        .value = std::sqrt(square_sum / static_cast<double>(audio.samples.size())),
        .values = {},
        .unit = "amplitude",
        .note = {},
    };
}

FeatureResult extract_rms_variance(const AudioData& audio) {
    if (audio.samples.empty()) {
        return FeatureResult{
            .name = std::string{feature_names::rms_variance},
            .status = FeatureStatus::Complete,
            .value = 0.0,
            .values = {},
            .unit = "amplitude^2",
            .note = "No samples were supplied.",
        };
    }

    const auto mean_square = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [](double sum, float sample) {
            return sum + static_cast<double>(sample) * static_cast<double>(sample);
        }
    ) / static_cast<double>(audio.samples.size());

    const auto variance = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [mean_square](double sum, float sample) {
            const auto square = static_cast<double>(sample) * static_cast<double>(sample);
            const auto delta = square - mean_square;
            return sum + delta * delta;
        }
    ) / static_cast<double>(audio.samples.size());

    return FeatureResult{
        .name = std::string{feature_names::rms_variance},
        .status = FeatureStatus::Complete,
        .value = variance,
        .values = {},
        .unit = "amplitude^4",
        .note = "Scaffold metric over raw sample energy; frame-based RMS variance can replace this extractor later.",
    };
}

}
