#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace afex
{

namespace
{

double parameter_or(const ExtractorParameters& parameters, std::string_view name, double fallback) {
    const auto found = parameters.find(std::string{name});
    if (found == parameters.end()) {
        return fallback;
    }

    return found->second;
}

}

FeatureResult extract_rms(const AudioData& audio, const ExtractorParameters& parameters) {
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

    const auto noise_floor = parameter_or(parameters, "noise_floor", 0.0);
    if (noise_floor < 0.0) {
        throw std::invalid_argument("rms.noise_floor must be greater than or equal to 0.");
    }

    const auto square_sum = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [noise_floor](double sum, float sample) {
            const auto value = std::abs(static_cast<double>(sample)) < noise_floor ? 0.0 : static_cast<double>(sample);
            return sum + value * value;
        }
    );

    auto note = std::string{};
    if (noise_floor > 0.0) {
        note = "Samples below noise_floor were treated as silence.";
    }

    return FeatureResult{
        .name = std::string{feature_names::rms},
        .status = FeatureStatus::Complete,
        .value = std::sqrt(square_sum / static_cast<double>(audio.samples.size())),
        .values = {},
        .unit = "amplitude",
        .note = note,
    };
}

FeatureResult extract_rms_variance(const AudioData& audio, const ExtractorParameters& parameters) {
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

    const auto noise_floor = parameter_or(parameters, "noise_floor", 0.0);
    if (noise_floor < 0.0) {
        throw std::invalid_argument("rms_variance.noise_floor must be greater than or equal to 0.");
    }

    const auto mean_square = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [noise_floor](double sum, float sample) {
            const auto value = std::abs(static_cast<double>(sample)) < noise_floor ? 0.0 : static_cast<double>(sample);
            return sum + value * value;
        }
    ) / static_cast<double>(audio.samples.size());

    const auto variance = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [mean_square, noise_floor](double sum, float sample) {
            const auto value = std::abs(static_cast<double>(sample)) < noise_floor ? 0.0 : static_cast<double>(sample);
            const auto square = value * value;
            const auto delta = square - mean_square;
            return sum + delta * delta;
        }
    ) / static_cast<double>(audio.samples.size());

    auto note = std::string{"Scaffold metric over raw sample energy; frame-based RMS variance can replace this extractor later."};
    if (noise_floor > 0.0) {
        note += " Samples below noise_floor were treated as silence.";
    }

    return FeatureResult{
        .name = std::string{feature_names::rms_variance},
        .status = FeatureStatus::Complete,
        .value = variance,
        .values = {},
        .unit = "amplitude^4",
        .note = note,
    };
}

}
