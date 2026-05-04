#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

double sample_or_silence(float sample, double noise_floor) {
    const auto value = static_cast<double>(sample);
    if (std::abs(value) < noise_floor) {
        return 0.0;
    }

    return value;
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
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("rms.noise_floor must be greater than or equal to 0.");
#endif
        return failed_feature(feature_names::rms, "rms.noise_floor must be greater than or equal to 0.");
    }

    const auto square_sum = std::accumulate(
        audio.samples.begin(), audio.samples.end(), 0.0,
        [noise_floor](double sum, float sample) {
            const auto value = sample_or_silence(sample, noise_floor);
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
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("rms_variance.noise_floor must be greater than or equal to 0.");
#endif
        return failed_feature(feature_names::rms_variance, "rms_variance.noise_floor must be greater than or equal to 0.");
    }

    const auto frame_size = positive_size_parameter(parameters, "frame_size", 1024);
    const auto hop_size = positive_size_parameter(parameters, "hop_size", 512);
    const auto keep_intermediate_values = current_analyze_settings().keep_intermediate_values;
    auto frame_rms_values = std::vector<double>{};
    if (keep_intermediate_values) {
        frame_rms_values.reserve((audio.samples.size() + hop_size - 1) / hop_size);
    }

    auto frame_rms_mean = 0.0;
    auto frame_rms_squared_delta_sum = 0.0;
    auto frame_rms_count = std::size_t{0};
    for (auto offset = std::size_t{0}; offset < audio.samples.size(); offset += hop_size) {
        const auto end = std::min(offset + frame_size, audio.samples.size());
        auto square_sum = 0.0;
        for (auto i = offset; i < end; ++i) {
            const auto value = sample_or_silence(audio.samples[i], noise_floor);
            square_sum += value * value;
        }

        const auto frame_rms = std::sqrt(square_sum / static_cast<double>(end - offset));
        ++frame_rms_count;
        const auto delta = frame_rms - frame_rms_mean;
        frame_rms_mean += delta / static_cast<double>(frame_rms_count);
        frame_rms_squared_delta_sum += delta * (frame_rms - frame_rms_mean);
        if (keep_intermediate_values) {
            frame_rms_values.push_back(frame_rms);
        }
        if (end == audio.samples.size()) {
            break;
        }
    }

    const auto variance = frame_rms_squared_delta_sum / static_cast<double>(frame_rms_count);

    auto note = std::string{"Variance of frame RMS values."};
    if (noise_floor > 0.0) {
        note += " Samples below noise_floor were treated as silence.";
    }

    return FeatureResult{
        .name = std::string{feature_names::rms_variance},
        .status = FeatureStatus::Complete,
        .value = variance,
        .values = std::move(frame_rms_values),
        .unit = "amplitude^2",
        .note = note,
    };
}

}
