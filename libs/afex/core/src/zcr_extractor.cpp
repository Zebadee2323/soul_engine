#include "zcr_extractor.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cmath>
#include <cstddef>
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

int sign_with_threshold(float sample, double threshold) {
    if (static_cast<double>(sample) > threshold) {
        return 1;
    }

    if (static_cast<double>(sample) < -threshold) {
        return -1;
    }

    return 0;
}

}

FeatureResult extract_zcr(const AudioData& audio, const ExtractorParameters& parameters) {
    if (audio.samples.size() < 2) {
        return complete_feature(feature_names::zcr, 0.0, {}, "ratio", "At least two samples are needed for zero crossing rate.");
    }

    const auto threshold = parameter_or(parameters, "threshold", 0.0);
    if (threshold < 0.0) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("zcr.threshold must be greater than or equal to 0.");
#endif
        return failed_feature(feature_names::zcr, "zcr.threshold must be greater than or equal to 0.");
    }

    auto crossings = std::size_t{0};
    auto comparable_steps = std::size_t{0};
    auto previous_sign = sign_with_threshold(audio.samples[0], threshold);

    for (auto i = std::size_t{1}; i < audio.samples.size(); ++i) {
        const auto current_sign = sign_with_threshold(audio.samples[i], threshold);
        if (previous_sign != 0 && current_sign != 0) {
            ++comparable_steps;
            if (previous_sign != current_sign) {
                ++crossings;
            }
        }

        if (current_sign != 0) {
            previous_sign = current_sign;
        }
    }

    auto note = std::string{};
#if !AFEX_EMBEDDED
    if (threshold > 0.0) {
        note = "Samples within +/-threshold were ignored for sign changes.";
    }
#endif

    return complete_feature(feature_names::zcr, comparable_steps == 0 ? 0.0 : static_cast<double>(crossings) / static_cast<double>(comparable_steps),
                            {}, "ratio", note);
}

}
