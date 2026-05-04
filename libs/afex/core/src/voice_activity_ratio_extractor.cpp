#include "voice_activity_ratio_extractor.hpp"
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

FeatureResult extract_voice_activity_ratio(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto& mono = mono_audio(audio);
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
    for (const auto offset : cached_frame_offsets(audio, mono.size(), settings.size, settings.hop)) {
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

    auto values = std::vector<double>{};
    auto note = std::string{"Simple RMS/ZCR voice activity estimate."};
    if (current_analyze_settings().include_feature_values) {
        values = {static_cast<double>(voiced), static_cast<double>(total)};
        note += " Values field contains voiced frame count and total frame count.";
    }

    return complete_feature(feature_names::voice_activity_ratio, total == 0 ? 0.0 : static_cast<double>(voiced) / static_cast<double>(total),
                            std::move(values), "ratio", note);
}
}
