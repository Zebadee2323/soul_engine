#include "spectral_flux_extractor.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include "extractor_helpers.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace afex
{

FeatureResult extract_spectral_flux(const AudioData& audio, const ExtractorParameters& parameters) {
    const auto& mono = mono_audio(audio);
    if (mono.empty()) {
        return complete_feature(feature_names::spectral_flux, 0.0, {}, "magnitude", "No samples were supplied.");
    }

    const auto settings = frame_settings(audio, parameters, 2048, 512);
    if (!frame_settings_valid(settings)) {
        return failed_feature(feature_names::spectral_flux, "Frame settings are invalid.");
    }
    auto values = std::vector<double>{};
    auto sum_flux = 0.0;
    auto flux_count = std::size_t{0};
    auto previous = std::vector<double>{};
    for (const auto offset : cached_frame_offsets(audio, mono.size(), settings.size, settings.hop)) {
        auto current = cached_magnitude_spectrum(audio, offset, settings.size);
        const auto norm = std::accumulate(current.begin(), current.end(), 0.0);
        if (norm > 0.0) {
            for (auto& magnitude : current) {
                magnitude /= norm;
            }
        }

        if (!previous.empty()) {
            auto flux = 0.0;
            for (auto i = std::size_t{0}; i < current.size(); ++i) {
                const auto increase = current[i] - previous[i];
                if (increase > 0.0) {
                    flux += increase * increase;
                }
            }
            const auto flux_value = std::sqrt(flux);
            sum_flux += flux_value;
            ++flux_count;
            if (current_analyze_settings().include_feature_values) {
                values.push_back(flux_value);
            }
        }
        previous = std::move(current);
    }

    const auto value = flux_count == 0 ? 0.0 : sum_flux / static_cast<double>(flux_count);
    return complete_feature(feature_names::spectral_flux, value, std::move(values), "ratio", "Mean positive spectral change between normalized frames.");
}

}
