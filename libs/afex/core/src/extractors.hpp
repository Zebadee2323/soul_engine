#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>

namespace afex
{

const AnalyzeSettings& current_analyze_settings     ();
FeatureResult          complete_feature             (std::string_view name, double value, std::vector<double> values = {},
                                                        std::string_view unit = {}, std::string_view note = {});
FeatureResult          failed_feature               (std::string_view name, std::string_view note = {});

FeatureResult   extract_rms                     (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_rms_variance            (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_zcr                     (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_pitch                   (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_pitch_confidence        (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_spectral_centroid       (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_spectral_flux           (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_onset_density           (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_voice_activity_ratio    (const AudioData& audio, const ExtractorParameters& parameters = {});

std::unique_ptr<FeatureExtractor> create_builtin_extractor(const ExtractorConfig& config);

}
