#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>

namespace afex
{

const AnalyzeSettings& current_analyze_settings     ();
FeatureResult          complete_feature             (std::string_view name, double value, std::vector<double> values = {},
                                                        std::string_view unit = {}, std::string_view note = {});
FeatureResult          failed_feature               (std::string_view name, std::string_view note = {});

struct NormalizationSettings
{
    bool                                enabled                     = false;
    double                              minimum                     = 0.0;
    double                              maximum                     = 1.0;
};

struct FrameParameterSettings
{
    std::size_t                         size                        = 2048;
    std::size_t                         hop                         = 512;
    double                              size_ms                     = 0.0;
    double                              hop_ms                      = 0.0;
};

struct PitchExtractorSettings
{
    double                              min_frequency_hz            = 50.0;
    double                              max_frequency_hz            = 500.0;
    double                              confidence_threshold        = 0.3;
    NormalizationSettings               normalization;
};

struct SpectralCentroidExtractorSettings
{
    FrameParameterSettings              frame;
    NormalizationSettings               normalization;
};

struct OnsetDensityExtractorSettings
{
    FrameParameterSettings              frame                       = {.size = 1024, .hop = 512};
    double                              rms_threshold               = 0.02;
    double                              rise_threshold              = 1.5;
    NormalizationSettings               normalization;
};

PitchExtractorSettings                  make_pitch_extractor_settings              (const ExtractorParameters& parameters);
SpectralCentroidExtractorSettings       make_spectral_centroid_extractor_settings  (const ExtractorParameters& parameters);
OnsetDensityExtractorSettings           make_onset_density_extractor_settings      (const ExtractorParameters& parameters);

FeatureResult   extract_rms                     (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_rms_variance            (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_zcr                     (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_pitch                   (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_pitch                   (const AudioData& audio, const PitchExtractorSettings& settings);
FeatureResult   extract_pitch_confidence        (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_pitch_confidence        (const AudioData& audio, const PitchExtractorSettings& settings);
FeatureResult   extract_spectral_centroid       (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_spectral_centroid       (const AudioData& audio, const SpectralCentroidExtractorSettings& settings);
FeatureResult   extract_spectral_flux           (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_onset_density           (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult   extract_onset_density           (const AudioData& audio, const OnsetDensityExtractorSettings& settings);
FeatureResult   extract_voice_activity_ratio    (const AudioData& audio, const ExtractorParameters& parameters = {});

std::unique_ptr<FeatureExtractor> create_builtin_extractor(const ExtractorConfig& config);

}
