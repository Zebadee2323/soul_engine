#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace afex
{

namespace
{

enum class BuiltinFeatureId
{
    Unknown,
    Rms,
    RmsVariance,
    Zcr,
    Pitch,
    PitchConfidence,
    SpectralCentroid,
    SpectralFlux,
    OnsetDensity,
    VoiceActivityRatio,
};

BuiltinFeatureId builtin_feature_id(std::string_view name) {
    if (name == feature_names::rms) {
        return BuiltinFeatureId::Rms;
    }

    if (name == feature_names::rms_variance) {
        return BuiltinFeatureId::RmsVariance;
    }

    if (name == feature_names::zcr) {
        return BuiltinFeatureId::Zcr;
    }

    if (name == feature_names::pitch) {
        return BuiltinFeatureId::Pitch;
    }

    if (name == feature_names::pitch_confidence) {
        return BuiltinFeatureId::PitchConfidence;
    }

    if (name == feature_names::spectral_centroid) {
        return BuiltinFeatureId::SpectralCentroid;
    }

    if (name == feature_names::spectral_flux) {
        return BuiltinFeatureId::SpectralFlux;
    }

    if (name == feature_names::onset_density) {
        return BuiltinFeatureId::OnsetDensity;
    }

    if (name == feature_names::voice_activity_ratio) {
        return BuiltinFeatureId::VoiceActivityRatio;
    }

    return BuiltinFeatureId::Unknown;
}

class BuiltinFeatureExtractor final : public FeatureExtractor
{

public:
    explicit                         BuiltinFeatureExtractor    (ExtractorConfig config);

    std::string_view                 name                       () const override;
    FeatureResult                    extract                    (const AudioData& audio) const override;

private:
    ExtractorConfig                  m_config;
    BuiltinFeatureId                 m_feature_id                 = BuiltinFeatureId::Unknown;
    PitchExtractorSettings           m_pitch_settings;
    SpectralCentroidExtractorSettings m_spectral_centroid_settings;
    OnsetDensityExtractorSettings    m_onset_density_settings;

};

BuiltinFeatureExtractor::BuiltinFeatureExtractor(ExtractorConfig config) :
    m_config                 (std::move(config)),
    m_feature_id             (builtin_feature_id(m_config.name))
{
    if (m_config.name.empty()) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Extractor config name cannot be empty.");
#endif
    }

    switch (m_feature_id) {
        case BuiltinFeatureId::Pitch:
        case BuiltinFeatureId::PitchConfidence:
            m_pitch_settings = make_pitch_extractor_settings(m_config.parameters);
            break;
        case BuiltinFeatureId::SpectralCentroid:
            m_spectral_centroid_settings = make_spectral_centroid_extractor_settings(m_config.parameters);
            break;
        case BuiltinFeatureId::OnsetDensity:
            m_onset_density_settings = make_onset_density_extractor_settings(m_config.parameters);
            break;
        case BuiltinFeatureId::Unknown:
        case BuiltinFeatureId::Rms:
        case BuiltinFeatureId::RmsVariance:
        case BuiltinFeatureId::Zcr:
        case BuiltinFeatureId::SpectralFlux:
        case BuiltinFeatureId::VoiceActivityRatio:
            break;
    }
}

std::string_view BuiltinFeatureExtractor::name() const {
    return m_config.name;
}

FeatureResult BuiltinFeatureExtractor::extract(const AudioData& audio) const {
    if (m_config.name.empty()) {
        return failed_feature(m_config.name, "Extractor config name cannot be empty.");
    }

    switch (m_feature_id) {
        case BuiltinFeatureId::Rms:
            return extract_rms(audio, m_config.parameters);
        case BuiltinFeatureId::RmsVariance:
            return extract_rms_variance(audio, m_config.parameters);
        case BuiltinFeatureId::Zcr:
            return extract_zcr(audio, m_config.parameters);
        case BuiltinFeatureId::Pitch:
            return extract_pitch(audio, m_pitch_settings);
        case BuiltinFeatureId::PitchConfidence:
            return extract_pitch_confidence(audio, m_pitch_settings);
        case BuiltinFeatureId::SpectralCentroid:
            return extract_spectral_centroid(audio, m_spectral_centroid_settings);
        case BuiltinFeatureId::SpectralFlux:
            return extract_spectral_flux(audio, m_config.parameters);
        case BuiltinFeatureId::OnsetDensity:
            return extract_onset_density(audio, m_onset_density_settings);
        case BuiltinFeatureId::VoiceActivityRatio:
            return extract_voice_activity_ratio(audio, m_config.parameters);
        case BuiltinFeatureId::Unknown:
            break;
    }

#if AFEX_EMBEDDED
    return failed_feature(m_config.name, {});
#else
    return failed_feature(m_config.name, "Unknown afex extractor: " + m_config.name);
#endif
}

}

std::unique_ptr<FeatureExtractor> create_builtin_extractor(const ExtractorConfig& config) {
    const auto names = builtin_feature_names();
    const auto found = std::find(names.begin(), names.end(), config.name);
    if (found == names.end()) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Unknown afex extractor: " + config.name);
#endif
        return nullptr;
    }

    return std::make_unique<BuiltinFeatureExtractor>(config);
}

}
