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

class BuiltinFeatureExtractor final : public FeatureExtractor
{

public:
    explicit                         BuiltinFeatureExtractor    (ExtractorConfig config);

    std::string_view                 name                       () const override;
    FeatureResult                    extract                    (const AudioData& audio) const override;

private:
    ExtractorConfig                  m_config;

};

BuiltinFeatureExtractor::BuiltinFeatureExtractor(ExtractorConfig config) :
    m_config                 (std::move(config))
{
    if (m_config.name.empty()) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Extractor config name cannot be empty.");
#endif
    }
}

std::string_view BuiltinFeatureExtractor::name() const {
    return m_config.name;
}

FeatureResult BuiltinFeatureExtractor::extract(const AudioData& audio) const {
    if (m_config.name.empty()) {
        return failed_feature(m_config.name, "Extractor config name cannot be empty.");
    }

    if (m_config.name == feature_names::rms) {
        return extract_rms(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::rms_variance) {
        return extract_rms_variance(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::zcr) {
        return extract_zcr(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::pitch) {
        return extract_pitch(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::pitch_confidence) {
        return extract_pitch_confidence(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::spectral_centroid) {
        return extract_spectral_centroid(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::spectral_flux) {
        return extract_spectral_flux(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::onset_density) {
        return extract_onset_density(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::voice_activity_ratio) {
        return extract_voice_activity_ratio(audio, m_config.parameters);
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
