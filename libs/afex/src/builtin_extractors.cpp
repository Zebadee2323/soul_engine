#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
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
        throw std::invalid_argument("Extractor config name cannot be empty.");
    }
}

std::string_view BuiltinFeatureExtractor::name() const {
    return m_config.name;
}

FeatureResult BuiltinFeatureExtractor::extract(const AudioData& audio) const {
    if (m_config.name == feature_names::rms) {
        return extract_rms(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::rms_variance) {
        return extract_rms_variance(audio, m_config.parameters);
    }

    if (m_config.name == feature_names::zcr) {
        return extract_zcr(audio, m_config.parameters);
    }

    throw std::invalid_argument("Unknown afex extractor: " + m_config.name);
}

}

std::unique_ptr<FeatureExtractor> create_builtin_extractor(const ExtractorConfig& config) {
    if (config.name != feature_names::rms && config.name != feature_names::rms_variance && config.name != feature_names::zcr) {
        throw std::invalid_argument("Unknown afex extractor: " + config.name);
    }

    return std::make_unique<BuiltinFeatureExtractor>(config);
}

}
