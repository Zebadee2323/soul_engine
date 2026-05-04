#include <afex/afex.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace afex
{

bool AudioData::empty() const {
    return samples.empty();
}

std::size_t AudioData::frame_count() const {
    if (channel_count == 0) {
        return 0;
    }

    return samples.size() / channel_count;
}

double AudioData::duration_seconds() const {
    if (sample_rate_hz == 0) {
        return 0.0;
    }

    return static_cast<double>(frame_count()) / static_cast<double>(sample_rate_hz);
}

const FeatureResult* AnalysisResult::find_feature(std::string_view name) const {
    const auto found = std::find_if(features.begin(), features.end(), [name](const FeatureResult& feature) {
        return feature.name == name;
    });

    if (found == features.end()) {
        return nullptr;
    }

    return &(*found);
}

CallbackFeatureExtractor::CallbackFeatureExtractor(std::string name, FeatureExtractorFn extract_fn) :
    m_name                  (std::move(name)),
    m_extract_fn            (std::move(extract_fn))
{
    if (m_name.empty()) {
        throw std::invalid_argument("Feature extractor name cannot be empty.");
    }

    if (!m_extract_fn) {
        throw std::invalid_argument("Feature extractor callback cannot be empty.");
    }
}

std::string_view CallbackFeatureExtractor::name() const {
    return m_name;
}

FeatureResult CallbackFeatureExtractor::extract(const AudioData& audio) const {
    auto result = m_extract_fn(audio);
    if (result.name.empty()) {
        result.name = m_name;
    }

    return result;
}

void ExtractorRegistry::register_extractor(std::unique_ptr<FeatureExtractor> extractor) {
    if (!extractor) {
        throw std::invalid_argument("Feature extractor cannot be null.");
    }

    const auto extractor_name = std::string{extractor->name()};
    unregister_extractor(extractor_name);
    m_extractors.push_back(std::move(extractor));
}

void ExtractorRegistry::register_extractor(std::string name, FeatureExtractorFn extract_fn) {
    register_extractor(std::make_unique<CallbackFeatureExtractor>(std::move(name), std::move(extract_fn)));
}

bool ExtractorRegistry::unregister_extractor(std::string_view name) {
    const auto old_size = m_extractors.size();
    m_extractors.erase(
        std::remove_if(m_extractors.begin(), m_extractors.end(), [name](const auto& extractor) {
            return extractor->name() == name;
        }),
        m_extractors.end()
    );

    return m_extractors.size() != old_size;
}

bool ExtractorRegistry::contains(std::string_view name) const {
    return std::any_of(m_extractors.begin(), m_extractors.end(), [name](const auto& extractor) {
        return extractor->name() == name;
    });
}

std::vector<std::string> ExtractorRegistry::names() const {
    auto result = std::vector<std::string>{};
    result.reserve(m_extractors.size());

    for (const auto& extractor : m_extractors) {
        result.push_back(std::string{extractor->name()});
    }

    return result;
}

std::vector<FeatureResult> ExtractorRegistry::extract_all(const AudioData& audio) const {
    auto result = std::vector<FeatureResult>{};
    result.reserve(m_extractors.size());

    for (const auto& extractor : m_extractors) {
        result.push_back(extractor->extract(audio));
    }

    return result;
}

Analyzer::Analyzer() = default;

void Analyzer::register_extractor(std::unique_ptr<FeatureExtractor> extractor) {
    m_registry.register_extractor(std::move(extractor));
}

void Analyzer::register_extractor(std::string name, FeatureExtractorFn extract_fn) {
    m_registry.register_extractor(std::move(name), std::move(extract_fn));
}

void Analyzer::register_default_extractors() {
    register_extractor(std::string{feature_names::rms}, extract_rms);
    register_extractor(std::string{feature_names::rms_variance}, extract_rms_variance);
    register_extractor(std::string{feature_names::zcr}, extract_zcr);
}

std::vector<std::string> Analyzer::registered_extractors() const {
    return m_registry.names();
}

AnalysisResult Analyzer::analyze(const AudioData& audio) const {
    return AnalysisResult{
        .sample_rate_hz = audio.sample_rate_hz,
        .channel_count = audio.channel_count,
        .sample_count = audio.samples.size(),
        .frame_count = audio.frame_count(),
        .duration_seconds = audio.duration_seconds(),
        .features = m_registry.extract_all(audio),
    };
}

Analyzer create_default_analyzer() {
    auto analyzer = Analyzer{};
    analyzer.register_default_extractors();
    return analyzer;
}

std::string placeholder_method() {
    return "afex placeholder method executed";
}

}
