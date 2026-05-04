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

AudioData LoadedAudioData::view() const {
    return AudioData{
        .samples = samples,
        .sample_rate_hz = sample_rate_hz,
        .channel_count = channel_count,
    };
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

void ExtractorRegistry::clear() {
    m_extractors.clear();
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

void Analyzer::register_builtin_extractor(const ExtractorConfig& config) {
    register_extractor(create_builtin_extractor(config));
}

void Analyzer::register_default_extractors() {
    configure_extractors({
        ExtractorConfig{.name = std::string{feature_names::rms}},
        ExtractorConfig{.name = std::string{feature_names::rms_variance}},
        ExtractorConfig{.name = std::string{feature_names::zcr}},
    });
}

void Analyzer::configure_extractors(const std::vector<ExtractorConfig>& configs) {
    m_registry.clear();

    for (const auto& config : configs) {
        register_builtin_extractor(config);
    }
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

AnalysisResult Analyzer::analyze_file(std::string_view audio_file_path) const {
    const auto audio = load_audio_file(audio_file_path);
    return analyze(audio.view());
}

Analyzer create_analyzer(const std::vector<ExtractorConfig>& extractor_configs) {
    auto analyzer = Analyzer{};
    if (extractor_configs.empty()) {
        analyzer.register_default_extractors();
        return analyzer;
    }

    analyzer.configure_extractors(extractor_configs);
    return analyzer;
}

Analyzer create_default_analyzer() {
    auto analyzer = Analyzer{};
    analyzer.register_default_extractors();
    return analyzer;
}

AnalysisResult analyze_file(std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs) {
    return create_analyzer(extractor_configs).analyze_file(audio_file_path);
}

AnalysisResult analyze_file(const AnalysisConfig& config) {
    if (config.audio_file_path.empty()) {
        throw std::invalid_argument("Analysis config must include an audio_file path or receive one from the caller.");
    }

    return analyze_file(config.audio_file_path, config.extractors);
}

std::string placeholder_method() {
    return "afex placeholder method executed";
}

}
