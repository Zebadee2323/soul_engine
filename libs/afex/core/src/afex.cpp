#include <afex/afex.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"
#include "extractor_helpers.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace afex
{

namespace
{

thread_local const AnalyzeSettings* active_analyze_settings = nullptr;

class ActiveAnalyzeSettingsScope
{

public:
    explicit ActiveAnalyzeSettingsScope(const AnalyzeSettings& settings) :
        m_previous              (active_analyze_settings)
    {
        active_analyze_settings = &settings;
    }

    ~ActiveAnalyzeSettingsScope() {
        active_analyze_settings = m_previous;
    }

private:
    const AnalyzeSettings*      m_previous;

};

FeatureResult failed_feature_for_extractor(std::string_view name, std::string_view note) {
    return FeatureResult{
        .name = std::string{name},
        .status = FeatureStatus::Failed,
        .value = std::nullopt,
        .values = {},
        .unit = {},
#if AFEX_EMBEDDED
        .note = {},
#else
        .note = std::string{note},
#endif
    };
}

FeatureResult complete_feature_for_extractor(std::string_view name, double value, std::vector<double> values, std::string_view unit,
                                             std::string_view note) {
    return FeatureResult{
        .name = std::string{name},
        .status = FeatureStatus::Complete,
        .value = value,
        .values = std::move(values),
#if AFEX_EMBEDDED
        .note = {},
#else
        .unit = std::string{unit},
        .note = std::string{note},
#endif
    };
}

bool frame_exceeds_silence_threshold(const AudioData& audio, std::size_t frame_index) {
    constexpr auto silence_threshold = 1.0e-4f;

    const auto first_sample = frame_index * audio.channel_count;
    for (auto channel = std::size_t{0}; channel < audio.channel_count; ++channel) {
        if (std::abs(audio.samples[first_sample + channel]) > silence_threshold) {
            return true;
        }
    }

    return false;
}

AudioData trim_silence(const AudioData& audio) {
    if (audio.channel_count == 0 || audio.empty()) {
        return audio;
    }

    const auto frame_count = audio.frame_count();
    auto first_audible_frame = std::size_t{0};
    while (first_audible_frame < frame_count && !frame_exceeds_silence_threshold(audio, first_audible_frame)) {
        ++first_audible_frame;
    }

    if (first_audible_frame == frame_count) {
        return AudioData{
            .samples = audio.samples.subspan(0, 0),
            .sample_rate_hz = audio.sample_rate_hz,
            .channel_count = audio.channel_count,
        };
    }

    auto one_past_last_audible_frame = frame_count;
    while (one_past_last_audible_frame > first_audible_frame
           && !frame_exceeds_silence_threshold(audio, one_past_last_audible_frame - 1)) {
        --one_past_last_audible_frame;
    }

    const auto first_sample = first_audible_frame * audio.channel_count;
    const auto sample_count = (one_past_last_audible_frame - first_audible_frame) * audio.channel_count;
    return AudioData{
        .samples = audio.samples.subspan(first_sample, sample_count),
        .sample_rate_hz = audio.sample_rate_hz,
        .channel_count = audio.channel_count,
    };
}

bool has_max_frame_size_limit(const AnalyzeSettings& settings) {
    return settings.max_frame_size != 0;
}

bool has_max_frame_length_limit(const AnalyzeSettings& settings) {
    return settings.max_frame_length != 0.0;
}

bool frame_limit_settings_valid(const AnalyzeSettings& settings) {
    if (!std::isfinite(settings.max_frame_length) || settings.max_frame_length < 0.0) {
        return false;
    }

    return !(has_max_frame_size_limit(settings) && has_max_frame_length_limit(settings));
}

AnalysisResult failed_analysis_result(std::string_view note) {
    return AnalysisResult{
        .features = {
            FeatureResult{
                .name = "analysis",
                .status = FeatureStatus::Failed,
                .note = std::string{note},
            },
        },
    };
}

}

bool Status::ok() const {
    return error == AfexError::None;
}

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

bool AnalysisResult::ok() const {
    return std::all_of(features.begin(), features.end(), [](const FeatureResult& feature) {
        return feature.status == FeatureStatus::Complete;
    });
}

const AnalyzeSettings& current_analyze_settings() {
    static const auto defaults = AnalyzeSettings{};
    if (active_analyze_settings == nullptr) {
        return defaults;
    }

    return *active_analyze_settings;
}

FeatureResult complete_feature(std::string_view name, double value, std::vector<double> values, std::string_view unit, std::string_view note) {
    if (!current_analyze_settings().include_feature_values) {
        values.clear();
    }

    return complete_feature_for_extractor(name, value, std::move(values), unit, note);
}

FeatureResult failed_feature(std::string_view name, std::string_view note) {
    return failed_feature_for_extractor(name, note);
}

CallbackFeatureExtractor::CallbackFeatureExtractor(std::string name, FeatureExtractorFn extract_fn) :
    m_name                  (std::move(name)),
    m_extract_fn            (std::move(extract_fn))
{
    if (m_name.empty()) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Feature extractor name cannot be empty.");
#endif
    }

    if (!m_extract_fn) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Feature extractor callback cannot be empty.");
#endif
    }
}

std::string_view CallbackFeatureExtractor::name() const {
    return m_name;
}

FeatureResult CallbackFeatureExtractor::extract(const AudioData& audio) const {
    if (!m_extract_fn) {
        return failed_feature_for_extractor(m_name, "Feature extractor callback is empty.");
    }

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
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Feature extractor cannot be null.");
#endif
        return;
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
    auto configs = std::vector<ExtractorConfig>{};
    for (const auto& name : builtin_feature_names()) {
        configs.push_back(ExtractorConfig{.name = name});
    }

    configure_extractors(configs);
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
    return analyze(audio, AnalyzeSettings{});
}

AnalysisResult Analyzer::analyze(const AudioData& audio, const AnalyzeSettings& settings) const {
    if (!frame_limit_settings_valid(settings)) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("AnalyzeSettings must use either max_frame_size or max_frame_length, not both, and max_frame_length must be non-negative.");
#else
        return failed_analysis_result("AnalyzeSettings must use either max_frame_size or max_frame_length, not both, and max_frame_length must be non-negative.");
#endif
    }

    const auto settings_scope = ActiveAnalyzeSettingsScope{settings};
    const auto analysis_audio = settings.trim_silence ? trim_silence(audio) : audio;
    auto workspace = AnalysisWorkspace{analysis_audio};
    const auto workspace_scope = ActiveAnalysisWorkspaceScope{workspace};
    auto features = m_registry.extract_all(analysis_audio);

    return AnalysisResult{
        .sample_rate_hz = analysis_audio.sample_rate_hz,
        .channel_count = analysis_audio.channel_count,
        .sample_count = analysis_audio.samples.size(),
        .frame_count = analysis_audio.frame_count(),
        .duration_seconds = analysis_audio.duration_seconds(),
        .features = std::move(features),
    };
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

std::vector<std::string> builtin_feature_names() {
    auto names = std::vector<std::string>{
        std::string{feature_names::rms},
        std::string{feature_names::rms_variance},
        std::string{feature_names::zcr},
    };

    names.insert(
        names.end(),
        {
        std::string{feature_names::pitch},
        std::string{feature_names::pitch_confidence},
        std::string{feature_names::spectral_centroid},
        std::string{feature_names::spectral_flux},
        std::string{feature_names::onset_density},
        std::string{feature_names::voice_activity_ratio},
        }
    );

    return names;
}

AnalysisResult analyze(const AudioData& audio, const std::vector<ExtractorConfig>& extractor_configs, const AnalyzeSettings& settings) {
    return create_analyzer(extractor_configs).analyze(audio, settings);
}

std::string placeholder_method() {
#if AFEX_EMBEDDED
    return {};
#else
    return "afex placeholder method executed";
#endif
}

}
