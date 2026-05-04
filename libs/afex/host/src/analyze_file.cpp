#include <afex/host.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <stdexcept>
#include <string_view>
#include <vector>

namespace afex
{

AnalysisResult analyze_file(std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs) {
    return analyze_file(audio_file_path, extractor_configs, AnalyzeSettings{});
}

AnalysisResult analyze_file(std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs,
                            const AnalyzeSettings& settings) {
    const auto audio = load_audio_file(audio_file_path);
    return create_analyzer(extractor_configs).analyze(audio.view(), settings);
}

AnalysisResult analyze_file(const AnalysisConfig& config) {
    return analyze_file(config, AnalyzeSettings{});
}

AnalysisResult analyze_file(const AnalysisConfig& config, const AnalyzeSettings& settings) {
    if (config.audio_file_path.empty()) {
#if AFEX_ENABLE_EXCEPTIONS
        throw std::invalid_argument("Analysis config must include an audio_file path or receive one from the caller.");
#endif
        return AnalysisResult{
            .features = {
                FeatureResult{
                    .name = "analysis",
                    .status = FeatureStatus::Failed,
                    .note = "Analysis config must include an audio_file path or receive one from the caller.",
                },
            },
        };
    }

    return analyze_file(config.audio_file_path, config.extractors, settings);
}

}
