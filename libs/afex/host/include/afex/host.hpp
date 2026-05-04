#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace afex
{

struct LoadedAudioData
{
    std::vector<float>                  samples;
    std::uint32_t                       sample_rate_hz              = 0;
    std::uint32_t                       channel_count               = 1;

    AudioData                           view                        () const;
};

struct AnalysisConfig
{
    std::string                         audio_file_path;
    std::vector<ExtractorConfig>        extractors;
};

LoadedAudioData                         load_audio_file             (std::string_view audio_file_path);
AnalysisConfig                          load_analysis_config_yaml   (std::string_view config_file_path);
Status                                  load_audio_file             (std::string_view audio_file_path, LoadedAudioData& audio);
Status                                  load_analysis_config_yaml   (std::string_view config_file_path, AnalysisConfig& config);
AnalysisResult                          analyze_file                (std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs);
AnalysisResult                          analyze_file                (std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs,
                                                                       const AnalyzeSettings& settings);
AnalysisResult                          analyze_file                (const AnalysisConfig& config);
AnalysisResult                          analyze_file                (const AnalysisConfig& config, const AnalyzeSettings& settings);

}
