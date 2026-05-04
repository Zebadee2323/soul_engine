#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef AFEX_ENABLE_EXCEPTIONS
#    define AFEX_ENABLE_EXCEPTIONS 0
#endif

#ifndef AFEX_ENABLE_SPECTRAL_EXTRACTORS
#    define AFEX_ENABLE_SPECTRAL_EXTRACTORS 1
#endif

#ifndef AFEX_EMBEDDED
#    define AFEX_EMBEDDED 0
#endif

namespace afex
{

enum class AfexError
{
    None,
    InvalidArgument,
    UnknownExtractor,
    FileOpenFailed,
    UnsupportedAudioFile,
    ParseFailed,
};

struct Status
{
    AfexError                           error                       = AfexError::None;
    std::string                         message;

    bool                                ok                          () const;
};

namespace feature_names
{

inline constexpr std::string_view       rms                         = "rms";
inline constexpr std::string_view       rms_variance                = "rms_variance";
inline constexpr std::string_view       zcr                         = "zcr";
inline constexpr std::string_view       pitch                       = "pitch";
inline constexpr std::string_view       pitch_confidence            = "pitch_confidence";
inline constexpr std::string_view       spectral_centroid           = "spectral_centroid";
inline constexpr std::string_view       spectral_flux               = "spectral_flux";
inline constexpr std::string_view       onset_density               = "onset_density";
inline constexpr std::string_view       voice_activity_ratio        = "voice_activity_ratio";

}

using ExtractorParameters = std::unordered_map<std::string, double>;

struct AudioData
{
    std::span<const float>              samples;
    std::uint32_t                       sample_rate_hz              = 0;
    std::uint32_t                       channel_count               = 1;

    bool                                empty                       () const;
    std::size_t                         frame_count                 () const;
    double                              duration_seconds            () const;
};

struct LoadedAudioData
{
    std::vector<float>                  samples;
    std::uint32_t                       sample_rate_hz              = 0;
    std::uint32_t                       channel_count               = 1;

    AudioData                           view                        () const;
};

struct ExtractorConfig
{
    std::string                         name;
    ExtractorParameters                 parameters;
};

struct AnalysisConfig
{
    std::string                         audio_file_path;
    std::vector<ExtractorConfig>        extractors;
};

struct AnalyzeSettings
{
    bool                                keep_intermediate_values    = false;
    std::size_t                         max_frame_size              = 2048;
};

enum class FeatureStatus
{
    Complete,
    Failed,
};

struct FeatureResult
{
    std::string                         name;
    FeatureStatus                       status                      = FeatureStatus::Complete;
    std::optional<double>               value;
    std::vector<double>                 values;
    std::string                         unit;
    std::string                         note;
};

struct AnalysisResult
{
    std::uint32_t                       sample_rate_hz              = 0;
    std::uint32_t                       channel_count               = 1;
    std::size_t                         sample_count                = 0;
    std::size_t                         frame_count                 = 0;
    double                              duration_seconds            = 0.0;
    std::vector<FeatureResult>          features;

    const FeatureResult*                find_feature                (std::string_view name) const;
    bool                                ok                          () const;
};

class FeatureExtractor
{

public:
    virtual                             ~FeatureExtractor           () = default;
    virtual std::string_view            name                        () const = 0;
    virtual FeatureResult               extract                     (const AudioData& audio) const = 0;

};

using FeatureExtractorFn = std::function<FeatureResult(const AudioData& audio)>;

class CallbackFeatureExtractor final : public FeatureExtractor
{

public:
    explicit                            CallbackFeatureExtractor    (std::string name, FeatureExtractorFn extract_fn);

    std::string_view                    name                        () const override;
    FeatureResult                       extract                     (const AudioData& audio) const override;

private:
    std::string                         m_name;
    FeatureExtractorFn                  m_extract_fn;

};

class ExtractorRegistry
{

public:
    void                                clear                       ();
    void                                register_extractor          (std::unique_ptr<FeatureExtractor> extractor);
    void                                register_extractor          (std::string name, FeatureExtractorFn extract_fn);
    bool                                unregister_extractor        (std::string_view name);
    bool                                contains                    (std::string_view name) const;
    std::vector<std::string>            names                       () const;
    std::vector<FeatureResult>          extract_all                 (const AudioData& audio) const;

private:
    std::vector<std::unique_ptr<FeatureExtractor>> m_extractors;

};

class Analyzer
{

public:
                                        Analyzer                    ();

    void                                register_extractor          (std::unique_ptr<FeatureExtractor> extractor);
    void                                register_extractor          (std::string name, FeatureExtractorFn extract_fn);
    void                                register_builtin_extractor  (const ExtractorConfig& config);
    void                                register_default_extractors ();
    void                                configure_extractors        (const std::vector<ExtractorConfig>& configs);
    std::vector<std::string>            registered_extractors       () const;
    AnalysisResult                      analyze                     (const AudioData& audio) const;
    AnalysisResult                      analyze                     (const AudioData& audio, const AnalyzeSettings& settings) const;
    AnalysisResult                      analyze_file                (std::string_view audio_file_path) const;

private:
    ExtractorRegistry                   m_registry;

};

LoadedAudioData                         load_audio_file             (std::string_view audio_file_path);
AnalysisConfig                          load_analysis_config_yaml   (std::string_view config_file_path);
Status                                  load_audio_file             (std::string_view audio_file_path, LoadedAudioData& audio);
Status                                  load_analysis_config_yaml   (std::string_view config_file_path, AnalysisConfig& config);
std::vector<std::string>                builtin_feature_names       ();
Analyzer                                create_analyzer             (const std::vector<ExtractorConfig>& extractor_configs);
Analyzer                                create_default_analyzer     ();
AnalysisResult                          analyze_file                (std::string_view audio_file_path, const std::vector<ExtractorConfig>& extractor_configs);
AnalysisResult                          analyze_file                (const AnalysisConfig& config);
AnalysisResult                          analyze                     (const AudioData& audio, const std::vector<ExtractorConfig>& extractor_configs,
                                                                       const AnalyzeSettings& settings = {});
std::string                             placeholder_method          ();

}
