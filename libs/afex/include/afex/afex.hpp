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
#include <vector>

namespace afex
{

namespace feature_names
{

inline constexpr std::string_view    rms                         = "rms";
inline constexpr std::string_view    rms_variance                = "rms_variance";
inline constexpr std::string_view    zcr                         = "zcr";
inline constexpr std::string_view    pitch                       = "pitch";
inline constexpr std::string_view    pitch_confidence            = "pitch_confidence";
inline constexpr std::string_view    spectral_centroid           = "spectral_centroid";
inline constexpr std::string_view    spectral_flux               = "spectral_flux";
inline constexpr std::string_view    onset_density               = "onset_density";
inline constexpr std::string_view    voice_activity_ratio        = "voice_activity_ratio";

}

struct AudioData
{
    std::span<const float>               samples;
    std::uint32_t                        sample_rate_hz              = 0;
    std::uint32_t                        channel_count               = 1;

    bool                                 empty                       () const;
    std::size_t                          frame_count                 () const;
    double                               duration_seconds            () const;
};

enum class FeatureStatus
{
    Complete,
    Failed,
};

struct FeatureResult
{
    std::string                          name;
    FeatureStatus                        status                      = FeatureStatus::Complete;
    std::optional<double>                value;
    std::vector<double>                  values;
    std::string                          unit;
    std::string                          note;
};

struct AnalysisResult
{
    std::uint32_t                        sample_rate_hz              = 0;
    std::uint32_t                        channel_count               = 1;
    std::size_t                          sample_count                = 0;
    std::size_t                          frame_count                 = 0;
    double                               duration_seconds            = 0.0;
    std::vector<FeatureResult>           features;

    const FeatureResult*                 find_feature                (std::string_view name) const;
};

class FeatureExtractor
{

public:
    virtual                              ~FeatureExtractor           () = default;
    virtual std::string_view             name                        () const = 0;
    virtual FeatureResult                extract                     (const AudioData& audio) const = 0;

};

using FeatureExtractorFn = std::function<FeatureResult(const AudioData& audio)>;

class CallbackFeatureExtractor final : public FeatureExtractor
{

public:
    explicit                             CallbackFeatureExtractor    (std::string name, FeatureExtractorFn extract_fn);

    std::string_view                     name                        () const override;
    FeatureResult                        extract                     (const AudioData& audio) const override;

private:
    std::string                          m_name;
    FeatureExtractorFn                   m_extract_fn;

};

class ExtractorRegistry
{

public:
    void                                 register_extractor          (std::unique_ptr<FeatureExtractor> extractor);
    void                                 register_extractor          (std::string name, FeatureExtractorFn extract_fn);
    bool                                 unregister_extractor        (std::string_view name);
    bool                                 contains                    (std::string_view name) const;
    std::vector<std::string>             names                       () const;
    std::vector<FeatureResult>           extract_all                 (const AudioData& audio) const;

private:
    std::vector<std::unique_ptr<FeatureExtractor>> m_extractors;

};

class Analyzer
{

public:
                                         Analyzer                    ();

    void                                 register_extractor          (std::unique_ptr<FeatureExtractor> extractor);
    void                                 register_extractor          (std::string name, FeatureExtractorFn extract_fn);
    void                                 register_default_extractors ();
    std::vector<std::string>             registered_extractors       () const;
    AnalysisResult                       analyze                     (const AudioData& audio) const;

private:
    ExtractorRegistry                    m_registry;

};

Analyzer                             create_default_analyzer     ();
std::string                          placeholder_method          ();

}
