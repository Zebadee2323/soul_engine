#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace afex
{

double                  parameter_or                         (const ExtractorParameters& parameters, std::string_view name, double fallback);
std::size_t             positive_size_parameter              (const ExtractorParameters& parameters, std::string_view name, std::size_t fallback);
double                  non_negative_parameter               (const ExtractorParameters& parameters, std::string_view name, double fallback);
NormalizationSettings   normalization_settings               (const ExtractorParameters& parameters, double default_minimum, double default_maximum);
NormalizationSettings   normalization_settings               (const ExtractorParameters& parameters, std::string_view min_name, std::string_view max_name,
                                                                 double default_minimum, double default_maximum);
FrameParameterSettings  frame_parameter_settings             (const ExtractorParameters& parameters, std::size_t default_size, std::size_t default_hop);
bool                    normalization_range_valid            (double minimum, double maximum);
double                  normalize_to_unit_interval           (double value, double minimum, double maximum);
void                    normalize_values_to_unit_interval    (std::vector<double>& values, double minimum, double maximum);

struct FrameSettings
{
    std::size_t         size;
    std::size_t         hop;
};

class AnalysisWorkspace
{

public:
    explicit                            AnalysisWorkspace       (const AudioData& audio);

    bool                                owns                    (const AudioData& audio) const;
    const std::vector<double>&          mono                    ();
    const std::vector<std::size_t>&     offsets                 (std::size_t sample_count, std::size_t frame_size, std::size_t hop_size);
    const std::vector<double>&          spectrum                (std::size_t offset, std::size_t frame_size);
    const PitchSummary*                 pitch                   (const PitchExtractorSettings& settings) const;
    void                                set_pitch               (const PitchExtractorSettings& settings, PitchSummary summary);

private:
    struct OffsetsCache
    {
        std::size_t                     sample_count             = 0;
        std::size_t                     frame_size               = 0;
        std::size_t                     hop_size                 = 0;
        std::vector<std::size_t>        values;
    };

    struct SpectrumCache
    {
        std::size_t                     offset                   = 0;
        std::size_t                     frame_size               = 0;
        std::vector<double>             values;
    };

    const AudioData*                    m_audio                  = nullptr;
    std::vector<double>                 m_mono;
    bool                                m_mono_ready             = false;
    std::vector<OffsetsCache>           m_offsets_caches;
    std::vector<SpectrumCache>          m_spectrum_caches;
    std::optional<PitchExtractorSettings> m_pitch_settings;
    std::optional<PitchSummary>         m_pitch_summary;

};

class ActiveAnalysisWorkspaceScope
{

public:
    explicit                            ActiveAnalysisWorkspaceScope  (AnalysisWorkspace& workspace);
                                        ~ActiveAnalysisWorkspaceScope ();

private:
    AnalysisWorkspace*                  m_previous;

};

std::vector<double>                     downmix_mono                         (const AudioData& audio);
const std::vector<double>&              mono_audio                           (const AudioData& audio);
FrameSettings                           frame_settings                       (const AudioData& audio, const FrameParameterSettings& parameters);
FrameSettings                           frame_settings                       (const AudioData& audio, const ExtractorParameters& parameters, std::size_t default_size,
                                                                                 std::size_t default_hop);
bool                                    frame_settings_valid                 (const FrameSettings& settings);
double                                  frame_rms                            (const std::vector<double>& mono, std::size_t offset, std::size_t size);
std::vector<std::size_t>                frame_offsets                        (std::size_t sample_count, std::size_t frame_size, std::size_t hop_size);
const std::vector<std::size_t>&         cached_frame_offsets                 (const AudioData& audio, std::size_t sample_count, std::size_t frame_size,
                                                                                 std::size_t hop_size);
std::vector<double>                     magnitude_spectrum                   (const std::vector<double>& mono, std::size_t offset, std::size_t frame_size);
const std::vector<double>&              cached_magnitude_spectrum            (const AudioData& audio, std::size_t offset, std::size_t frame_size);
const PitchSummary*                     cached_pitch_summary                 (const AudioData& audio, const PitchExtractorSettings& settings);
void                                    cache_pitch_summary                  (const AudioData& audio, const PitchExtractorSettings& settings, PitchSummary summary);

}
