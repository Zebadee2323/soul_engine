#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"

namespace afex
{

OnsetDensityExtractorSettings make_onset_density_extractor_settings  (const ExtractorParameters& parameters);
FeatureResult                 extract_onset_density                  (const AudioData& audio, const ExtractorParameters& parameters);
FeatureResult                 extract_onset_density                  (const AudioData& audio, const OnsetDensityExtractorSettings& settings);

}
