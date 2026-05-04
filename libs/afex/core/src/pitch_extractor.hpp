#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"

namespace afex
{

PitchExtractorSettings  make_pitch_extractor_settings    (const ExtractorParameters& parameters);
FeatureResult           extract_pitch                    (const AudioData& audio, const ExtractorParameters& parameters);
FeatureResult           extract_pitch                    (const AudioData& audio, const PitchExtractorSettings& settings);
FeatureResult           extract_pitch_confidence         (const AudioData& audio, const ExtractorParameters& parameters);
FeatureResult           extract_pitch_confidence         (const AudioData& audio, const PitchExtractorSettings& settings);

}
