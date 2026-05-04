#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"

namespace afex
{

FeatureResult   extract_rms             (const AudioData& audio, const ExtractorParameters& parameters);
FeatureResult   extract_rms_variance    (const AudioData& audio, const ExtractorParameters& parameters);

}
