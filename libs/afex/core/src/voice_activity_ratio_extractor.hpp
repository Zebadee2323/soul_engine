#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"

namespace afex
{

FeatureResult   extract_voice_activity_ratio  (const AudioData& audio, const ExtractorParameters& parameters);

}
