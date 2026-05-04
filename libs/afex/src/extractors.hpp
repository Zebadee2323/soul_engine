#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>

namespace afex
{

FeatureResult               extract_rms             (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult               extract_rms_variance    (const AudioData& audio, const ExtractorParameters& parameters = {});
FeatureResult               extract_zcr             (const AudioData& audio, const ExtractorParameters& parameters = {});
std::unique_ptr<FeatureExtractor> create_builtin_extractor(const ExtractorConfig& config);

}
