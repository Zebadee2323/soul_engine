#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>

namespace afex
{

FeatureResult               extract_rms             (const AudioData& audio);
FeatureResult               extract_rms_variance    (const AudioData& audio);
FeatureResult               extract_zcr             (const AudioData& audio);

}
