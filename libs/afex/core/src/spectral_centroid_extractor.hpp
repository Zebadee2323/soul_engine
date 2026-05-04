#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include "extractors.hpp"

namespace afex
{

SpectralCentroidExtractorSettings make_spectral_centroid_extractor_settings  (const ExtractorParameters& parameters);
FeatureResult                     extract_spectral_centroid                  (const AudioData& audio, const ExtractorParameters& parameters);
FeatureResult                     extract_spectral_centroid                  (const AudioData& audio, const SpectralCentroidExtractorSettings& settings);

}
