add_library(afex
    src/afex.cpp
    src/audio_file.cpp
    src/builtin_extractors.cpp
    src/rms_extractor.cpp
    src/spectral_voice_extractors.cpp
    src/yaml_config.cpp
    src/zcr_extractor.cpp
)

target_compile_features(afex PUBLIC cxx_std_20)
set_target_properties(afex PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(afex
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
