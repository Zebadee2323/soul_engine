add_library(afex_host
  src/audio_file.cpp
  src/yaml_config.cpp
)

target_compile_features(afex_host PUBLIC cxx_std_20)
set_target_properties(afex_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(afex_host PUBLIC afex)
