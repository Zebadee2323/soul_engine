add_library(afex_host
  src/analyze_file.cpp
  src/audio_file.cpp
  src/yaml_config.cpp
)

target_compile_features(afex_host PUBLIC cxx_std_20)
set_target_properties(afex_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(afex_host
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(afex_host PUBLIC afex)
