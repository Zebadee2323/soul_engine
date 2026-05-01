add_library(afex
    src/afex.cpp
)

target_compile_features(afex PUBLIC cxx_std_20)
target_include_directories(afex
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
