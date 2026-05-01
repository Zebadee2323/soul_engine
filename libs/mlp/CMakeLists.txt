find_package(Eigen3 REQUIRED)

add_library(mlp
  src/math.cpp
  src/dense_layer.cpp
  src/mlp.cpp
)

target_compile_features(mlp PUBLIC cxx_std_17)
target_include_directories(mlp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(mlp PUBLIC Eigen3::Eigen)
