cmake_minimum_required(VERSION 3.20)

project(mlp_xor LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_executable(mlp_xor
  src/main.cpp
  src/math.cpp
  src/dense_layer.cpp
  src/mlp.cpp
)

target_compile_features(mlp_xor PRIVATE cxx_std_17)
target_include_directories(mlp_xor PRIVATE src)
target_link_libraries(mlp_xor PRIVATE Eigen3::Eigen)
