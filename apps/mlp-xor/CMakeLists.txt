cmake_minimum_required(VERSION 3.20)

project(mlp_xor LANGUAGES CXX)

add_executable(mlp_xor
  src/main.cpp
)

target_compile_features(mlp_xor PRIVATE cxx_std_17)
target_link_libraries(mlp_xor PRIVATE mlp)
