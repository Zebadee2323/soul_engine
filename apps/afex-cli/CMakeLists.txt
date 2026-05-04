cmake_minimum_required(VERSION 3.20)

project(afex_cli LANGUAGES CXX)

add_executable(afex_cli
  src/main.cpp
)

target_compile_features(afex_cli PRIVATE cxx_std_20)
target_link_libraries(afex_cli PRIVATE afex_host)
