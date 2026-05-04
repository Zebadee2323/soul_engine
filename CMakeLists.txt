cmake_minimum_required(VERSION 3.20)

project(soul_engine LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(libs/mlp)
add_subdirectory(libs/afex)
add_subdirectory(apps/mlp-xor)
add_subdirectory(apps/afex-cli)

add_subdirectory(python/lib/pyafex)
add_subdirectory(python/apps/pyafex-cli)
