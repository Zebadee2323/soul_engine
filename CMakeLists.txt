cmake_minimum_required(VERSION 3.20)

project(soul_engine LANGUAGES CXX)

add_subdirectory(libs/mlp)
add_subdirectory(libs/afex)
add_subdirectory(apps/mlp-xor)
add_subdirectory(apps/afex-cli)
