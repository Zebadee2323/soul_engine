# CMake Cheatsheet and Best Practices

This is a practical reference for small C++ projects like `apps/mlp-xor`.

## Basic Workflow

Configure:

```bash
cmake -S . -B build
```

Build:

```bash
cmake --build build
```

Run:

```bash
./build/my_app
```

Common variants:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target my_app
cmake --build build -j
cmake --fresh -S . -B build
```

## Minimal Project Template

```cmake
cmake_minimum_required(VERSION 3.20)

project(my_project LANGUAGES CXX)

add_executable(my_app
  src/main.cpp
  src/foo.cpp
)

target_compile_features(my_app PRIVATE cxx_std_17)
target_include_directories(my_app PRIVATE src)
target_compile_options(my_app PRIVATE -Wall -Wextra -Wpedantic)
```

## Core Commands

### `project`

Defines the project and enabled languages.

```cmake
project(my_project LANGUAGES CXX)
```

### `add_executable`

Defines an executable target.

```cmake
add_executable(my_app src/main.cpp)
```

### `add_library`

Defines a library target.

```cmake
add_library(my_lib STATIC src/my_lib.cpp)
add_library(my_shared SHARED src/my_shared.cpp)
```

### `target_include_directories`

Adds header include paths.

```cmake
target_include_directories(my_app PRIVATE src)
```

### `target_compile_features`

Requests language features or a minimum C++ standard.

```cmake
target_compile_features(my_app PRIVATE cxx_std_17)
```

### `target_compile_options`

Adds compiler flags for one target.

```cmake
target_compile_options(my_app PRIVATE -Wall -Wextra -Wpedantic)
```

### `target_link_libraries`

Links a target to libraries.

```cmake
target_link_libraries(my_app PRIVATE my_lib)
```

### `add_subdirectory`

Pulls another CMake project into the build.

```cmake
add_subdirectory(libs/some_lib)
```

### `find_package`

Finds installed dependencies.

```cmake
find_package(Threads REQUIRED)
target_link_libraries(my_app PRIVATE Threads::Threads)
```

## Visibility: `PRIVATE`, `PUBLIC`, `INTERFACE`

- `PRIVATE`: only this target uses the setting
- `PUBLIC`: this target uses it, and consumers inherit it
- `INTERFACE`: only consumers inherit it

Example:

```cmake
target_include_directories(my_lib PUBLIC include)
```

If `my_app` links `my_lib`, it also gets `include`.

## Variables

Use variables when they simplify a file. Do not over-abstract small projects.

```cmake
set(MY_SOURCES
  src/main.cpp
  src/foo.cpp
)

add_executable(my_app ${MY_SOURCES})
```

## Testing

```cmake
enable_testing()

add_executable(my_test tests/test_main.cpp)
add_test(NAME my_test COMMAND my_test)
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

## Install

```cmake
install(TARGETS my_app DESTINATION bin)
```

## Example for `apps/mlp-xor`

```cmake
cmake_minimum_required(VERSION 3.20)

project(mlp_xor LANGUAGES CXX)

add_executable(mlp_xor
  src/main.cpp
  src/DenseLayer.cpp
  src/MLP.cpp
)

target_compile_features(mlp_xor PRIVATE cxx_std_17)
target_include_directories(mlp_xor PRIVATE src)
target_compile_options(mlp_xor PRIVATE -Wall -Wextra -Wpedantic)
```

Build just that app:

```bash
cmake -S apps/mlp-xor -B build/mlp-xor
cmake --build build/mlp-xor
```

## Best Practices

- Prefer target-based commands like `target_include_directories` over global commands like `include_directories`.
- Keep builds out of the source tree. Use a separate `build/` directory.
- Set a clear minimum CMake version with `cmake_minimum_required`.
- Declare the project languages explicitly.
- Set compile features per target with `target_compile_features`.
- Keep source lists explicit in small projects. It is easier to review and debug.
- Use `PRIVATE`, `PUBLIC`, and `INTERFACE` intentionally. They control transitive build behavior.
- Add warning flags per target instead of globally when possible.
- Split reusable code into libraries, then link executables against those libraries.
- Use `find_package` and imported targets when integrating external dependencies.
- Avoid unnecessary CMake metaprogramming. Prefer simple, readable files.

## Common Mistakes

- Using global include paths or compiler flags for the whole project without need.
- Mixing source files and generated build files in the same directory.
- Relying on implicit compiler defaults instead of setting the requested C++ standard.
- Using file globbing for sources in small projects, which can hide changes from review.
- Putting too much logic in one top-level `CMakeLists.txt` instead of splitting by module.

## Quick Reference

Configure:

```bash
cmake -S . -B build
```

Build:

```bash
cmake --build build
```

Debug build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```
