# C++ Code Style Guide

This guide captures the C++ style used by `libs/mlp`. Use it for new C++ code in this repo so headers, sources, naming, and formatting stay consistent.

## File Layout

Prefer this order in `.cpp` and `.hpp` files:

```cpp
#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <standard_header>
// --------------------------------------------------------------------------------------------------------------------
#include <third_party/header.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <project/header.hpp>

namespace my_namespace
{

// declarations

}
```

Guidelines:

- Use `#pragma once` in headers.
- Group includes with the long separator comment:

  ```cpp
  // --------------------------------------------------------------------------------------------------------------------
  ```

- Put standard library includes before third-party and project includes in headers.
- Put the matching project header first in source files when practical.
- Keep each library under its own namespace, for example `namespace mlp`.
- Use the Allman namespace style:

  ```cpp
  namespace mlp
  {

  }
  ```

## Naming

Use clear, descriptive names.

| Thing | Style | Example |
| --- | --- | --- |
| Namespaces | lower case | `mlp` |
| Classes / structs | PascalCase | `DenseLayer`, `DenseLayerConfig` |
| Enums | PascalCase | `ActivationType` |
| Enum values | PascalCase | `None`, `Relu`, `Sigmoid`, `TanH` |
| Functions | snake_case | `apply_gradients`, `set_random_matrix` |
| Variables | snake_case | `learning_rate`, `total_loss` |
| Members | `m_` + snake_case | `m_weights`, `m_input_size` |
| Type aliases | PascalCase or existing library style | `MatrixXf`, `VectorXf` |

## Classes and Structs

Use `struct` for simple data bundles with public fields:

```cpp
struct Sample
{
    VectorXf                    x;
    VectorXf                    y_target;
};
```

Use `class` for types with behavior and private state:

```cpp
class DenseLayer
{

public:
    explicit                    DenseLayer          (const DenseLayerConfig& config);
    mlp::VectorXf               forward             (const mlp::VectorXf& x);

private:
    Eigen::Index                m_input_size;
    mlp::MatrixXf               m_weights;

};
```

Guidelines:

- Put `public` before `private`.
- Keep data members private unless the type is a plain config/data struct.
- Prefix private member variables with `m_`.
- Mark single-argument constructors `explicit`.
- Group related public methods and private members with separator comments when it improves readability.

## Function Declarations

Headers align types, names, and call/initializer suffixes into columns. Use one consistent column grid across a header where practical, including public methods, private members, free functions, and constants.

Choose the smallest column widths that fit the declarations being aligned:

- The name column starts at the next 4-space boundary after the longest type/specifier in the aligned file or declaration block.
- The call/initializer column starts at the next 4-space boundary after the longest declaration name in that same grid.
- Do not add arbitrary extra padding once those minimum 4-space-aligned widths are met.
- Ignore excessively long type names when choosing the shared type-column width; let those declarations use a single space before the name instead of forcing every other declaration to shift right.
- For plain data members without an initializer, keep the semicolon directly after the name; do not pad out to the initializer column.
- Keep the same relative columns at each indentation level. Class members will be indented, but their type/name/suffix spacing should use the same grid as the rest of the header.

```cpp
float                                          mse                        (const mlp::VectorXf& prediction, const mlp::VectorXf& target);
mlp::VectorXf                                  mse_derivative             (const mlp::VectorXf& prediction, const mlp::VectorXf& target);
std::vector<std::unique_ptr<FeatureExtractor>> m_extractors;
```

For long declarations, wrap parameters onto the next line and align continuation indentation:

```cpp
static void                                    train                      (Mlp& mlp, const std::vector<Sample>& dataset, size_t epochs,
                                                                             float learning_rate);
```

## Function Definitions

Use compact braces for functions:

```cpp
mlp::VectorXf DenseLayer::forward(const mlp::VectorXf& x) {
    assert(x.size() == m_input_size);
    m_last_input = x;
    m_last_z = m_weights * x + m_biases;
    m_last_activation = apply_activation(m_last_z, m_activation_type);
    return m_last_activation;
}
```

Guidelines:

- Opening brace goes on the same line for functions, loops, `if`, and `switch` blocks.
- Indent with 4 spaces.
- Prefer one statement per line.
- Add spaces around binary operators: `a + b`, `epoch < epochs`.
- Do not add spaces inside parentheses.
- Prefer early assertions for preconditions.

## Constructors

Use initializer lists for member initialization. In library code, align initialized members vertically:

```cpp
DenseLayer::DenseLayer(Eigen::Index input_size, Eigen::Index output_size, ActivationType activation_type) :
    m_input_size            (input_size),
    m_output_size           (output_size),
    m_activation_type       (activation_type),
    m_weights               (output_size, input_size),
    m_biases                (output_size)
{
    auto rng = std::mt19937();
    randomize_weights(rng, -1.0f, 1.0f);
}
```

Guidelines:

- Initialize members in the same order they are declared in the class.
- Use constructor delegation when it avoids duplication.
- Keep constructor bodies for setup that cannot be expressed in the initializer list.

## Types and Constants

Guidelines:

- Use `std::size_t` for container sizes and counts.
- Use `float` literals with an `f` suffix in float code: `0.0f`, `1.0f`, `2.0f`.
- Prefer `const` for values that should not change:

  ```cpp
  const bool should_print = epoch % print_interval == 0 || epoch == epochs - 1;
  ```

- Use `auto` when the type is obvious or verbose:

  ```cpp
  auto gradient = d_loss;
  auto rng = std::mt19937();
  ```

## Parameters and References

Use references intentionally:

```cpp
void set_random_vector(mlp::VectorXf& vector, std::mt19937& rng, float min, float max);
mlp::VectorXf mse_derivative(const mlp::VectorXf& prediction, const mlp::VectorXf& target);
```

Guidelines:

- Pass large read-only objects by `const&`.
- Pass mutable output/input-output objects by non-const `&`.
- Pass small scalar values by value.
- Return new Eigen vectors/matrices by value; Eigen handles this well.

## Loops and Control Flow

Prefer range-based loops for containers:

```cpp
for (DenseLayer& layer : m_layers) {
    a = layer.forward(a);
}
```

Use index loops when the index matters:

```cpp
for (std::size_t i = 0; i < configs.size(); ++i) {
    assert(i == 0 || configs[i - 1].output_size == configs[i].input_size);
    m_layers.emplace_back(configs[i]);
}
```

Use reverse iterators for reverse traversal:

```cpp
for (auto layer = m_layers.rbegin(); layer != m_layers.rend(); ++layer) {
    gradient = layer->backward(gradient);
}
```

Switch statements should cover all enum values:

```cpp
switch (activation_type) {
    case ActivationType::None:
        return z;
    case ActivationType::Relu:
        return relu(z);
    case ActivationType::Sigmoid:
        return sigmoid(z);
    case ActivationType::TanH:
        return tanh(z);
}
```

## Assertions and Error Handling

Use `assert` for internal invariants and programming mistakes:

```cpp
assert(prediction.size() == target.size());
assert(x.size() == m_input_size);
```

Guidelines:

- Include `<cassert>` in source files that use `assert`.
- Use `assert` for conditions that should never fail in correct code.
- For user input, file IO, or runtime recoverable errors, prefer explicit error handling instead of `assert`.
- Do not use c++ exceptions.

## Comments

Keep comments useful and sparse.

Good comments explain sections, intent, or non-obvious behavior:

```cpp
// --------------------------------------------------------------------------------------------------------------------
// Loss helpers
```

Avoid comments that only repeat the next line of code.

## CMake Style

For CMake files, prefer target-based modern CMake:

```cmake
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
```

Guidelines:

- Keep source lists explicit.
- Use `target_compile_features` instead of relying on compiler defaults.
- Use `target_include_directories` and `target_link_libraries` with `PUBLIC`, `PRIVATE`, or `INTERFACE` deliberately.
- Link imported targets such as `Eigen3::Eigen` when available.

## Quick Checklist

Before committing C++ code, check that:

- Headers use `#pragma once`.
- Includes are grouped and ordered consistently.
- Code is inside the correct namespace.
- Names follow the table above.
- Private members use `m_`.
- Constructors use initializer lists.
- Large inputs are passed by `const&`.
- Mutable inputs are passed by `&`.
- Float literals use `f` suffixes.
- Eigen dimensions use `Eigen::Index` where appropriate.
- Preconditions are guarded with `assert` where useful.
- CMake uses target-based commands.
