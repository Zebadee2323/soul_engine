# Eigen C++ Cheatsheet

This is a practical quick reference for using Eigen in small C++ projects like `apps/mlp-xor`.

## Include and CMake

Include the dense matrix/vector API:

```cpp
#include <Eigen/Dense>
```

Link Eigen in CMake:

```cmake
find_package(Eigen3 REQUIRED)
target_link_libraries(my_app PRIVATE Eigen3::Eigen)
```

## Common Types

Dynamic size:

```cpp
Eigen::VectorXf v;
Eigen::MatrixXf m;
```

Fixed size:

```cpp
Eigen::Vector2f v2;
Eigen::Vector3f v3;
Eigen::Matrix2f m2;
Eigen::Matrix3f m3;
```

General form:

```cpp
Eigen::Matrix<float, 3, 3> a;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b;
```

Useful mental model:

- `VectorXf` means "dynamic-size vector of `float`".
- `MatrixXf` means "dynamic-size matrix of `float`".
- Fixed-size types are often simpler and faster for tiny known shapes.

## Construction and Initialization

Default construction:

```cpp
Eigen::VectorXf v;   // size 0
Eigen::MatrixXf m;   // 0 x 0
```

Create with size:

```cpp
Eigen::VectorXf v(4);
Eigen::MatrixXf m(3, 2);
```

Fill with comma initialization:

```cpp
Eigen::Vector2f v2;
v2 << 1.0f, 2.0f;

Eigen::Matrix2f m2;
m2 << 1.0f, 2.0f,
      3.0f, 4.0f;
```

Resize dynamic objects:

```cpp
v.resize(8);
m.resize(5, 3);
```

## Common Factory Helpers

Zeros:

```cpp
Eigen::VectorXf v = Eigen::VectorXf::Zero(4);
Eigen::MatrixXf m = Eigen::MatrixXf::Zero(3, 2);
```

Ones:

```cpp
Eigen::VectorXf v = Eigen::VectorXf::Ones(4);
Eigen::MatrixXf m = Eigen::MatrixXf::Ones(3, 2);
```

Constant value:

```cpp
Eigen::VectorXf v = Eigen::VectorXf::Constant(4, 0.5f);
Eigen::MatrixXf m = Eigen::MatrixXf::Constant(3, 2, -1.0f);
```

Identity:

```cpp
Eigen::Matrix3f i = Eigen::Matrix3f::Identity();
Eigen::MatrixXf dyn_i = Eigen::MatrixXf::Identity(4, 4);
```

Random:

```cpp
Eigen::VectorXf v = Eigen::VectorXf::Random(4);     // values in [-1, 1]
Eigen::MatrixXf m = Eigen::MatrixXf::Random(3, 2);
```

## Element Access

Vectors:

```cpp
float a = v[0];
float b = v(1);
v[2] = 10.0f;
```

Matrices:

```cpp
float x = m(0, 1);
m(1, 0) = 3.0f;
```

Rule of thumb:

- Use `v(i)` and `m(r, c)` when you want Eigen-style indexing everywhere.
- `v[i]` is fine for vectors.

## Basic Arithmetic

Vector addition and subtraction:

```cpp
Eigen::VectorXf c = a + b;
Eigen::VectorXf d = a - b;
```

Scalar multiply and divide:

```cpp
Eigen::VectorXf x = 2.0f * v;
Eigen::VectorXf y = v / 3.0f;
```

Matrix-vector and matrix-matrix multiply:

```cpp
Eigen::VectorXf y = m * v;
Eigen::MatrixXf p = a * b;
```

Transpose:

```cpp
Eigen::MatrixXf mt = m.transpose();
```

Dot product:

```cpp
float d = a.dot(b);
```

Norms:

```cpp
float n1 = v.norm();
float n2 = v.squaredNorm();
v.normalize();              // modifies in place
Eigen::VectorXf u = v.normalized();  // returns normalized copy
```

For 3D vectors:

```cpp
Eigen::Vector3f c = a.cross(b);
```

## Coefficient-Wise Operations

Eigen distinguishes between linear algebra and element-wise math.

Element-wise multiply:

```cpp
Eigen::VectorXf z = a.array() * b.array();
```

Element-wise division:

```cpp
Eigen::VectorXf z = a.array() / b.array();
```

Apply scalar functions element-wise:

```cpp
Eigen::VectorXf z = v.array().sqrt();
Eigen::VectorXf s = v.array().sin();
Eigen::VectorXf e = v.array().exp();
```

Convert back to matrix/vector expression:

```cpp
Eigen::VectorXf z = (a.array() * b.array()).matrix();
```

Good default rule:

- Use `*` for matrix multiplication.
- Use `.array()` when you mean "do this to each element".

## Rows, Columns, and Blocks

Access a row or column:

```cpp
auto row0 = m.row(0);
auto col1 = m.col(1);
```

Assign a row or column:

```cpp
m.row(0) = Eigen::RowVector2f(1.0f, 2.0f);
m.col(1) = Eigen::VectorXf::Ones(m.rows());
```

Take a block:

```cpp
auto sub = m.block(1, 0, 2, 2);  // start row, start col, rows, cols
```

Head and tail of a vector:

```cpp
auto first3 = v.head(3);
auto last2 = v.tail(2);
```

Middle segment:

```cpp
auto mid = v.segment(2, 4);  // start index, length
```

## Aggregates

Sum, mean, min, max:

```cpp
float s = v.sum();
float avg = v.mean();
float mn = v.minCoeff();
float mx = v.maxCoeff();
```

For matrices:

```cpp
float s = m.sum();
float mn = m.minCoeff();
float mx = m.maxCoeff();
```

## Shape Information

```cpp
Eigen::Index n = v.size();
Eigen::Index rows = m.rows();
Eigen::Index cols = m.cols();
bool empty = (m.size() == 0);
```

Prefer `Eigen::Index` for Eigen dimensions and indices.

## Solving Systems

For a linear system `A x = b`:

```cpp
Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
```

For symmetric positive definite matrices:

```cpp
Eigen::VectorXf x = A.ldlt().solve(b);
```

Avoid computing an explicit inverse unless you truly need the inverse matrix itself.

## Array Type

If your work is mostly element-wise, Eigen also has array types:

```cpp
Eigen::ArrayXf a(4);
Eigen::ArrayXXf m(3, 2);
```

Example:

```cpp
Eigen::ArrayXf x(3);
x << 1.0f, 2.0f, 3.0f;

Eigen::ArrayXf y = x * x + 2.0f;
```

Rule of thumb:

- Use Eigen matrix/vector types for linear algebra.
- Use `Array` for coefficient-wise math-heavy code.

## Passing Eigen Objects to Functions

Simple and clear:

```cpp
float l2_norm(const Eigen::VectorXf& v) {
    return v.norm();
}
```

For writable output parameters:

```cpp
void scale_in_place(Eigen::VectorXf& v, float s) {
    v *= s;
}
```

This is a good default for a small project using `Eigen::VectorXf` and `Eigen::MatrixXf`.

## Example Matching This Repo

```cpp
#include <Eigen/Dense>

Eigen::VectorXf forward(
    const Eigen::MatrixXf& weights,
    const Eigen::VectorXf& input,
    const Eigen::VectorXf& bias) {
    return weights * input + bias;
}
```

## Common Mistakes

- Mixing up matrix multiplication and element-wise multiplication.
- Forgetting to size a dynamic vector or matrix before assigning coefficients.
- Using `inverse()` when `solve()` is the better operation.
- Passing mismatched shapes, like multiplying a `3 x 2` matrix by a size-3 vector.
- Forgetting that some expressions produce row vectors versus column vectors.

## Handy Example

```cpp
Eigen::Matrix2f matrix;
matrix << 1.0f, 2.0f,
          3.0f, 4.0f;

Eigen::Vector2f vector;
vector << 5.0f, 6.0f;

Eigen::Vector2f result = matrix * vector;
```
