#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

using Vector = std::vector<float>;
using Matrix = std::vector<std::vector<float>>;

inline Vector make_vector(std::size_t size, float value = 0.0f) {
    return Vector(size, value);
}

inline Matrix make_matrix(std::size_t rows, std::size_t cols, float value = 0.0f) {
    return Matrix(rows, Vector(cols, value));
}

inline Vector mat_vec_mul(const Matrix& m, const Vector& v) {
    assert(!m.empty());
    assert(m[0].size() == v.size());

    Vector result(m.size(), 0.0f);

    for (std::size_t row = 0; row < m.size(); ++row) {
        assert(m[row].size() == v.size());

        for (std::size_t col = 0; col < v.size(); ++col) {
            result[row] += m[row][col] * v[col];
        }
    }

    return result;
}

inline Vector add(const Vector& a, const Vector& b) {
    assert(a.size() == b.size());

    Vector result(a.size(), 0.0f);

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}
