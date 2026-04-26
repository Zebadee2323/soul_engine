#include <iostream>

#include "math.hpp"

int main() {
    Matrix matrix = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
    };
    Vector vector = {5.0f, 6.0f};

    Vector result = mat_vec_mul(matrix, vector);

    std::cout << "Result: [" << result[0] << ", " << result[1] << "]" << '\n';
    return 0;
}
