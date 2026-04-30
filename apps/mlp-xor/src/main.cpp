#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2f matrix;
    matrix << 1.0f, 2.0f,
              3.0f, 4.0f;

    Eigen::Vector2f vector;
    vector << 5.0f, 6.0f;

    const Eigen::Vector2f result = matrix * vector;

    std::cout << "Result: [" << result[0] << ", " << result[1] << "]" << '\n';
    return 0;
}
