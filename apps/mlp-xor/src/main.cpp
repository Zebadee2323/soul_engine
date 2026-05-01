#include <Eigen/Dense>
#include <iostream>
#include "dense_layer.hpp"

bool confirm_phase_2_output(const Eigen::VectorXf& output)
{
    const Eigen::VectorXf expected =
        Eigen::VectorXf::Constant(3, 3.0f);

    const bool matches = output.size() == expected.size()
        && output.isApprox(expected);

    std::cout << "Expected:\n" << expected << '\n';
    std::cout << "Actual:\n" << output << '\n';
    std::cout << (matches ? "Phase 2 check passed.\n"
                          : "Phase 2 check failed.\n");

    return matches;
}

int main()
{
    Eigen::VectorXf input(2);
    input << 1.0f, 2.0f;

    DenseLayer dense_layer(2, 3);
    const Eigen::VectorXf output = dense_layer.forward(input);

    std::cout << "Result:\n" << output << '\n';
    confirm_phase_2_output(output);
    return 0;
}
