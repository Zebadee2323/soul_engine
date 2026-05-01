#include <vector>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include "mlp.hpp"

int main() {
    const std::vector<mlp::Sample> dataset = {
        {Eigen::Vector2f(0.0f, 0.0f), Eigen::VectorXf::Constant(1, 0.0f)},
        {Eigen::Vector2f(0.0f, 1.0f), Eigen::VectorXf::Constant(1, 1.0f)},
        {Eigen::Vector2f(1.0f, 0.0f), Eigen::VectorXf::Constant(1, 1.0f)},
        {Eigen::Vector2f(1.0f, 1.0f), Eigen::VectorXf::Constant(1, 0.0f)},
    };

    mlp::Mlp mlp({
        {2, 3, mlp::ActivationType::Sigmoid},
        {3, 1, mlp::ActivationType::Sigmoid},
    });

    mlp::Mlp::train(mlp, dataset, 50000, 0.1f);

    return 0;
}
