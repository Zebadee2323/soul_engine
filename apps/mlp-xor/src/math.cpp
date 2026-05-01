#include "math.hpp"
#include <Eigen/Core>
#include <cassert>
// --------------------------------------------------------------------------------------------------------------------

namespace mlp
{

float random_float(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    return dist(rng);
}

float random_float(std::mt19937& rng, float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return random_float(rng, dist);
}

void set_random_vector(Eigen::VectorXf& vector, std::mt19937& rng, float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        vector(i) = random_float(rng, dist);
    }
}

void set_random_matrix(Eigen::MatrixXf& matrix, std::mt19937& rng, float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            matrix(row, col) = random_float(rng, dist);
        }
    }
}

Eigen::VectorXf apply_activation(const Eigen::VectorXf& z, ActivationType activation_type) {
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

    return z;
}

float mse (const Eigen::VectorXf& prediction, const Eigen::VectorXf& target) {
    assert(prediction.size() == target.size());
    return (prediction - target).cwisePow(2.0f).mean();
}

Eigen::VectorXf mse_derivative (const Eigen::VectorXf& prediction, const Eigen::VectorXf& target) {
    assert(prediction.size() == target.size());
    return 2.0f * (prediction - target) / static_cast<float>(prediction.size());
}

Eigen::VectorXf activation_derivative(const Eigen::VectorXf& a, ActivationType activation_type) {
    switch (activation_type) {
        case ActivationType::None:
            return Eigen::VectorXf::Ones(a.size());
        case ActivationType::Relu:
            return relu_derivative(a);
        case ActivationType::Sigmoid:
            return sigmoid_derivative(a);
        case ActivationType::TanH:
            return tanh_derivative(a);
    }

    return Eigen::VectorXf::Ones(a.size());
}

}
