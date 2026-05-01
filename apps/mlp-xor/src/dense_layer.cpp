#include <cassert>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include "dense_layer.hpp"
// --------------------------------------------------------------------------------------------------------------------

namespace mlp
{

DenseLayer::DenseLayer(const DenseLayerConfig& config) :
    DenseLayer              (config.input_size, config.output_size, config.activation_type)
{
}

DenseLayer::DenseLayer(Eigen::Index input_size, Eigen::Index output_size, ActivationType activation_type) :
    m_input_size            (input_size),
    m_output_size           (output_size),
    m_activation_type       (activation_type),
    m_weights               (output_size, input_size),
    m_biases                (output_size),
    m_last_input            (mlp::VectorXf::Zero(input_size)),
    m_last_z                (mlp::VectorXf::Zero(output_size)),
    m_last_activation       (mlp::VectorXf::Zero(output_size)),
    m_weight_gradients      (mlp::MatrixXf::Zero(output_size, input_size)),
    m_bias_gradients        (mlp::VectorXf::Zero(output_size))
{
    auto rng = std::mt19937();
    randomize_weights(rng, -1.0f, 1.0f);
    randomize_biases(rng, -1.0f, 1.0f);
}

mlp::VectorXf DenseLayer::forward(const mlp::VectorXf& x) {
    assert(x.size() == m_input_size);
    m_last_input = x;
    m_last_z = m_weights * x + m_biases;
    m_last_activation = apply_activation(m_last_z, m_activation_type);
    return m_last_activation;
}

mlp::VectorXf DenseLayer::backward(const mlp::VectorXf& d_y) {
    assert(d_y.size() == m_output_size);
    mlp::VectorXf d_a = activation_derivative(m_last_activation, m_activation_type);
    mlp::VectorXf d_z = d_y.cwiseProduct(d_a);
    m_weight_gradients = d_z * m_last_input.transpose();
    m_bias_gradients = d_z;
    return m_weights.transpose() * d_z;
}

void DenseLayer::apply_gradients(float learning_rate) {
    m_weights -= learning_rate * m_weight_gradients;
    m_biases -= learning_rate * m_bias_gradients;
}

void DenseLayer::randomize_weights(std::mt19937& rng, float min, float max) {
    set_random_matrix(m_weights, rng, min, max);
}

void DenseLayer::randomize_biases(std::mt19937& rng, float min, float max) {
    set_random_vector(m_biases, rng, min, max);
}

}
