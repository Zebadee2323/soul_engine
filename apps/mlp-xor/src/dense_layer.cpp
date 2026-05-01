#include <Eigen/Core>
#include <cassert>
// --------------------------------------------------------------------------------------------------------------------
#include "dense_layer.hpp"
// --------------------------------------------------------------------------------------------------------------------

namespace mlp
{

DenseLayer::DenseLayer(const DenseLayerConfig& config) :
    DenseLayer      (config.input_size, config.output_size, config.activation_type)
{
}

DenseLayer::DenseLayer(std::size_t input_size, std::size_t output_size, ActivationType activation_type) :
    m_input_size    (input_size),
    m_output_size   (output_size),
    m_weights       (Eigen::MatrixXf::Ones(output_size, input_size)),
    m_biases        (Eigen::VectorXf::Zero(output_size)),
    m_activation    (activation_type)
{
    assert(input_size > 0);
    assert(output_size > 0);
}

Eigen::VectorXf DenseLayer::forward(const Eigen::VectorXf& input) const {
    assert(input.size() == static_cast<Eigen::Index>(m_input_size));
    return apply_activation(m_weights * input + m_biases, m_activation);
}

void DenseLayer::randomize_weights(std::mt19937& rng, float min, float max) {
    set_random_matrix(m_weights, rng, min, max);
}

void DenseLayer::randomize_biases(std::mt19937& rng, float min, float max) {
    set_random_vector(m_biases, rng, min, max);
}

}
