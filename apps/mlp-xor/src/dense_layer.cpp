#include <cassert>
#include "dense_layer.hpp"

DenseLayer::DenseLayer(std::size_t input_size, std::size_t output_size) :
    m_input_size(input_size),
    m_output_size(output_size),
    m_weights(Eigen::MatrixXf::Ones(output_size, input_size)),
    m_biases(Eigen::VectorXf::Zero(output_size))
{
    assert(input_size > 0);
    assert(output_size > 0);
}

Eigen::VectorXf DenseLayer::forward(const Eigen::VectorXf& input) const
{
    assert(input.size() == static_cast<Eigen::Index>(m_input_size));
    return m_weights * input + m_biases;
}
