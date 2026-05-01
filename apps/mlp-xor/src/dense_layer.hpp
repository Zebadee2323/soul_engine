#pragma once

#include <cstddef>
#include <Eigen/Dense>

class DenseLayer
{
public:
    DenseLayer(std::size_t input_size, std::size_t output_size);

    Eigen::VectorXf forward(const Eigen::VectorXf& input) const;

private:
    std::size_t m_input_size;
    std::size_t m_output_size;
    Eigen::MatrixXf m_weights;
    Eigen::VectorXf m_biases;
};
