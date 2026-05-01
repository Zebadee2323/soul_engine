#pragma once

#include <cstddef>
#include <Eigen/Dense>
#include <random>

class DenseLayer
{
public:
    DenseLayer          (std::size_t input_size, std::size_t output_size);

    Eigen::VectorXf     forward(const Eigen::VectorXf& input) const;
    void                randomize_weights(std::mt19937& rng, float min, float max);
    void                randomize_biases(std::mt19937& rng, float min, float max);

private:
    std::size_t         m_input_size;
    std::size_t         m_output_size;
    Eigen::MatrixXf     m_weights;
    Eigen::VectorXf     m_biases;
};
