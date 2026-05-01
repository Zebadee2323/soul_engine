#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>

namespace mlp
{

class DenseLayer
{
public:
    DenseLayer                              (std::size_t input_size, std::size_t output_size);

    Eigen::VectorXf     forward             (const Eigen::VectorXf& input) const;
    void                randomize_weights   (std::mt19937& rng, float min, float max);
    void                randomize_biases    (std::mt19937& rng, float min, float max);

private:
    std::size_t         m_input_size;
    std::size_t         m_output_size;
    Eigen::MatrixXf     m_weights;
    Eigen::VectorXf     m_biases;
};

}
