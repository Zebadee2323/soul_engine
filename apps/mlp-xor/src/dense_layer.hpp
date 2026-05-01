#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include "math.hpp"

namespace mlp
{

struct DenseLayerConfig
{
    std::size_t             input_size;
    std::size_t             output_size;
    ActivationType          activation_type;
};

class DenseLayer
{
public:
    explicit                DenseLayer              (const DenseLayerConfig& config);
                            DenseLayer              (Eigen::Index input_size, Eigen::Index output_size,
                                                        ActivationType activation_type);

    Eigen::VectorXf         forward                 (const Eigen::VectorXf& input);
    Eigen::VectorXf         backward                (const Eigen::VectorXf& output_gradient);

    void                    randomize_weights       (std::mt19937& rng, float min, float max);
    void                    randomize_biases        (std::mt19937& rng, float min, float max);

    Eigen::Index            input_size              () const { return m_input_size;     }
    Eigen::Index            output_size             () const { return m_output_size;    }
    ActivationType          activation_type         () const { return m_activation_type;     }

private:
    Eigen::Index            m_input_size;
    Eigen::Index            m_output_size;
    ActivationType          m_activation_type;

    Eigen::MatrixXf         m_weights;
    Eigen::VectorXf         m_biases;

    Eigen::VectorXf         m_last_input;
    Eigen::VectorXf         m_last_z;
    Eigen::VectorXf         m_last_activation;

    Eigen::MatrixXf         m_weight_gradients;
    Eigen::VectorXf         m_bias_gradients;
};

}
