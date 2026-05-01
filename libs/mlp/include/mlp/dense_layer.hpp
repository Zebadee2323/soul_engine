#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <cstddef>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include <mlp/types.hpp>
#include <mlp/math.hpp>

namespace mlp
{

struct DenseLayerConfig
{
    std::size_t         input_size;
    std::size_t         output_size;
    ActivationType      activation_type;
};

class DenseLayer
{

public:
    explicit            DenseLayer              (const DenseLayerConfig& config);
                        DenseLayer              (Eigen::Index input_size, Eigen::Index output_size,
                                                        ActivationType activation_type);
// --------------------------------------------------------------------------------------------------------------------
    mlp::VectorXf       forward                 (const mlp::VectorXf& x);
    mlp::VectorXf       backward                (const mlp::VectorXf& d_y);
    void                apply_gradients         (float learning_rate);
// --------------------------------------------------------------------------------------------------------------------
    void                randomize_weights       (std::mt19937& rng, float min, float max);
    void                randomize_biases        (std::mt19937& rng, float min, float max);
// --------------------------------------------------------------------------------------------------------------------
    Eigen::Index        input_size              () const { return m_input_size;         }
    Eigen::Index        output_size             () const { return m_output_size;        }
    ActivationType      activation_type         () const { return m_activation_type;    }

private:
    Eigen::Index        m_input_size;
    Eigen::Index        m_output_size;
    ActivationType      m_activation_type;
// --------------------------------------------------------------------------------------------------------------------
    mlp::MatrixXf       m_weights;
    mlp::VectorXf       m_biases;
// --------------------------------------------------------------------------------------------------------------------
    mlp::VectorXf       m_last_input;
    mlp::VectorXf       m_last_z;
    mlp::VectorXf       m_last_activation;
// --------------------------------------------------------------------------------------------------------------------
    mlp::MatrixXf       m_weight_gradients;
    mlp::VectorXf       m_bias_gradients;

};

}
