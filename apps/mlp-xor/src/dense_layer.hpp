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
                            DenseLayer              (std::size_t input_size, std::size_t output_size,
                                                        ActivationType activation_type);

    Eigen::VectorXf         forward                 (const Eigen::VectorXf& input) const;
    void                    randomize_weights       (std::mt19937& rng, float min, float max);
    void                    randomize_biases        (std::mt19937& rng, float min, float max);

    std::size_t             input_size              () const { return m_input_size;     }
    std::size_t             output_size             () const { return m_output_size;    }
    ActivationType          activation_type         () const { return m_activation;     }

private:
    std::size_t             m_input_size;
    std::size_t             m_output_size;
    Eigen::MatrixXf         m_weights;
    Eigen::VectorXf         m_biases;
    ActivationType          m_activation;
};

}
