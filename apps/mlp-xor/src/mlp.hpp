#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <vector>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include "dense_layer.hpp"

namespace mlp
{


class Mlp
{
public:
    explicit                    Mlp                 (const std::vector<DenseLayerConfig>& configs);
    Eigen::VectorXf             forward             (const Eigen::VectorXf& input) const;

private:
    std::vector<DenseLayer>     m_layers;
};

}
