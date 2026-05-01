#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Core>
#include <vector>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include "dense_layer.hpp"

namespace mlp
{

struct Sample
{
    Eigen::VectorXf             input;
    Eigen::VectorXf             target;
};


class Mlp
{
public:
    explicit                    Mlp                 (const std::vector<DenseLayerConfig>& configs);
    Eigen::VectorXf             forward             (const Eigen::VectorXf& input) const;

private:
    std::vector<DenseLayer>     m_layers;
};

}
