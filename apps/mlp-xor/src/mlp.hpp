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
    Eigen::VectorXf             x;
    Eigen::VectorXf             y_target;
};


class Mlp
{
public:
    explicit                    Mlp                 (const std::vector<DenseLayerConfig>& configs);
    Eigen::VectorXf             forward             (const Eigen::VectorXf& x);
    void                        backward            (const Eigen::VectorXf& d_loss);
    void                        apply_gradients     (float learning_rate);

    static void                 train               (Mlp& mlp, const std::vector<Sample>& dataset, size_t epochs,
                                                        float learning_rate);

private:
    std::vector<DenseLayer>     m_layers;
};

}
