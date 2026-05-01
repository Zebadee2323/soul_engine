#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <vector>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include <mlp/types.hpp>
#include <mlp/dense_layer.hpp>

namespace mlp
{

struct Sample
{
    VectorXf                    x;
    VectorXf                    y_target;
};


class Mlp
{

public:
    explicit                    Mlp                 (const std::vector<DenseLayerConfig>& configs);
// --------------------------------------------------------------------------------------------------------------------
    mlp::VectorXf               forward             (const mlp::VectorXf& x);
    void                        backward            (const mlp::VectorXf& d_loss);
    void                        apply_gradients     (float learning_rate);
// --------------------------------------------------------------------------------------------------------------------
    static void                 train               (Mlp& mlp, const std::vector<Sample>& dataset, size_t epochs,
                                                        float learning_rate);
private:
    std::vector<DenseLayer>     m_layers;

};

}
