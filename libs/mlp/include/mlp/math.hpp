#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------
#include <mlp/types.hpp>

namespace mlp
{

float                       random_float            (std::mt19937& rng, std::uniform_real_distribution<float>& dist);
float                       random_float            (std::mt19937& rng, float min, float max);

void                        set_random_vector       (mlp::VectorXf& vector, std::mt19937& rng, float min, float max);
void                        set_random_matrix       (mlp::MatrixXf& matrix, std::mt19937& rng, float min, float max);

// --------------------------------------------------------------------------------------------------------------------
//
enum class                  ActivationType          {None, Relu, Sigmoid, TanH};

inline mlp::VectorXf        relu                    (const mlp::VectorXf& z) { return z.cwiseMax(0.0f);                               }
inline mlp::VectorXf        sigmoid                 (const mlp::VectorXf& z) { return (1.0f / (1.0f + (-z.array()).exp())).matrix();  }
inline mlp::VectorXf        tanh                    (const mlp::VectorXf& z) { return z.array().tanh().matrix();                      }
mlp::VectorXf               apply_activation        (const mlp::VectorXf& z, ActivationType activation_type);

inline mlp::VectorXf        relu_derivative         (const mlp::VectorXf& a) { return a.cwiseMax(0.0f).cwiseSign(); }
inline mlp::VectorXf        sigmoid_derivative      (const mlp::VectorXf& a) { return (a.array() * (1.0f - a.array())).matrix(); }
inline mlp::VectorXf        tanh_derivative         (const mlp::VectorXf& a) { return (1.0f - a.array().square()).matrix(); }
mlp::VectorXf               activation_derivative   (const mlp::VectorXf& a, ActivationType activation_type);
//
// --------------------------------------------------------------------------------------------------------------------
//
float                       mse                     (const mlp::VectorXf& prediction, const mlp::VectorXf& target);
mlp::VectorXf               mse_derivative          (const mlp::VectorXf& prediction, const mlp::VectorXf& target);

}
