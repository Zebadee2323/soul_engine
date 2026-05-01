#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Core>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------

namespace mlp
{

float                       random_float            (std::mt19937& rng, std::uniform_real_distribution<float>& dist);
float                       random_float            (std::mt19937& rng, float min, float max);

void                        set_random_vector       (Eigen::VectorXf& vector, std::mt19937& rng, float min, float max);
void                        set_random_matrix       (Eigen::MatrixXf& matrix, std::mt19937& rng, float min, float max);

// --------------------------------------------------------------------------------------------------------------------

enum class                  ActivationType          {None, Relu, Sigmoid, TanH};

inline Eigen::VectorXf      relu                    (const Eigen::VectorXf& z) { return z.cwiseMax(0.0f);                               }
inline Eigen::VectorXf      sigmoid                 (const Eigen::VectorXf& z) { return (1.0f / (1.0f + (-z.array()).exp())).matrix();  }
inline Eigen::VectorXf      tanh                    (const Eigen::VectorXf& z) { return z.array().tanh().matrix();                      }
Eigen::VectorXf             apply_activation        (const Eigen::VectorXf& z, ActivationType activation_type);

inline Eigen::VectorXf      relu_derivative         (const Eigen::VectorXf& a) { return a.cwiseMax(0.0f).cwiseSign(); }
inline Eigen::VectorXf      sigmoid_derivative      (const Eigen::VectorXf& a) { return (a.array() * (1.0f - a.array())).matrix(); }
inline Eigen::VectorXf      tanh_derivative         (const Eigen::VectorXf& a) { return (1.0f - a.array().square()).matrix(); }
Eigen::VectorXf             activation_derivative   (const Eigen::VectorXf& a, ActivationType activation_type);

// --------------------------------------------------------------------------------------------------------------------

float                       mse                     (const Eigen::VectorXf& prediction, const Eigen::VectorXf& target);
Eigen::VectorXf             mse_derivative          (const Eigen::VectorXf& prediction, const Eigen::VectorXf& target);

}
