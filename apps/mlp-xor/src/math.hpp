#pragma once
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Core>
#include <random>
// --------------------------------------------------------------------------------------------------------------------
#include <Eigen/Dense>
// --------------------------------------------------------------------------------------------------------------------

namespace mlp
{

float                       random_float        (std::mt19937& rng, std::uniform_real_distribution<float>& dist);
float                       random_float        (std::mt19937& rng, float min, float max);

void                        set_random_vector   (Eigen::VectorXf& vector, std::mt19937& rng, float min, float max);
void                        set_random_matrix   (Eigen::MatrixXf& matrix, std::mt19937& rng, float min, float max);

enum class                  ActivationType      { None, Relu, Sigmoid, TanH };

inline Eigen::VectorXf      relu                (const Eigen::VectorXf& x) { return x.cwiseMax(0.0f);                               }
inline Eigen::VectorXf      sigmoid             (const Eigen::VectorXf& x) { return (1.0f / (1.0f + (-x.array()).exp())).matrix();  }
inline Eigen::VectorXf      tanh                (const Eigen::VectorXf& x) { return x.array().tanh().matrix();                      }

inline Eigen::VectorXf      relu_derivative     (const Eigen::VectorXf& x) { return x.cwiseMax(0.0f).cwiseSign(); }
inline Eigen::VectorXf      sigmoid_derivative  (const Eigen::VectorXf& x) { return (x.array() * (1.0f - x.array())).matrix(); }
inline Eigen::VectorXf      tanh_derivative     (const Eigen::VectorXf& x) { return (1.0f - x.array().square()).matrix(); }

Eigen::VectorXf             apply_activation    (const Eigen::VectorXf& x, ActivationType activation_type);

}
