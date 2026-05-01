#pragma once

#include <random>
#include <Eigen/Dense>

float random_float      (std::mt19937& rng, float min, float max);

void set_random_vector  (Eigen::VectorXf& vector, std::mt19937& rng, float min, float max);
void set_random_matrix  (Eigen::MatrixXf& matrix, std::mt19937& rng, float min, float max);
