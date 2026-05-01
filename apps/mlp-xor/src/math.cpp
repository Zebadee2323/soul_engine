#include "math.hpp"
// --------------------------------------------------------------------------------------------------------------------

float random_float(std::mt19937& rng, float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

void set_random_vector(Eigen::VectorXf& vector, std::mt19937& rng, float min, float max)
{
    for (Eigen::Index i = 0; i < vector.size(); ++i)
    {
        vector(i) = random_float(rng, min, max);
    }
}

void set_random_matrix(Eigen::MatrixXf& matrix, std::mt19937& rng, float min, float max)
{
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        {
            matrix(row, col) = random_float(rng, min, max);
        }
    }
}
