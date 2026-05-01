#include "mlp.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <cassert>

namespace mlp
{

Mlp::Mlp(const std::vector<DenseLayerConfig>& configs) {
    m_layers.reserve(configs.size());

    for (std::size_t i = 0; i < configs.size(); ++i) {
        assert(i == 0 || configs[i - 1].output_size == configs[i].input_size);
        m_layers.emplace_back(configs[i]);
    }
}

Eigen::VectorXf Mlp::forward (const Eigen::VectorXf& input) const {
    auto result = input;
    for (const DenseLayer& layer : m_layers) {
        result = layer.forward(result);
    }
    return result;
}

}
