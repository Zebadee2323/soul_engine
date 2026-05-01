#include "mlp.hpp"
#include "dense_layer.hpp"
#include "math.hpp"
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cassert>
#include <iostream>

namespace mlp
{

Mlp::Mlp(const std::vector<DenseLayerConfig>& configs) {
    m_layers.reserve(configs.size());

    for (std::size_t i = 0; i < configs.size(); ++i) {
        assert(i == 0 || configs[i - 1].output_size == configs[i].input_size);
        m_layers.emplace_back(configs[i]);
    }
}

mlp::VectorXf Mlp::forward (const mlp::VectorXf& x) {
    auto a = x;
    for (DenseLayer& layer : m_layers) {
        a = layer.forward(a);
    }
    return a;
}

void Mlp::backward(const mlp::VectorXf& d_loss) {
    auto gradient = d_loss;
    for (auto layer = m_layers.rbegin(); layer != m_layers.rend(); ++layer) {
        gradient = layer->backward(gradient);
    }
}

void Mlp::apply_gradients(float learning_rate) {
    for (auto& layer : m_layers) {
        layer.apply_gradients(learning_rate);
    }
}

void Mlp::train(Mlp &mlp, const std::vector<Sample> &dataset, size_t epochs, float learning_rate) {
    std::cout << "Training for " << epochs << " epochs, learning rate: " << learning_rate << '\n';

    const size_t print_interval = std::max<size_t>(1, epochs / 10);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (const Sample& sample : dataset) {
            auto y = mlp.forward(sample.x);
            auto loss = mse(y, sample.y_target);
            auto d_loss = mse_derivative(y, sample.y_target);
            mlp.backward(d_loss);
            mlp.apply_gradients(learning_rate);
            total_loss += loss;
        }

        const bool should_print = epoch % print_interval == 0 || epoch == epochs - 1;
        if (should_print) {
            const float average_loss = total_loss / static_cast<float>(dataset.size());
            std::cout << "Epoch: " << epoch << ", loss: " << average_loss << '\n';
        }
    }
}

}
