#include "Layer.hpp"
#include <stdexcept>

Layer::Layer(int num_neurons, int input_size, std::shared_ptr<Activation> activation) 
    : input_size(input_size)
{
    neurons.reserve(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(input_size, activation);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    outputs.clear();
    for (auto& n : neurons)
        outputs.push_back(n.forward(inputs));
    return outputs;
}

std::vector<double> Layer::backward(const std::vector<double>& dL_dout, double learning_rate) {
    std::vector<double> dL_din(input_size, 0.0);

    for (size_t i = 0; i < neurons.size(); ++i) {
        // Copy weights before update for correct backpropagation
        std::vector<double> w = neurons[i].get_weights();
        
        double dL_dz = neurons[i].backward(dL_dout[i], learning_rate);

        for (size_t j = 0; j < input_size; ++j)
            dL_din[j] += dL_dz * w[j];
    }
    return dL_din;
}