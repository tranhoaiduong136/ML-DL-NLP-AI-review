#pragma once
#include <vector>
#include <memory>
#include "../core/Neuron.hpp"

class Layer {
private:
    std::vector<Neuron> neurons;
    std::vector<double> outputs;
    int input_size;

public:
    Layer(int num_neurons, int input_size, std::shared_ptr<Activation> activation);

    // Forward pass
    std::vector<double> forward(const std::vector<double>& inputs);

    // Backward pass
    std::vector<double> backward(const std::vector<double>& dL_dout, double learning_rate);
};
