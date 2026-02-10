#include "Neuron.hpp"
#include <stdexcept>

Neuron::Neuron(int input_size, std::shared_ptr<Activation> act)
    : weights(input_size, 0.0), bias(0.0), activation(act) {}

double Neuron::forward(const std::vector<double>& inputs) const {
    if (inputs.size() != weights.size())
        throw std::runtime_error("Input size mismatch");

    double sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += weights[i] * inputs[i];

    return activation->forward(sum);
}

/* update_weights: 
    inputs: vector of inputs to the neuron
    error: error to be propagated back to the neuron
    learning_rate: learning rate for the neuron
*/

void Neuron::update_weights(
    const std::vector<double>& inputs,
    double error,
    double learning_rate
) {
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] += learning_rate * error * inputs[i];

    bias += learning_rate * error;
}

const std::vector<double>& Neuron::get_weights() const {
    return weights;
}

