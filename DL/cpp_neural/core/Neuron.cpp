#include "Neuron.hpp"
#include <stdexcept>

#include <random>
#include <chrono>

Neuron::Neuron(int input_size, std::shared_ptr<Activation> act)
    : bias(0.0), activation(act) 
{
    // Random initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    weights.reserve(input_size);
    for (int i = 0; i < input_size; ++i) {
        weights.push_back(distribution(generator) * 0.1); // Small random weights
    }
}

double Neuron::forward(const std::vector<double>& inputs) {
    this->last_inputs = inputs;
    if (inputs.size() != weights.size())
        throw std::runtime_error("Input size mismatch");

    double sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += weights[i] * inputs[i];

    this->z = sum; // Store for backward pass
    return activation->forward(sum);
}


double Neuron::backward(double dL_dout, double learning_rate) { 
    double d_out_dz = activation->backward(z);
    double dL_dz = dL_dout * d_out_dz;
    
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] -= learning_rate * dL_dz * last_inputs[i];
    bias -= learning_rate * dL_dz;
    return dL_dz;
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

