#pragma once
#include <vector>
#include <memory>
#include "../activations/Activation.hpp"

class Neuron {
private:
    std::vector<double> weights;
    double bias;
    std::vector<double> last_inputs; // To store inputs for backward pass

    double z; // wx + b
    double output; // activation(z)

    std::shared_ptr<Activation> activation;

public:
    Neuron(int input_size, std::shared_ptr<Activation> act);

    double forward(const std::vector<double>& inputs);
    double backward(double dL_dout, double learning_rate);
    void update_weights(
        const std::vector<double>& inputs,
        double error,
        double learning_rate
    );

    const std::vector<double>& get_weights() const;
};
