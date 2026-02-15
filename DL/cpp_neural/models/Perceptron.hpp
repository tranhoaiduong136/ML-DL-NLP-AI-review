#pragma once
#include "../core/Neuron.hpp"

class Perceptron {
private:
    Neuron neuron;
    double learning_rate;

public:
    Perceptron(int input_size, double lr);

    double predict(const std::vector<double>& x);

    void train(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        int epochs
    );
};
