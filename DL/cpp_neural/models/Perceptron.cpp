#include "Perceptron.hpp"
#include "../activations/Step.hpp"
#include <iostream>

Perceptron::Perceptron(int input_size, double lr)
    : neuron(input_size, std::make_shared<Step>()),
      learning_rate(lr) {}

double Perceptron::predict(const std::vector<double>& x) {
    return neuron.forward(x);
}

void Perceptron::train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    int epochs
) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int errors = 0;

        for (size_t i = 0; i < X.size(); ++i) {
            double y_pred = predict(X[i]);
            double error = y[i] - y_pred;

            if (error != 0) {
                neuron.update_weights(X[i], error, learning_rate);
                errors++;
            }
        }

        std::cout << "Epoch " << epoch
                  << " | Errors: " << errors << "\n";
        
        if (errors == 0) {
            std::cout << "Converged at epoch " << epoch << "\n";
            break;
        }
    }
}
