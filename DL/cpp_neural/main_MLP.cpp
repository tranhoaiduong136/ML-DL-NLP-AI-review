#include "models/MLP.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    // XOR Problem
    // Inputs: 2
    // Hidden: 3 neurons
    // Output: 1 neuron
    std::vector<int> topology = {2, 3, 1};
    double learning_rate = 0.5; // High LR for XOR
    
    MLP mlp(topology, learning_rate);

    // Training Data (XOR)
    std::vector<std::vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    std::vector<double> y = {0, 1, 1, 0};

    std::cout << "Training MLP on XOR problem..." << std::endl;
    mlp.train(X, y, 10000, learning_rate);

    std::cout << "\nTesting:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> output = mlp.predict(X[i]);
        std::cout << "Input: (" << X[i][0] << ", " << X[i][1] << ") "
                  << "-> Prediction: " << output[0]
                  << " (Expected: " << y[i] << ")" << std::endl;
    }

    return 0;
}
