#include "models/Perceptron.hpp"
#include <iostream>
#include <vector>

int main() {
    // 2 inputs for AND gate, learning rate 0.1
    Perceptron p(3, 0.1);

    // AND gate training data
    std::vector<std::vector<double>> X = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 1}
    };
    std::vector<double> y = {0, 0, 0, 1};

    std::cout << "Training Perceptron on AND gate..." << std::endl;
    p.train(X, y, 10); // Train for 10 epochs

    // Test
    std::cout << "\nTesting:" << std::endl;
    for (const auto& input : X) {
        double output = p.predict(input);
        std::cout << "Input: (" << input[0] << ", " << input[1] << ", " << input[2] << ") "
                  << "-> Prediction: " << output 
                  << " (Expected: " << ((input[0] && input[1]) ? 1 : 0) << ")" << std::endl;
    }
    std::vector<double> test_input = {{1,1,0}};
    std::cout << p.predict(test_input) << std::endl;
    return 0;
}
