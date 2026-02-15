#pragma once
#include <vector>
#include <memory>
#include "../layers/Layer.hpp"
#include "../loss/MSE.hpp"
#include "../activations/Sigmoid.hpp"

class MLP {
private:
    std::vector<Layer> layers;
    std::unique_ptr<Loss> loss;

public: 
    MLP(const std::vector<int>& topology, double learning_rate = 0.01);

    std::vector<double> forward(const std::vector<double>& inputs);
    void backward(const std::vector<double>& dL_dout, double lr);
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double lr);
    std::vector<double> predict(const std::vector<double>& inputs);
};
