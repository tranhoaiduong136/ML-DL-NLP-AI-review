#include "MLP.hpp"
#include <iostream>

MLP::MLP(const std::vector<int>& topology, double learning_rate) {
    if (topology.size() < 2)
        throw std::runtime_error("Topology must define at least input and output layers");

    auto activation = std::make_shared<Sigmoid>();
    loss = std::make_unique<MSE>();

    for (size_t i = 1; i < topology.size(); ++i) {
        layers.emplace_back(topology[i], topology[i-1], activation);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    for (auto& layer : layers)
        outputs = layer.forward(outputs);
    return outputs;
}

void MLP::backward(const std::vector<double>& dL_dout, double lr) {
    std::vector<double> dL_din = dL_dout;
    for (int i = layers.size() - 1; i >= 0; --i)
        dL_din = layers[i].backward(dL_din, lr);
}

std::vector<double> MLP::predict(const std::vector<double>& inputs) {
    return forward(inputs);
}

void MLP::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> y_pred = forward(X[i]);
            total_loss += loss->forward(y[i], y_pred[0]);
            
            // For MSE derivative wrt y_pred is (y_pred - y).
            // loss->backward returns that.
            double grad = loss->backward(y[i], y_pred[0]);
            
            backward({grad}, lr);
        }
        if (epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss / X.size() << std::endl;
    }
}