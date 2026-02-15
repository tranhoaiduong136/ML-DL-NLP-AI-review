

#pragma once
#include "Loss.hpp"

class MSE : public Loss {
public:
    double forward(double y, double y_pred) override;
    double backward(double y, double y_pred) override;
};
