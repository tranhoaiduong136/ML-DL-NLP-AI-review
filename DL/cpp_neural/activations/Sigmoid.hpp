#pragma once
#include "Activation.hpp"
class Sigmoid : public Activation {
public:
    double forward(double x) const override;
    double backward(double x) const override;
};