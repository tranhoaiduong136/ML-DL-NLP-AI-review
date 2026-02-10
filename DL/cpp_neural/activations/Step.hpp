// Best practice- replace for

// #ifndef STEP_HPP
// #define STEP_HPP

#pragma once

#include "Activation.hpp"

class Step : public Activation {
public:
    double forward(double x) const override;
    double backward(double x) const override;
};