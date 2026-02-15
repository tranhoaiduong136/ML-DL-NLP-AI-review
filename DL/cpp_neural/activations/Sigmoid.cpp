#include "Sigmoid.hpp"
#include <cmath>


double Sigmoid::forward(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::backward(double x) const {
    double s = forward(x);
    return s * (1 - s);
}