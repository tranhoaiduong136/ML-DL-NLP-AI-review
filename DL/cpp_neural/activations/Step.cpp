#include "Step.hpp"

double Step::forward(double x) const {
    return (x >= 0) ? 1.0 : 0.0;
}

double Step::backward(double /*x*/) const {
    return 0.0; // Derivative is zero almost everywhere
}
