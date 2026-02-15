# include "MSE.hpp"


double MSE::forward(double y, double y_pred) {
    return 0.5 * (y - y_pred) * (y - y_pred);
}

double MSE::backward(double y, double y_pred) {
    return y_pred - y;
}