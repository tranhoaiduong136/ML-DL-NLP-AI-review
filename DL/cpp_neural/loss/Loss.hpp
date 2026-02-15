#pragma once

class Loss {
public:
    virtual double forward(double y, double y_pred) = 0;
    virtual double backward(double y, double y_pred) = 0;
};

