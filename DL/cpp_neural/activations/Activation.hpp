#pragma once

class Activation {
public: 
    virtual ~Activation()=default;
    
    virtual double forward(double x)const = 0;
    virtual double backward(double x)const = 0;

};



