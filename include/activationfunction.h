#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include "matrix.h"

//激活函数类型
enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    LINEAR,
    SOFTMAX
};

class ActivationFunction
{
public:
    //标量激活函数
    static double activate(double x, ActivationType type);
    //标量激活函数导数
    static double derivative(double x, ActivationType type);

    //矩阵激活函数
    static Matrix activate(const Matrix& input, ActivationType type);
    //矩阵激活函数导数
    static Matrix derivative(const Matrix& input, ActivationType type);

    //特殊处理激活函数
    static Matrix softmax(const Matrix& input);
    //特殊处理激活函数导数
    static Matrix softmaxDerivative(const Matrix& input);
};

#endif // ACTIVATIONFUNCTION_H
