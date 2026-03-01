#include "activationfunction.h"
#include <cmath>
#include <algorithm>

//标量激活函数
double ActivationFunction::activate(double x, ActivationType type)
{
    switch(type) {
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ActivationType::RELU:
            return std::max(0.0, x);
        case ActivationType::TANH:
            return tanh(x);
        case ActivationType::LINEAR:
            return x;
        case ActivationType::SOFTMAX:
            throw std::logic_error("Softmax不适用于标量！");
        default:
            return x;
    }
}

//标量激活函数导数
double ActivationFunction::derivative(double x, ActivationType type)
{
    switch(type) {
        case ActivationType::SIGMOID:
        {
            double s = ActivationFunction::activate(x, type);
            return s * (1.0 - s);
        }
        case ActivationType::RELU:
            return (x > 0.0) ? 1.0 : 0.0;
        case ActivationType::TANH:
        {
            double s = ActivationFunction::activate(x, type);
            return 1.0 - s * s;
        }
        case ActivationType::LINEAR:
            return 1.0;
        case ActivationType::SOFTMAX:
            return 1.0;
        default:
            return 1.0;
    }
}

//矩阵激活函数
Matrix ActivationFunction::activate(const Matrix &input, ActivationType type)
{
    if(type == ActivationType::SOFTMAX)
        return softmax(input);

    Matrix result(input.getRows(), input.getCols());

    for(size_t i = 0; i < input.getRows();++i) {
        for(size_t j = 0;j < input.getCols();++j) {
            result(i, j) = ActivationFunction::activate(input[i][j], type);
        }
    }
    return result;
}

//矩阵激活函数导数
Matrix ActivationFunction::derivative(const Matrix &input, ActivationType type)
{
    if(type == ActivationType::SOFTMAX)
        return softmaxDerivative(input);

    Matrix result(input.getRows(), input.getCols());

    for(size_t i = 0;i < input.getRows();++i) {
        for(size_t j = 0;j < input.getCols();++j) {
            result(i, j) = ActivationFunction::derivative(input[i][j], type);
        }
    }
    return result;
}

//特殊处理激活函数
Matrix ActivationFunction::softmax(const Matrix &input)
{
    if(!input.isColVector()) {
        throw std::invalid_argument("输入必须是列向量！");
    }

    size_t n = input.getRows();

    Matrix result(n, 1);

    double maxn = input(0, 0);
    for(size_t i = 1;i < n;++i) {
        maxn = std::max(maxn, input(i, 0));
    }

    double sum = 0.0;
    for(size_t i = 0;i < n;++i) {
        result(i, 0) = exp(input(i, 0) - maxn);
        sum += result(i, 0);
    }

    for(size_t i = 0;i < n;++i) {
        result(i, 0) /= sum;
    }
    return result;
}

//特殊处理激活函数导数
Matrix ActivationFunction::softmaxDerivative(const Matrix &input)
{
    Matrix softmaxResult = softmax(input);
    size_t n = input.getRows();
    Matrix result(n, 1);

    for(size_t i = 0; i < n; ++i) {
        result(i, 0) = softmaxResult(i, 0) * (1.0 - softmaxResult(i, 0));
    }
    return result;
}
