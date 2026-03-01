#ifndef MODELTEST_H
#define MODELTEST_H

#include <vector>
#include "mlpnetwork.h"
#include "mlpgraph.h"
#include "graphnode.h"
#include "matrix.h"
#include "activationfunction.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include<memory>

class ModelTest
{
private:
    //训练轮次
    int epochs = 10000;
    //学习率
    double learningRate = 0.1;
    //输入层神经元数
    int NN_input = 1;
    //隐藏层层数以及各层神经元数
    std::vector<int> NN_hidden = {1};
    //输出层神经元数
    int NN_output = 1;
    //激活函数类型
    ActivationType type = ActivationType::SIGMOID;
    //输入数据
    std::vector<Matrix> trainingInputs;
    //输出数据
    std::vector<Matrix> trainingOutputs;
public:
    //无参构造函数
    ModelTest();
    //带参数构造函数
    ModelTest(int e = 10000, double lr = 0.1, int in = 1, std::vector<int> hid = {1}, int out = 1, ActivationType ty = ActivationType::SIGMOID);
    //简单数据集
    void SimpleDataTest();
    //鸢尾花数据集
    void TestCsv();
    //简单神经网络计算值对比
    void ComparisonTest();
    //加载csv文件
    void LoadSimpleData();
};

#endif // MODELTEST_H
