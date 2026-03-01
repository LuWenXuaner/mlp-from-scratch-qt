#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "mlpgraph.h"
#include "activationfunction.h"
#include <QMap>
#include<memory>

class MLPNetwork
{
public:
    //析构函数
    virtual ~MLPNetwork() = default;
    //建立神经网络拓扑结构
    virtual void setGraph(std::shared_ptr<MLPGraph> graph) = 0;
    //设置训练数据
    virtual void setTrainingData(const std::vector<Matrix>& inputs,
                                 const std::vector<Matrix>& outputs) = 0;
    //训练过程设置
    virtual void train(int epochs, double learningRate) = 0;
    //使用训练好的模型进行预测
    virtual Matrix predict(const Matrix& input) = 0;
    //获取训练损失值
    virtual double getLoss() const = 0;
    //损失计算
    virtual double calculateLoss(const Matrix& predicted, const Matrix& target) = 0;
};

class MLPNetworkImpl:public MLPNetwork
{
private:
    //拓扑结构
    std::shared_ptr<MLPGraph> graph;
    //输入数据
    std::vector<Matrix> trainingInputs;
    //输出数据
    std::vector<Matrix> trainingOutputs;
    //节点值
    QMap<int, Matrix> nodeValues;
    //节点梯度
    QMap<int, Matrix> nodeGradients;
    //保存每个节点的加权输入z值
    QMap<int, Matrix> lastZ;
    //当前训练轮数
    int currentEpoch;
    //当前训练损失
    double currentLoss;
    //激活函数类型
    ActivationType activationType;
public:
    //前向传播
    void forwardPass(const Matrix& input);
    //反向传播
    void backwardPass(const Matrix& target);
    //权重更新
    void updateWeights(double learningRate);
    //输入节点值拼接
    Matrix concatenateInputs(const QList<int>& inputNodeIds) const;
    //损失计算
    double calculateLoss(const Matrix& predicted, const Matrix& target) override;

    //建立神经网络拓扑结构
    void setGraph(std::shared_ptr<MLPGraph> g) override;
    //设置训练数据
    void setTrainingData(const std::vector<Matrix>& inputs,
                        const std::vector<Matrix>& outputs) override;
    //训练
    void train(int epochs, double learningRate) override;
    //使用训练好的模型进行预测
    Matrix predict(const Matrix& input) override;
    //获取训练损失值
    double getLoss() const override{return currentLoss;}

    //获取网络中各层节点输出值
    QMap<int, Matrix> getNodeValues() const {return nodeValues;}
    //获取当前训练轮数
    //int getCurrentEpoch() const {return currentEpoch;}
    //获得加权输入值
    Matrix getLastZ(int nodeId) const {
        return lastZ.value(nodeId);
    }
    //获得前向传播输出值
    Matrix getNodeValue(int nodeId) const {
        return nodeValues.value(nodeId);
    }
    //获得反向传播梯度值
    Matrix getGradient(int nodeId) const {
        return nodeGradients.value(nodeId);
    }

    //设置激活函数类型
    void setActivationType(ActivationType type) {activationType = type;}
};

#endif // MLPNETWORK_H
