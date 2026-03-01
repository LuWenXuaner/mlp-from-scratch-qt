#include "graphnode.h"

//权重初始化
void GraphNodeImpl::initializeWeights() {
    if(nodetype == NodeType::INPUT) return;

    if (inputSize <= 10 && outputSize <= 10) {
        double range = std::sqrt(2.0 / (inputSize + outputSize));
        weightMatrix.randomize(-range, range);
    } else {
        double range = std::sqrt(6.0 / (inputSize + outputSize));
        weightMatrix.randomize(-range, range);
    }

    biasVector.randomize(-0.1, 0.1);
}

//构造函数
GraphNodeImpl::GraphNodeImpl(int id, NodeType type,
                        size_t inputSize, size_t outputSize):
    Id(id), nodetype(type), inputSize(inputSize), outputSize(outputSize)
{
    if(nodetype != NodeType::INPUT) {
        weightMatrix = Matrix::zeros(outputSize, inputSize);
        biasVector = Matrix::zeros(outputSize, 1);
        initializeWeights();
    }
}

//构造函数
GraphNodeImpl::GraphNodeImpl(int id, NodeType type, size_t inputSize, size_t outputSize, int layerId)
    : Id(id), nodetype(type), inputSize(inputSize), outputSize(outputSize), layerId(layerId)
{
    if (nodetype != NodeType::INPUT) {
        weightMatrix = Matrix::zeros(outputSize, inputSize);
        biasVector = Matrix::zeros(outputSize, 1);
        initializeWeights();
    }
}

//获取所属层编号
int GraphNodeImpl::getLayerId() const
{
    return layerId;
}

//设置所属层编号
void GraphNodeImpl::setLayerId(int layerId)
{
    this->layerId = layerId;
}

//添加节点为当前节点输入
void GraphNodeImpl::addInputNode(int nodeId)
{
    if(!inputNodes.contains(nodeId)) {
        inputNodes.append(nodeId);
    }
}

//添加节点为当前节点输出
void GraphNodeImpl::addOutputNode(int nodeId)
{
    if(!outputNodes.contains(nodeId)) {
        outputNodes.append(nodeId);
    }
}

//移除当前节点的某个输入节点
void GraphNodeImpl::removeInputNode(int nodeId)
{
    inputNodes.removeAll(nodeId);
    if(nodetype != NodeType::INPUT) {
        initializeWeights();
    }
}

//移除当前节点的某个输出节点
void GraphNodeImpl::removeOutputNode(int nodeId)
{
    outputNodes.removeAll(nodeId);
}

//设置权重矩阵
void GraphNodeImpl::setWeightMatrix(const Matrix &weights)
{
    if(weights.getRows() != outputSize || weights.getCols() != inputSize) {
        throw std::invalid_argument("维度不匹配！");
    }
    weightMatrix = weights;
}

//设置当前节点偏置向量
void GraphNodeImpl::setBiasVector(const Matrix &bias)
{
    if(bias.getRows() != outputSize || bias.getCols() != 1) {
        throw std::invalid_argument("维度不匹配！");
    }
    biasVector = bias;
}

//设置当前节点输入维度
void GraphNodeImpl::setInputSize(size_t size)
{
    inputSize = size;
    if(nodetype != NodeType::INPUT) {
        weightMatrix = Matrix::zeros(outputSize, inputSize);
        initializeWeights();
    }
}

//设置当前节点输出维度
void GraphNodeImpl::setOutputSize(size_t size)
{
    outputSize = size;
    if(nodetype != NodeType::INPUT) {
        weightMatrix = Matrix::zeros(outputSize, inputSize);
        biasVector = Matrix::zeros(outputSize, 1);
        initializeWeights();
    }
}
