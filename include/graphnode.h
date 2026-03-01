#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include "matrix.h"
#include <QString>
#include <QPointF>
#include <QList>

//节点类型
enum class NodeType {
    INPUT,
    HIDDEN,
    OUTPUT
};

class GraphNode
{
public:
    //析构函数
    virtual ~GraphNode() = default;
    //获取节点编号
    virtual int getId() const = 0;
    //获取节点类型
    virtual NodeType getNodeType() const = 0;
    //获取节点位置
    //virtual QPointF getPosition() const = 0;
    //设置节点位置
    //virtual void setPosition(const QPointF& pos) = 0;
    //获取当前节点输入节点列表
    virtual QList<int> getInputNodes() const = 0;
    //获取当前节点输出节点列表
    virtual QList<int> getOutputNodes() const = 0;
    //添加节点为当前节点输入
    virtual void addInputNode(int nodeId) = 0;
    //添加节点为当前节点输出
    virtual void addOutputNode(int nodeId) = 0;
    //移除当前节点的某个输入节点
    virtual void removeInputNode(int nodeId) = 0;
    //移除当前节点的某个输出节点
    virtual void removeOutputNode(int nodeId) = 0;

    //获取权重矩阵
    virtual Matrix getWeightMatrix() const = 0;
    //设置权重矩阵
    virtual void setWeightMatrix(const Matrix& weights) = 0;
    //获取当前节点偏置向量
    virtual Matrix getBiasVector() const = 0;
    //设置当前节点偏置向量
    virtual void setBiasVector(const Matrix& bias) = 0;

    //获取当前节点输入维度
    virtual size_t getInputSize() const = 0;
    //获取当前节点输出维度
    virtual size_t getOutputSize() const = 0;
    //设置当前节点输入维度
    virtual void setInputSize(size_t size) = 0;
    //设置当前节点输出维度
    virtual void setOutputSize(size_t size) = 0;

    //初始化权重
    virtual void initializeWeights() = 0;
    //设置节点ID
    virtual void setId(const int& id) = 0;

    // 获取当前节点所属层编号
    virtual int getLayerId() const = 0;
    // 设置当前节点所属层编号
    virtual void setLayerId(int layerId) = 0;
};

class GraphNodeImpl: public GraphNode
{
private:
    //节点编号
    int Id;
    //节点类型
    NodeType nodetype;
    //输入维度
    size_t inputSize;
    //输出维度
    size_t outputSize;
    //层编号
    int layerId;
    //当前节点输入节点列表
    QList<int> inputNodes;
    //当前节点输出节点列表
    QList<int> outputNodes;

    //权重矩阵
    Matrix weightMatrix;
    //偏置向量
    Matrix biasVector;
public:
    //构造函数
    GraphNodeImpl(int id, NodeType type,
                  size_t inputSize = 1,
                  size_t outputSize = 1);
    //构造函数
    GraphNodeImpl(int id, NodeType type, size_t inputSize, size_t outputSize, int layerId);
    //获取所属层编号
    int getLayerId() const override;
    //设置所属层编号
    void setLayerId(int layerId) override;
    //获取节点编号
    int getId() const override{return Id;}
    //设置节点ID
    void setId(const int& id) override{Id = id;}
    //获取节点类型
    NodeType getNodeType() const override{return nodetype;}
    //获取当前节点输入节点列表
    QList<int> getInputNodes() const override{return inputNodes;}
    //获取当前节点输出节点列表
    QList<int> getOutputNodes() const override{return outputNodes;}
    //添加节点为当前节点输入
    void addInputNode(int nodeId) override;
    //添加节点为当前节点输出
    void addOutputNode(int nodeId) override;
    //移除当前节点的某个输入节点
    void removeInputNode(int nodeId) override;
    //移除当前节点的某个输出节点
    void removeOutputNode(int nodeId) override;

    //获取权重矩阵
    Matrix getWeightMatrix() const override{return weightMatrix;}
    //设置权重矩阵
    void setWeightMatrix(const Matrix& weights) override;
    //获取当前节点偏置向量
    Matrix getBiasVector() const override{return biasVector;}
    //设置当前节点偏置向量
    void setBiasVector(const Matrix& bias) override;

    //获取当前节点输入维度
    size_t getInputSize() const override{return inputSize;}
    //获取当前节点输出维度
    size_t getOutputSize() const override{return outputSize;}
    //设置当前节点输入维度
    void setInputSize(size_t size) override;
    //设置当前节点输出维度
    void setOutputSize(size_t size) override;

    //初始化权重
    void initializeWeights() override;
};

#endif // GRAPHNODE_H



