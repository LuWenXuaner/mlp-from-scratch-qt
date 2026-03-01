#ifndef MLPGRAPH_H
#define MLPGRAPH_H

#include "graphnode.h"
#include <QMap>
#include <random>
#include<memory>
class MLPGraph
{
public:
    //析构函数
    virtual ~MLPGraph() = default;
    //在图中添加节点
    virtual void addNode(std::shared_ptr<GraphNode> node) = 0;
    //在图中添加边
    virtual void addEdge(int st, int ed) = 0;
    //在图中删除节点
    virtual void removeNode(int nodeId) = 0;
    //在图中删除边
    virtual void removeEdge(int st, int ed) = 0;
    //获取图的拓扑序列
    virtual QList<int> getTopoLogicalOrder() const = 0;
    //返回图中所有节点
    virtual QList<std::shared_ptr<GraphNode>> getAllNodes() const = 0;
    //根据节点编号返回节点
    virtual std::shared_ptr<GraphNode> getNode(int nodeId) const = 0;
    //判断是否是有效的神经网络
    virtual bool isValidNetwork() const = 0;
    //清空所有内容初始化
    virtual void clear() = 0;
    //获取下一个可用节点编号
    virtual int getNextNodeId() const = 0;
    //验证并调整所有节点维度
    virtual void validateAndAdjustDimensions() = 0;
};

class MLPGraphImpl:public MLPGraph
{
private:
    //建立节点编号与指针对应关系
    QMap<int, std::shared_ptr<GraphNode>> nodes;
    //建立节点正向邻接表
    QMap<int, QList<int>> adjacencyList;
    //建立节点反向邻接表
    QMap<int, QList<int>> reverseAdjacencyList;

    //拓扑排序结果
    mutable QList<int> topoLogicalOrder;
    //标记拓扑排序结果是否有效
    mutable bool isTopoLogValid;

    //下一个节点可用编号
    int nextNodeId;

    //更新拓扑序
    void updateTopoLogicalOrder() const;
    //图中是否存在环
    bool hasCycle() const;
public:
    //无参构造函数
    MLPGraphImpl();
    //在图中添加节点
    void addNode(std::shared_ptr<GraphNode> node) override;
    //在图中添加边
    void addEdge(int st, int ed) override;
    //在图中删除节点
    void removeNode(int nodeId) override;
    //在图中删除边
    void removeEdge(int st, int ed) override;
    //获取图的拓扑序列
    QList<int> getTopoLogicalOrder() const override;
    //返回图中所有节点
    QList<std::shared_ptr<GraphNode>> getAllNodes() const override;
    //根据节点编号返回节点
    std::shared_ptr<GraphNode> getNode(int nodeId) const override;
    //判断是否是有效的神经网络
    bool isValidNetwork() const override;
    //清空所有内容初始化
    void clear() override;
    //获取下一个可用节点编号
    int getNextNodeId() const override{return nextNodeId;}
    //验证并调整所有节点维度
    void validateAndAdjustDimensions() override;
};

#endif // MLPGRAPH_H
