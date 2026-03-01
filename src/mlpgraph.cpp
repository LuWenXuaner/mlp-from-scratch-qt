#include "mlpgraph.h"
#include <QQueue>

MLPGraphImpl::MLPGraphImpl():isTopoLogValid(false), nextNodeId(1){}

//在图中添加节点
void MLPGraphImpl::addNode(std::shared_ptr<GraphNode> node)
{
    int id = node->getId();
    nodes[id] = node;

    adjacencyList[id] = QList<int>();
    reverseAdjacencyList[id] = QList<int>();

    isTopoLogValid = false;
    if(id >= nextNodeId) {
        nextNodeId = id + 1;
    }
}

//在图中添加边
void MLPGraphImpl::addEdge(int st, int ed)
{
    if(!nodes.contains(st) || !nodes.contains(ed))
        return;

    if(!adjacencyList[st].contains(ed)) {
        adjacencyList[st].append(ed);
        reverseAdjacencyList[ed].append(st);

        nodes[st]->addOutputNode(ed);
        nodes[ed]->addInputNode(st);

        isTopoLogValid = false;
    }
}

//在图中删除节点
void MLPGraphImpl::removeNode(int nodeId)
{
    if(!nodes.contains(nodeId))
        return;

    for(int id : nodes[nodeId]->getOutputNodes()) {
        reverseAdjacencyList[id].removeAll(nodeId);
        nodes[id]->removeInputNode(nodeId);
    }

    for(int id : nodes[nodeId]->getInputNodes()) {
        adjacencyList[id].removeAll(nodeId);
        nodes[id]->removeOutputNode(nodeId);
    }

    nodes.remove(nodeId);
    adjacencyList.remove(nodeId);;
    reverseAdjacencyList.remove(nodeId);
    isTopoLogValid = false;
}

//在图中删除边
void MLPGraphImpl::removeEdge(int st, int ed)
{
    if(!nodes.contains(st) || !nodes.contains(ed))
        return;

    if(!adjacencyList[st].contains(ed))
        return;

    adjacencyList[st].removeAll(ed);
    reverseAdjacencyList[ed].removeAll(st);
    nodes[st]->removeOutputNode(ed);
    nodes[ed]->removeInputNode(st);
    isTopoLogValid = false;
}

//获取图的拓扑序列
QList<int> MLPGraphImpl::getTopoLogicalOrder() const
{
    if(!isTopoLogValid) {
        updateTopoLogicalOrder();
    }
    return topoLogicalOrder;
}

//更新拓扑序
void MLPGraphImpl::updateTopoLogicalOrder() const
{
    topoLogicalOrder.clear();

    QMap<int, int> inDegree;
    for(auto it = nodes.begin();it != nodes.end();++it) {
        inDegree[it.key()] = reverseAdjacencyList[it.key()].size();
    }

    QQueue<int> q;

    for(auto it = inDegree.begin();it != inDegree.end();++it) {
        if(it.value() == 0) {
            q.enqueue(it.key());
        }
    }

    while(!q.empty()) {
        int t = q.dequeue();
        for(auto i : adjacencyList[t]) {
            inDegree[i]--;
            if(inDegree[i] == 0) {
                q.enqueue(i);
                inDegree.remove(i);
            }
        }
        topoLogicalOrder.append(t);
    }
    isTopoLogValid = true;
}

//图中是否存在环
bool MLPGraphImpl::hasCycle() const
{
    return getTopoLogicalOrder().size() != nodes.size();
}

//返回图中所有节点
QList<std::shared_ptr<GraphNode>> MLPGraphImpl::getAllNodes() const
{
    return nodes.values();
}

//根据节点编号返回节点
std::shared_ptr<GraphNode> MLPGraphImpl::getNode(int nodeId) const
{
    return nodes.value(nodeId, nullptr);
}

//判断是否是有效的神经网络
bool MLPGraphImpl::isValidNetwork() const
{
    if(nodes.isEmpty())
        return false;

    bool hasInput = false, hasOutput = false;
    for(auto it = nodes.begin();it != nodes.end();++it) {
        if(it.value()->getNodeType() == NodeType::INPUT) {
            hasInput = true;
        }
        if(it.value()->getNodeType() == NodeType::OUTPUT) {
            hasOutput = true;
        }
    }

    return hasInput && hasOutput && !hasCycle();
}

//清空所有内容初始化
void MLPGraphImpl::clear()
{
    nodes.clear();
    adjacencyList.clear();
    reverseAdjacencyList.clear();
    topoLogicalOrder.clear();
    isTopoLogValid = false;
    nextNodeId = 1;
}

//验证并调整所有节点维度
void MLPGraphImpl::validateAndAdjustDimensions()
{
    QList<int> topoOrder = getTopoLogicalOrder();

    for(int id : topoOrder) {
        auto node = nodes[id];
        if(!node || node->getNodeType() == NodeType::INPUT)
            continue;

        QList<int> inputNodeIds = node->getInputNodes();
        if(inputNodeIds.isEmpty())
            continue;

        size_t sum = 0;
        for(int i : inputNodeIds) {
            if(nodes[i]) {
                sum += nodes[i]->getOutputSize();
            }
        }

        if(sum > 0) {
            nodes[id]->setInputSize(sum);
        }
    }

    for(int id : topoOrder) {
        auto node = nodes[id];
        if(node && node->getNodeType() != NodeType::INPUT) {
            node->initializeWeights();
        }
    }
}
