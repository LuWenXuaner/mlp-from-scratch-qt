#include "mlpnetwork.h"

//建立神经网络拓扑结构
void MLPNetworkImpl::setGraph(std::shared_ptr<MLPGraph> g)
{
    graph = g;
    if(graph) {
        graph->validateAndAdjustDimensions();
    }
}

//设置训练数据
void MLPNetworkImpl::setTrainingData(const std::vector<Matrix> &inputs, const std::vector<Matrix> &outputs)
{
    trainingInputs = inputs;
    trainingOutputs = outputs;
}

//训练
void MLPNetworkImpl::train(int epochs, double learningRate)
{
    if(!graph || !graph->isValidNetwork() || trainingInputs.empty())
        return;

    currentEpoch = 1;

    for(int epoch = 1;epoch <= epochs;++epoch) {
        currentEpoch = epoch;
        double totalLoss = 0.0;

        for(size_t i = 0;i < trainingInputs.size();++i) {
            forwardPass(trainingInputs[i]);

            Matrix predictResult = predict(trainingInputs[i]);
            totalLoss += calculateLoss(predictResult, trainingOutputs[i]);

            backwardPass(trainingOutputs[i]);
            updateWeights(learningRate);

        }

        currentLoss = totalLoss / trainingInputs.size();
    }

    // training = false;
    // emit trainingCompleted();
}

//使用训练好的模型进行预测
Matrix MLPNetworkImpl::predict(const Matrix &input)
{
    if(!graph)
        return Matrix();

    forwardPass(input);

    QList<int> topoOrder = graph->getTopoLogicalOrder();

    std::vector<Matrix> outputValues;
    for(int id : topoOrder) {
        auto node = graph->getNode(id);
        if(node && node->getNodeType() == NodeType::OUTPUT) {
            outputValues.push_back(nodeValues[id]);
        }
    }

    Matrix result;
    if(outputValues.size() == 1) {
        result = outputValues[0];
    } else if(outputValues.size() > 1) {
        size_t totalRows = 0, t = 0;
        for(const auto& m : outputValues) {
            totalRows += m.getRows();
        }

        result = Matrix::zeros(totalRows, 1);
        for(const auto& m : outputValues) {
            for(size_t i = 0;i < m.getRows();++i) {
                result(t++, 0) = m[i][0];
            }
        }
    }
    return result;
}

//前向传播
void MLPNetworkImpl::forwardPass(const Matrix &input)
{
    nodeValues.clear();
    QList<int> topoOrder = graph->getTopoLogicalOrder();

    if(input.isVector()) {
        std::vector<double> t = input.toVector();
        size_t i = 0;

        for(int id : topoOrder) {
            auto node = graph->getNode(id);
            if(node && node->getNodeType() == NodeType::INPUT) {
                size_t n = node->getOutputSize();
                Matrix result = Matrix::zeros(n, 1);
                for(size_t k = 0;k < n && i < t.size();++k) {
                    result(k, 0) = t[i++];
                }
                nodeValues[id] = result;
            }
        }
    } else {
        bool assigned = false;
        for(int id : topoOrder) {
            auto node = graph->getNode(id);
            if(node && node->getNodeType() == NodeType::INPUT && !assigned) {
                nodeValues[id] = input;
                assigned = true;
                break;
            }
        }
    }

    for(int id : topoOrder) {
        auto node = graph->getNode(id);
        if(!node || node->getNodeType() == NodeType::INPUT)
            continue;

        Matrix concatResult = concatenateInputs(node->getInputNodes());

        if(concatResult.isEmpty())
            continue;

        Matrix weightResult = node->getWeightMatrix();
        Matrix biasResult = node->getBiasVector();

        Matrix result = weightResult * concatResult + biasResult;
        Matrix activateResult = ActivationFunction::activate(result, activationType);

        lastZ[id] = result;
        nodeValues[id] = activateResult;
    }
}

//反向传播
void MLPNetworkImpl::backwardPass(const Matrix &target)
{
    nodeGradients.clear();
    QList<int> topoOrder = graph->getTopoLogicalOrder();

    std::vector<int> outputNodeIds;
    for(int id : topoOrder) {
        auto node = graph->getNode(id);
        if(node && node->getNodeType() == NodeType::OUTPUT) {
            outputNodeIds.push_back(id);
        }
    }

    if(outputNodeIds.size() == 1) {
        int id = outputNodeIds[0];
        Matrix output = nodeValues[id];

        // 改进梯度计算 - 移除可能导致梯度爆炸的缩放
        Matrix error = (output - target); // 简化梯度计算
        Matrix activationDerivative = ActivationFunction::derivative(output, activationType);
        nodeGradients[id] = error.hadamard(activationDerivative);

        // std::cout << "Output layer delta₂ (Node ID: " << id << "):" << std::endl;
        // nodeGradients[id].printMatrix();
    } else {
        // 处理多输出的情况
        size_t i = 0;
        for(int id : outputNodeIds) {
            auto node = graph->getNode(id);
            Matrix output = nodeValues[id];
            size_t n = output.getRows();

            Matrix outputTarget = Matrix::zeros(n, 1);
            for(size_t k = 0; k < n && i < target.getRows(); ++k) {
                outputTarget(k, 0) = target(i++, 0);
            }

            Matrix error = (output - outputTarget);
            Matrix activationDerivative = ActivationFunction::derivative(output, activationType);
            nodeGradients[id] = error.hadamard(activationDerivative);

            // std::cout << "Output layer delta₂ (Node ID: " << id << "):" << std::endl;
            // nodeGradients[id].printMatrix();
        }
    }

    // 反向传播到隐藏层
    for(int i = topoOrder.size() - 1; i >= 0; i--) {
        int id = topoOrder[i];
        auto node = graph->getNode(id);

        if(!node || node->getNodeType() == NodeType::INPUT)
            continue;

        if(!nodeGradients.contains(id)) {
            Matrix gradient = Matrix::zeros(node->getOutputSize(), 1);

            for(int outputNodeid : node->getOutputNodes()) {
                auto outputNode = graph->getNode(outputNodeid);
                if(outputNode && nodeGradients.contains(outputNodeid)) {
                    Matrix outputGradient = nodeGradients[outputNodeid];
                    Matrix outputWeight = outputNode->getWeightMatrix();

                    QList<int> inputNodes = outputNode->getInputNodes();
                    size_t stIndex = 0;

                    for(int i : inputNodes) {
                        if(i == id)
                            break;
                        auto tNode = graph->getNode(i);
                        if(tNode) {
                            stIndex += tNode->getOutputSize();
                        }
                    }

                    for(size_t j = 0; j < gradient.getRows(); ++j) {
                        for(size_t k = 0; k < outputGradient.getRows(); ++k) {
                            if(stIndex + j < outputWeight.getCols()) {
                                gradient(j, 0) += outputWeight(k, stIndex + j) * outputGradient(k, 0);
                            }
                        }
                    }
                }
            }

            Matrix activationDerivative = ActivationFunction::derivative(nodeValues[id], activationType);
            nodeGradients[id] = gradient.hadamard(activationDerivative);

            // std::cout << "Hidden layer delta₁ (Node ID: " << id << "):" << std::endl;
            // nodeGradients[id].printMatrix();
        }
    }
}

//权重更新
void MLPNetworkImpl::updateWeights(double learningRate)
{
    QList<int> topoOrder = graph->getTopoLogicalOrder();
    const double gradientClipThreshold = 5.0; // 梯度裁剪阈值

    for(int id : topoOrder) {
        auto node = graph->getNode(id);
        if(!node || node->getNodeType() == NodeType::INPUT || !nodeGradients.contains(id))
            continue;

        Matrix gradientResult = nodeGradients[id];

        // 梯度裁剪
        double gradientNorm = gradientResult.norm();
        if(gradientNorm > gradientClipThreshold) {
            gradientResult *= (gradientClipThreshold / gradientNorm);
        }

        Matrix inputResult = concatenateInputs(node->getInputNodes());
        if(inputResult.isEmpty())
            continue;

        Matrix weightGradient = gradientResult * inputResult.transpose();

        // 权重梯度裁剪
        double weightGradientNorm = weightGradient.norm();
        if(weightGradientNorm > gradientClipThreshold) {
            weightGradient *= (gradientClipThreshold / weightGradientNorm);
        }

        // std::cout << "Node ID: " << id << ", Weight gradient:" << std::endl;
        // weightGradient.printMatrix();
        // std::cout << "Node ID: " << id << ", Bias gradient:" << std::endl;
        // gradientResult.printMatrix();

        Matrix weightResult = node->getWeightMatrix();
        Matrix newWeights = weightResult - learningRate * weightGradient;
        node->setWeightMatrix(newWeights);

        Matrix biasResult = node->getBiasVector();
        Matrix newBias = biasResult - learningRate * gradientResult;
        node->setBiasVector(newBias);
    }
}

//损失计算
double MLPNetworkImpl::calculateLoss(const Matrix &predicted, const Matrix &target)
{
    if(predicted.getRows() != target.getRows() || predicted.getCols() != target.getCols())
        return 0.0;

    size_t n = predicted.getRows() * predicted.getCols();
    if(n == 0)
        return 0.0;

    Matrix diff = predicted - target;
    double ans = diff.hadamard(diff).sum();

    return ans / (2.0 * n);
}

//输入节点值拼接
Matrix MLPNetworkImpl::concatenateInputs(const QList<int> &inputNodeIds) const
{
    if(inputNodeIds.isEmpty())
        return Matrix();

    std::vector<Matrix> inputMatrix;
    size_t totalRows = 0;
    for(int id : inputNodeIds) {
        if(nodeValues.contains(id)) {
            inputMatrix.push_back(nodeValues[id]);
            totalRows += nodeValues[id].getRows();
        }
    }

    if(inputMatrix.empty())
        return Matrix();

    Matrix result = Matrix::zeros(totalRows, 1);
    size_t t = 0;
    for(auto m : inputMatrix) {
        for(size_t i = 0;i < m.getRows();++i) {
            result(t++, 0) = m(i, 0);
        }
    }
    return result;
}
