#include "modeltest.h"

//无参构造函数
ModelTest::ModelTest() {}

//带参数构造函数
ModelTest::ModelTest(int e, double lr, int in, std::vector<int> hid, int out, ActivationType ty)
{
    this->epochs = e;
    this->learningRate = lr;
    this->NN_input = in;
    this->NN_hidden = hid;
    this->NN_output = out;
    this->type = ty;
}

//1-1-1神经网络
void TestModel_1_1_1() {
    std::cout << "\n Neural Network Manual Gradient Check ===" << std::endl;

    // 创建网络和图
    auto mlp = std::make_shared<MLPNetworkImpl>();
    auto graph = std::make_shared<MLPGraphImpl>();

    // 输入层
    auto inputNode = std::make_shared<GraphNodeImpl>(0, NodeType::INPUT, 0, 1, 0);
    graph->addNode(inputNode);

    // 隐藏层
    auto hiddenNode = std::make_shared<GraphNodeImpl>(1, NodeType::HIDDEN, 1, 1, 1);
    graph->addNode(hiddenNode);
    graph->addEdge(inputNode->getId(), hiddenNode->getId());

    // 输出层
    auto outputNode = std::make_shared<GraphNodeImpl>(2, NodeType::OUTPUT, 2, 1, 2);
    graph->addNode(outputNode);
    graph->addEdge(hiddenNode->getId(), outputNode->getId());

    mlp->setGraph(graph);
    mlp->setActivationType(ActivationType::SIGMOID);

    hiddenNode->setWeightMatrix(Matrix({{1.0}}));
    hiddenNode->setBiasVector(Matrix({{0.0}}));

    outputNode->setWeightMatrix(Matrix({{1.0}}));
    outputNode->setBiasVector(Matrix({{0.0}}));

    // 输入样本 x = 1，目标输出 y = 0
    Matrix input = Matrix({{1.0}}).transpose();
    Matrix target = Matrix({{0.0}});
    mlp->setTrainingData({input}, {target});

    // 正向传播
    std::cout << "\n---- Forward Propagation ----\n";
    mlp->forwardPass(input);

    // 输出隐藏层状态
    {
        int id = hiddenNode->getId();
        Matrix z = mlp->getLastZ(id);
        Matrix a = mlp->getNodeValue(id);
        std::cout << "Hidden z=" << z(0, 0)
                  << ", a=" << a(0, 0) << "\n";
    }

    // 输出输出层状态
    {
        int id = outputNode->getId();
        Matrix z = mlp->getLastZ(id);
        Matrix a = mlp->getNodeValue(id);
        std::cout << "Output z=" << z(0, 0)
                  << ", a=" << a(0, 0) << "\n";
    }

    // 反向传播
    std::cout << "\n---- Backward Propagation ----\n";
    mlp->backwardPass(target);

    // 输出隐藏层梯度
    {
        int id = hiddenNode->getId();
        Matrix d = mlp->getGradient(id);
        std::cout << "Hidden delta=" << d(0, 0) << "\n";
    }

    // 输出输出层梯度
    {
        int id = outputNode->getId();
        Matrix d = mlp->getGradient(id);
        std::cout << "Output delta=" << d(0, 0) << "\n";
    }

    // 手动梯度计算
    std::cout << "\n---- Manual Gradient Calculation ----\n";
    double x = input(0,0);
    double y = target(0,0);

    // 隐藏层计算
    double w1 = 1.0;
    double b1 = 0.0;
    double z1 = x * w1 + b1;
    double a1 = 1.0 / (1.0 + std::exp(-z1));

    // 输出层计算
    double w2 = 1.0;
    double b2 = 0.0;
    double z2 = a1 * w2 + b2;
    double y_hat = 1.0 / (1.0 + std::exp(-z2));

    // 导数计算
    double dL_dyhat = y_hat - y;
    double dyhat_dz2 = y_hat * (1 - y_hat);
    double dz2_da1 = w2;

    double da1_dz1 = a1 * (1 - a1);
    double dz1_dw1 = x;

    // 输出层delta
    double delta2 = dL_dyhat * dyhat_dz2;

    // 隐藏层delta
    double delta1 = delta2 * dz2_da1 * da1_dz1;

    // 梯度计算
    double grad_w1 = delta1 * dz1_dw1;
    double grad_b1 = delta1;

    double grad_w2 = delta2 * a1;
    double grad_b2 = delta2;

    // 输出手动计算结果
    std::cout << "Input x = " << x << "\n";
    std::cout << "Target y = " << y << "\n\n";

    std::cout << "Hidden: z1 = " << z1 << ", a1 = " << a1 << "\n";
    std::cout << "Output: z2 = " << z2 << ", y_hat = " << y_hat << "\n\n";

    std::cout << "Output delta = " << delta2 << "\n";
    std::cout << "Hidden delta = " << delta1 << "\n\n";

    std::cout << "Gradient w1 = " << grad_w1 << ", b1 = " << grad_b1 << "\n";
    std::cout << "Gradient w2 = " << grad_w2 << ", b2 = " << grad_b2 << "\n";

    // 权重更新
    std::cout << "\n---- Updated Parameters (Learning Rate=0.1) ----\n";
    mlp->updateWeights(0.1);

    // 输出更新后的权重和偏置
    {
        Matrix w = hiddenNode->getWeightMatrix();
        Matrix b = hiddenNode->getBiasVector();
        std::cout << "Hidden weight=" << w(0, 0)
                  << ", bias=" << b(0, 0) << "\n";
    }

    {
        Matrix w = outputNode->getWeightMatrix();
        Matrix b = outputNode->getBiasVector();
        std::cout << "Output weight=" << w(0, 0)
                  << ", bias=" << b(0, 0) << "\n";
    }
}

//1-2-1神经网络
void TestModel_1_2_1() {
    auto graph = std::make_shared<MLPGraphImpl>();
    std::vector<std::shared_ptr<GraphNode>> input, hidden, output;

    // 输入层
    input.push_back(std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::INPUT, 0, 1, 0));
    graph->addNode(input[0]);

    // 隐藏层
    for (int i = 0; i < 2; ++i) {
        auto n = std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::HIDDEN, 1, 1, 1);
        graph->addNode(n);
        graph->addEdge(input[0]->getId(), n->getId());
        hidden.push_back(n);
    }

    // 输出层
    auto n_out = std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::OUTPUT, 2, 1, 2);
    graph->addNode(n_out);
    for (auto& h : hidden)
        graph->addEdge(h->getId(), n_out->getId());

    graph->validateAndAdjustDimensions();
    auto mlp = std::make_shared<MLPNetworkImpl>();
    mlp->setGraph(graph);

    // 设置权重和偏置
    hidden[0]->setWeightMatrix(Matrix({{0.5}}));
    hidden[0]->setBiasVector(Matrix({{0.0}}));

    hidden[1]->setWeightMatrix(Matrix({{-0.5}}));
    hidden[1]->setBiasVector(Matrix({{0.0}}));

    n_out->setWeightMatrix(Matrix({{1.0, 1.0}}));
    n_out->setBiasVector(Matrix({{0.0}}));

    mlp->setActivationType(ActivationType::SIGMOID);

    // 正向传播
    Matrix x({{1.0}});
    mlp->forwardPass(x);

    std::cout << "---- Forward ----\n";
    for (int i = 0; i < 2; ++i) {
        int id = hidden[i]->getId();
        Matrix z = mlp->getLastZ(id);
        Matrix a = mlp->getNodeValue(id);
        std::cout << "Hidden[" << i << "] z=" << z(0, 0) << ", a=" << a(0, 0) << "\n";
    }
    {
        int id = n_out->getId();
        Matrix z = mlp->getLastZ(id);
        Matrix a = mlp->getNodeValue(id);
        std::cout << "Output z=" << z(0, 0) << ", a=" << a(0, 0) << "\n";
    }

    // 反向传播
    Matrix y({{1.0}});
    mlp->backwardPass(y);

    std::cout << "---- Backward ----\n";
    for (int i = 0; i < 2; ++i) {
        int id = hidden[i]->getId();
        Matrix d = mlp->getGradient(id);
        std::cout << "Hidden[" << i << "] delta=" << d(0, 0) << "\n";
    }
    {
        int id = n_out->getId();
        Matrix d = mlp->getGradient(id);
        std::cout << "Output delta=" << d(0, 0) << "\n";
    }

    // 权重更新（学习率 0.1）
    mlp->updateWeights(0.1);

    std::cout << "---- Updated Parameters ----\n";
    for (int i = 0; i < 2; ++i) {
        Matrix w = hidden[i]->getWeightMatrix();
        Matrix b = hidden[i]->getBiasVector();
        std::cout << "Hidden[" << i << "] weight=" << w(0, 0) << ", bias=" << b(0, 0) << "\n";
    }
    {
        Matrix w = n_out->getWeightMatrix();
        Matrix b = n_out->getBiasVector();
        std::cout << "Output weight=(" << w(0, 0) << ", " << w(0, 1) << "), bias=" << b(0, 0) << "\n";
    }
}

//简单数据集
void ModelTest::SimpleDataTest()
{
    // 创建 MLP 网络
    std::shared_ptr<MLPNetwork> mlp = std::make_shared<MLPNetworkImpl>();

    // 创建 MLP 图
    std::shared_ptr<MLPGraph> graph = std::make_shared<MLPGraphImpl>();

    // 定义每层神经元数量
    int inputNeurons = this->NN_input;
    std::vector<int> hiddenLayerSizes = this->NN_hidden;
    int outputNeurons = this->NN_output;

    // 创建输入层（每个输入特征作为一个节点）
    std::vector<std::shared_ptr<GraphNode>> inputNodes;
    for (int i = 0; i < inputNeurons; ++i) {
        auto node = std::make_shared<GraphNodeImpl>(
            graph->getNextNodeId(), NodeType::INPUT,
            0,  // 输入层无输入
            1,  // 每个输入节点输出1个值
            0   // 层ID=0
            );
        graph->addNode(node);
        inputNodes.push_back(node);
    }

    // 创建隐藏层
    std::vector<std::shared_ptr<GraphNode>> previousLayer = inputNodes;
    for (size_t layerIndex = 0; layerIndex < hiddenLayerSizes.size(); ++layerIndex) {
        size_t neurons = hiddenLayerSizes[layerIndex];
        std::vector<std::shared_ptr<GraphNode>> currentLayer;

        // 计算前层总输出维度（所有神经元输出值之和）
        size_t prevOutputDim = 0;
        for (const auto& node : previousLayer) {
            prevOutputDim += node->getOutputSize();
        }

        for (size_t i = 0; i < neurons; ++i) {
            auto node = std::make_shared<GraphNodeImpl>(
                graph->getNextNodeId(), NodeType::HIDDEN,
                prevOutputDim,  // 关键修正：输入维度=前层总输出维度
                1,              // 每个神经元输出1个值
                layerIndex + 1  // 层ID
                );
            graph->addNode(node);
            currentLayer.push_back(node);

            // 连接到前层所有神经元
            for (const auto& prevNode : previousLayer) {
                graph->addEdge(prevNode->getId(), node->getId());
            }
        }
        previousLayer = currentLayer;
    }

    // 创建输出层
    size_t prevOutputDim = 0;
    for (const auto& node : previousLayer) {
        prevOutputDim += node->getOutputSize();
    }

    std::vector<std::shared_ptr<GraphNode>> outputNodes;
    for (int i = 0; i < outputNeurons; ++i) {
        auto node = std::make_shared<GraphNodeImpl>(
            graph->getNextNodeId(), NodeType::OUTPUT,
            prevOutputDim,  // 输入维度=前层总输出维度
            1,              // 每个输出神经元输出1个值
            hiddenLayerSizes.size() + 1
            );
        graph->addNode(node);
        outputNodes.push_back(node);

        // 连接到前层所有神经元
        for (const auto& prevNode : previousLayer) {
            graph->addEdge(prevNode->getId(), node->getId());
        }
    }

    // 验证并调整所有节点维度
    graph->validateAndAdjustDimensions();

    // 设置 MLP 网络的拓扑结构
    mlp->setGraph(graph);

    //加载数据
    LoadSimpleData();

    // 设置训练数据
    mlp->setTrainingData(trainingInputs, trainingOutputs);

    // 设置激活函数类型
    static_cast<MLPNetworkImpl*>(mlp.get())->setActivationType(this->type);

    // 训练过程
    std::cout << "Training started..." << std::endl;
    double prevLoss = std::numeric_limits<double>::max();
    int stagnantCount = 0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        mlp->train(1, learningRate);
        double loss = mlp->getLoss();

        // 每1000个epoch打印一次
        if (epoch % 1000 == 0 || epoch == 1) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss << std::endl;
        }

        // 早停机制
        if (loss < 0.01) {
            std::cout << "Early stopping at epoch " << epoch << " with loss: " << loss << std::endl;
            break;
        }

        // 检查是否陷入局部最优
        if (std::abs(prevLoss - loss) < 1e-8) {
            stagnantCount++;
            if (stagnantCount > 1000) {
                std::cout << "Training stagnant, stopping at epoch " << epoch << std::endl;
                break;
            }
        } else {
            stagnantCount = 0;
        }
        prevLoss = loss;
    }

    std::cout << "Training completed." << std::endl;

    // 输出每个神经元的权重矩阵和偏置
    QList<int> topoOrder = graph->getTopoLogicalOrder();
    for (int id : topoOrder) {
        auto node = graph->getNode(id);
        if (node && node->getNodeType() != NodeType::INPUT) {
            std::cout << "Node ID: " << id << ", Layer ID: " << node->getLayerId() << std::endl;
            std::cout << "Weight Matrix:" << std::endl;
            node->getWeightMatrix().printMatrix();
            std::cout << "Bias Vector:" << std::endl;
            node->getBiasVector().printMatrix();
            std::cout << std::endl;
        }
    }

    // 使用训练好的模型进行预测
    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < trainingInputs.size(); ++i) {
        // 动态构建输入字符串
        std::string inputStr = "(";
        for (size_t row = 0; row < trainingInputs[i].getRows(); ++row) {
            inputStr += std::to_string(trainingInputs[i](row, 0));
            if (row < trainingInputs[i].getRows() - 1) {
                inputStr += ", ";
            }
        }
        inputStr += ")";

        Matrix result = mlp->predict(trainingInputs[i]);
        std::cout << "Input: " << inputStr
                  << " -> Output: " << result(0,0)
                  << " (Expected: " << trainingOutputs[i](0,0) << ")" << std::endl;
    }
}

//加载csv文件
bool LoadIrisCSV(const std::string& filepath,
                 std::vector<Matrix>& inputs,
                 std::vector<Matrix>& outputs) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> features;
        std::string label;
        int colIdx = 0;

        // 鸢尾花数据集每行有5个值：前4个为特征，最后1个为类别
        while (std::getline(ss, cell, ',')) {
            if (colIdx < 4) {  // 前4列是特征
                features.push_back(std::stod(cell));
            } else {  // 第5列是标签
                label = cell;
            }
            ++colIdx;
        }

        // 特征矩阵转置为列向量
        Matrix input = Matrix({features}).transpose();
        // 构建独热编码输出(3个类别)
        Matrix output(3, 1);
        if (label == "Iris-setosa") {
            output = Matrix({{1.0}, {0.0}, {0.0}});
        } else if (label == "Iris-versicolor") {
            output = Matrix({{0.0}, {1.0}, {0.0}});
        } else if (label == "Iris-virginica") {
            output = Matrix({{0.0}, {0.0}, {1.0}});
        } else {
            continue;  // 跳过无效标签
        }

        inputs.push_back(input);
        outputs.push_back(output);
    }

    file.close();
    return true;
}

//鸢尾花数据集
void ModelTest::TestCsv()
{
    std::cout << "📥 测试 Iris 鸢尾花数据集 (多分类，SIGMOID激活函数)..." << std::endl;

    // 1. 加载原始数据
    std::vector<Matrix> allInputs, allOutputs;
    if (!LoadIrisCSV("E:/QTproject/mpl6/MLP/iris.data", allInputs, allOutputs)) {
        std::cerr << "❌ 加载 iris.data 文件失败！" << std::endl;
        return;
    }

    // 2. 划分训练集和测试集（7:3比例）
    const double trainRatio = 0.7;  // 训练集占比
    const int totalSamples = allInputs.size();
    const int trainSize = static_cast<int>(totalSamples * trainRatio);

    // 为了结果可重复，固定随机种子
    std::srand(42);

    // 生成随机索引并打乱（确保每个类别均匀分布）
    std::vector<int> indices(totalSamples);
    for (int i = 0; i < totalSamples; ++i) indices[i] = i;
    std::random_shuffle(indices.begin(), indices.end());

    // 分离训练集和测试集
    std::vector<Matrix> trainInputs, trainOutputs;
    std::vector<Matrix> testInputs, testOutputs;
    for (int i = 0; i < totalSamples; ++i) {
        if (i < trainSize) {
            trainInputs.push_back(allInputs[indices[i]]);
            trainOutputs.push_back(allOutputs[indices[i]]);

        } else {
            testInputs.push_back(allInputs[indices[i]]);
            testOutputs.push_back(allOutputs[indices[i]]);
        }
    }

    // 3. 对训练集进行归一化（注意：用训练集的min/max归一化测试集，避免数据泄露）
    int inputNeurons = trainInputs[0].getRows();  // 4个特征
    std::vector<double> minVals(inputNeurons, std::numeric_limits<double>::max());
    std::vector<double> maxVals(inputNeurons, std::numeric_limits<double>::lowest());

    // 计算训练集的min和max
    for (const auto& input : trainInputs) {
        for (int i = 0; i < inputNeurons; ++i) {
            double val = input(i, 0);
            if (val < minVals[i]) minVals[i] = val;
            if (val > maxVals[i]) maxVals[i] = val;
        }
    }

    // 归一化训练集
    for (auto& input : trainInputs) {
        for (int i = 0; i < inputNeurons; ++i) {
            double range = maxVals[i] - minVals[i];
            input(i, 0) = (range < 1e-9) ? 0.5 : (input(i, 0) - minVals[i]) / range;
        }
    }

    // 用训练集的min/max归一化测试集（关键：避免测试集信息泄露到训练中）
    for (auto& input : testInputs) {
        for (int i = 0; i < inputNeurons; ++i) {
            double range = maxVals[i] - minVals[i];
            input(i, 0) = (range < 1e-9) ? 0.5 : (input(i, 0) - minVals[i]) / range;
        }
    }
    // 直接输出 trainInputs 所有内容
    std::cout << "=== trainInputs 数据详情 ===" << std::endl;
    for (size_t sample = 0; sample < trainInputs.size(); ++sample) {
        const Matrix& mat = trainInputs[sample];
        std::cout << "样本 " << sample << " (维度: "
                  << mat.getRows() << "行 × " << mat.getCols() << "列):" << std::endl;

        // 遍历矩阵行和列
        for (int r = 0; r < mat.getRows(); ++r) {
            for (int c = 0; c < mat.getCols(); ++c) {
                std::cout << std::fixed << std::setprecision(4)
                          << mat(r, c) << "\t";  // 保留4位小数，制表符分隔
            }
            std::cout << std::endl;  // 每行结束换行
        }
        std::cout << std::endl;  // 样本间空行分隔
    }
    // 4. 创建网络并训练（仅用训练集）
    std::shared_ptr<MLPNetwork> mlp = std::make_shared<MLPNetworkImpl>();
    std::shared_ptr<MLPGraph> graph = std::make_shared<MLPGraphImpl>();

    std::vector<int> hiddenLayerSizes = {15};
    int outputNeurons = trainOutputs[0].getRows();  // 3个类别

    // 构建网络结构（与之前相同）
    std::vector<std::shared_ptr<GraphNode>> inputNodes;
    for (int i = 0; i < inputNeurons; ++i) {
        auto node = std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::INPUT, 0, 1, 0);
        graph->addNode(node);
        inputNodes.push_back(node);
    }

    std::vector<std::shared_ptr<GraphNode>> previousLayer = inputNodes;
    for (size_t layerIndex = 0; layerIndex < hiddenLayerSizes.size(); ++layerIndex) {
        int neurons = hiddenLayerSizes[layerIndex];
        std::vector<std::shared_ptr<GraphNode>> currentLayer;
        int prevOutputDim = 0;
        for (auto& n : previousLayer) prevOutputDim += n->getOutputSize();

        for (int i = 0; i < neurons; ++i) {
            auto node = std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::HIDDEN, prevOutputDim, 1, layerIndex + 1);
            graph->addNode(node);
            currentLayer.push_back(node);
            for (auto& prev : previousLayer)
                graph->addEdge(prev->getId(), node->getId());
        }
        previousLayer = currentLayer;
    }

    int prevOutputDim = 0;
    for (auto& n : previousLayer) prevOutputDim += n->getOutputSize();
    auto outputNode = std::make_shared<GraphNodeImpl>(graph->getNextNodeId(), NodeType::OUTPUT, prevOutputDim, outputNeurons, hiddenLayerSizes.size() + 1);
    graph->addNode(outputNode);
    for (auto& prev : previousLayer)
        graph->addEdge(prev->getId(), outputNode->getId());

    graph->validateAndAdjustDimensions();
    mlp->setGraph(graph);
    mlp->setTrainingData(trainInputs, trainOutputs);  // 仅用训练集训练
    static_cast<MLPNetworkImpl*>(mlp.get())->setActivationType(ActivationType::SIGMOID);

    // 5. 训练模型
    std::cout << "训练中..." << std::endl;
    int epochs = 1000;
    double learningRate = 0.05;
    for (int e = 1; e <= epochs; ++e) {
        mlp->train(1, learningRate);

            std::cout << "Epoch " << e << ", 训练损失: " << mlp->getLoss() << std::endl;

        if (mlp->getLoss() < 0.01) {
            std::cout << "Early stopping at epoch " << e << " with loss: " << mlp->getLoss() << std::endl;
            break;
        }
    }

    // 6. 新增：输出预测结果（训练集和测试集）
    auto printPredictions = [&](const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, const std::string& name) {
        std::cout << "\n===== " << name << "预测结果 =====" << std::endl;
        for (size_t i = 0; i < inputs.size(); ++i) {  // 遍历所有样本
            Matrix pred = mlp->predict(inputs[i]);
            std::cout << "样本 " << i << " 预测: [ ";
            // 打印预测值（保留4位小数）
            for (size_t j = 0; j < pred.getRows(); ++j) {
                std::cout << std::fixed << std::setprecision(4) << pred(j, 0) << " ";
            }
            std::cout << "]，实际: [ ";
            // 打印实际值
            for (size_t j = 0; j < outputs[i].getRows(); ++j) {
                std::cout << outputs[i](j, 0) << " ";
            }
            std::cout << "]" << std::endl;
        }
    };

    // 输出训练集和测试集的预测结果（可根据需要注释其中一个）
    printPredictions(trainInputs, trainOutputs, "训练集");
    printPredictions(testInputs, testOutputs, "测试集");

    // 7. 评估：分别计算训练集和测试集准确率
    auto calculateAccuracy = [&](const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, const std::string& name) {
        int correct = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Matrix pred = mlp->predict(inputs[i]);
            int predictedClass = 0;
            double maxVal = pred(0, 0);
            for (int j = 1; j < 3; ++j) {
                if (pred(j, 0) > maxVal) {
                    maxVal = pred(j, 0);
                    predictedClass = j;
                }
            }
            int actualClass = 0;
            for (int j = 0; j < 3; ++j) {
                if (outputs[i](j, 0) == 1.0) {
                    actualClass = j;
                    break;
                }
            }
            if (predictedClass == actualClass) correct++;
        }
        double acc = static_cast<double>(correct) / inputs.size() * 100.0;
        std::cout << "🔍 " << name << "准确率: " << acc << "%" << std::endl;
        return acc;
    };

    // 输出评估结果
    calculateAccuracy(trainInputs, trainOutputs, "训练集");  // 反映模型拟合能力
    calculateAccuracy(testInputs, testOutputs, "测试集");    // 反映泛化能力
}

//加载简单数据集
void ModelTest::LoadSimpleData()
{
    // // 准备训练数据
    // this->trainingInputs = {
    //     Matrix({{0}}).transpose(),
    //     Matrix({{1}}).transpose()
    // };
    // this->trainingOutputs = {
    //     Matrix({{0}}),
    //     Matrix({{1}})
    // };

    // 准备训练数据 (AND逻辑门)
    this->trainingInputs = {
        Matrix({{0, 0}}).transpose(),
        Matrix({{1, 1}}).transpose(),
        Matrix({{0, 1}}).transpose(),
        Matrix({{1, 0}}).transpose()
    };
    this->trainingOutputs = {
        Matrix({{0}}),
        Matrix({{0}}),
        Matrix({{1}}),
        Matrix({{1}})
    };

    // // 准备训练数据 (XOR逻辑门)
    // this->trainingInputs = {
    //     Matrix({{0, 0}}).transpose(),
    //     Matrix({{0, 1}}).transpose(),
    //     Matrix({{1, 0}}).transpose(),
    //     Matrix({{1, 1}}).transpose()
    // };
    // this->trainingOutputs = {
    //     Matrix({{0}}),
    //     Matrix({{1}}),
    //     Matrix({{1}}),
    //     Matrix({{0}})
    // };

    // // 准备训练数据 (XNOR逻辑门)
    // this->trainingInputs = {
    //     Matrix({{0, 0}}).transpose(),
    //     Matrix({{0, 1}}).transpose(),
    //     Matrix({{1, 0}}).transpose(),
    //     Matrix({{1, 1}}).transpose()
    // };
    // this->trainingOutputs = {
    //     Matrix({{1}}),
    //     Matrix({{0}}),
    //     Matrix({{0}}),
    //     Matrix({{1}})
    // };

    // // 准备训练数据 (三输入AND逻辑门)
    // this->trainingInputs = {
    //     Matrix({{0, 0, 0}}).transpose(),
    //     Matrix({{0, 0, 1}}).transpose(),
    //     Matrix({{0, 1, 0}}).transpose(),
    //     Matrix({{0, 1, 1}}).transpose(),
    //     Matrix({{1, 0, 0}}).transpose(),
    //     Matrix({{1, 0, 1}}).transpose(),
    //     Matrix({{1, 1, 0}}).transpose(),
    //     Matrix({{1, 1, 1}}).transpose()
    // };
    // this->trainingOutputs = {
    //     Matrix({{0}}), Matrix({{0}}), Matrix({{0}}), Matrix({{0}}),
    //     Matrix({{0}}), Matrix({{0}}), Matrix({{0}}), Matrix({{1}})
    // };

    // // 准备训练数据 (3输入奇偶校验：1的个数为奇数时输出1)
    // this->trainingInputs = {
    //     Matrix({{0, 0, 0}}).transpose(),  // 输入：0,0,0 → 输出：0
    //     Matrix({{0, 0, 1}}).transpose(),  // 输入：0,0,1 → 输出：1
    //     Matrix({{0, 1, 0}}).transpose(),  // 输入：0,1,0 → 输出：1
    //     Matrix({{0, 1, 1}}).transpose(),  // 输入：0,1,1 → 输出：0
    //     Matrix({{1, 0, 0}}).transpose(),  // 输入：1,0,0 → 输出：1
    //     Matrix({{1, 0, 1}}).transpose(),  // 输入：1,0,1 → 输出：0
    //     Matrix({{1, 1, 0}}).transpose(),  // 输入：1,1,0 → 输出：0
    //     Matrix({{1, 1, 1}}).transpose()   // 输入：1,1,1 → 输出：1
    // };
    // this->trainingOutputs = {
    //     Matrix({{0}}),  // 对应输入 (0,0,0)
    //     Matrix({{1}}),  // 对应输入 (0,0,1)
    //     Matrix({{1}}),  // 对应输入 (0,1,0)
    //     Matrix({{0}}),  // 对应输入 (0,1,1)
    //     Matrix({{1}}),  // 对应输入 (1,0,0)
    //     Matrix({{0}}),  // 对应输入 (1,0,1)
    //     Matrix({{0}}),  // 对应输入 (1,1,0)
    //     Matrix({{1}})   // 对应输入 (1,1,1)
    // };
}

//简单神经网络计算值对比
void ModelTest::ComparisonTest()
{
    TestModel_1_1_1();
    /*
    梯度项                 MLP输出值	手动计算值	验证
    输出层 delta₂           0.1509	0.1481	    相近
    输出层 w₂ 梯度（权重）	   0.1103	0.1083      相近
    输出层 b₂ 梯度（偏置）	   0.1509	0.1481      相近
    隐藏层 delta₁	       0.0331	0.0291      可接受（轻微误差）
    隐藏层 w₁ 梯度（权重）	   0.0331	0.0291      可接受
    隐藏层 b₁ 梯度（偏置）	   0.0331	0.0291      可接受
    */
    //TestModel_1_2_1();
    /*
    前向传播
    参数          输出          理论值     是否一致
    Hidden[0]z    0.5           0.5         ✅
    Hidden[0]a    0.622459      0.622459	✅
    Hidden[1]z    –0.5          –0.5        ✅
    Hidden[1]a    0.377541      0.377541	✅
    Output z      1.0           1.0         ✅
    Output a      0.731059      0.731059	✅

    反向传播
    节点          理论 δ      实际 δ        说明
    输出层 δₒᵤₜ	–0.05248	–0.059      误差略大，浮点精度差异导致
    隐藏层 δ₀	–0.01233	–0.01341	与理论值接近，σ′(0.5)浮点略有差异
    隐藏层 δ₁	–0.01233	–0.01424	与理论值接近，同上
    */
}

