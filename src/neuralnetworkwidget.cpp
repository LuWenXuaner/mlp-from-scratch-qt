#include "neuralnetworkwidget.h"
#include "modeltest.h"
#include <QApplication>
#include <QMessageBox>
#include <QFileDialog>
#include <QDateTime>
#include <QSplitter>
#include <QScrollArea>
#include <QtMath>
#include <QHeaderView>
#include <QTableWidget>
#include <QVBoxLayout>
#include <cmath>
#include <random>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QDebug>
#include<QHBoxLayout>

// NetworkVisualization 实现
NetworkVisualization::NetworkVisualization(QWidget* parent)
    : QGraphicsView(parent)
{
    scene = new QGraphicsScene(this);
    setScene(scene);
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    // 为了结果可重复，固定随机种子
    std::srand(42);


}

NetworkVisualization::~NetworkVisualization()
{
    clearVisualization();
}

void NetworkVisualization::setNetwork(std::shared_ptr<MLPGraph> graph)
{  srand(40);
    network = graph;
    drawNetwork();
}

void NetworkVisualization::drawNetwork()
{
    clearVisualization();

    if (!network) return;

    auto nodes = network->getAllNodes();
    if (nodes.isEmpty()) return;

    // 按层和层数分组节点
    QMap<NodeType, QMap<int, QList<std::shared_ptr<GraphNode>>>> layerNodes;
    for (const auto &node : nodes) {
        layerNodes[node->getNodeType()][node->getLayerId()].append(node);
    }

    // 计算每层的最大节点数
    int maxNodesPerLayer = 0;
    for (auto typeIt = layerNodes.constBegin(); typeIt != layerNodes.constEnd(); ++typeIt) {
        for (auto layerIt = typeIt.value().constBegin(); layerIt != typeIt.value().constEnd(); ++layerIt) {
            maxNodesPerLayer = qMax(maxNodesPerLayer, layerIt.value().size());
        }
    }

    // 绘制节点
    int layerIndex = 0;
    QList<NodeType> layerOrder = {NodeType::INPUT, NodeType::HIDDEN, NodeType::OUTPUT};
    layerPositions.clear();

    for (NodeType type : layerOrder) {
        if (!layerNodes.contains(type)) continue;

        for (auto layerIdIt = layerNodes[type].constBegin(); layerIdIt != layerNodes[type].constEnd(); ++layerIdIt) {
            auto layerNodeList = layerIdIt.value();
            QPointF layerPos(100 + layerIndex * 250, 200);
            layerPositions[type] = layerPos;

            for (int i = 0; i < layerNodeList.size(); ++i) {
                auto node = layerNodeList[i];
                QPointF pos = calculateNodePosition(node->getId(), layerIndex, i, layerNodeList.size(), maxNodesPerLayer);

                // 绘制节点圆圈
                QGraphicsEllipseItem* nodeItem = scene->addEllipse(pos.x() - 30, pos.y() - 30, 60, 60);
                nodeItem->setBrush(QBrush(type == NodeType::INPUT ? Qt::lightGray :
                                         type == NodeType::OUTPUT ? Qt::lightGray : Qt::lightGray));
                nodeItem->setPen(QPen(Qt::black, 2));
                nodeItems[node->getId()] = nodeItem;
                nodePositions[node->getId()] = pos;

                // 节点标签
                QGraphicsTextItem* label = scene->addText(QString::number(node->getId()));
                label->setPos(pos.x() - 10, pos.y() - 10);
                nodeLabels[node->getId()] = label;

                // 节点值显示
                QGraphicsTextItem* valueLabel = scene->addText("0.00");
                valueLabel->setPos(pos.x() - 15, pos.y() + 35);
                valueLabel->setDefaultTextColor(Qt::red);
                nodeValues[node->getId()] = valueLabel;

                // 节点标题（层类型）
                if (i == 0) {
                    QString layerName;
                    if (type == NodeType::INPUT) layerName = "输入层";
                    else if (type == NodeType::HIDDEN) layerName = QString("隐藏层%1").arg(layerIdIt.key());
                    else layerName = "输出层";

                    QGraphicsTextItem* layerTitle = scene->addText(layerName);
                    layerTitle->setPos(pos.x() - 30, pos.y() - 80);
                    layerTitle->setFont(QFont("Arial", 12, QFont::Bold));
                    layerTitles[layerIndex] = layerTitle;
                }
            }
            layerIndex++;
        }
    }

    // 绘制连接
    updateWeights();
}

void NetworkVisualization::updateWeights()
{
    // 清除旧的边和权重标签
    foreach (const auto &edge, edgeItems) {
        scene->removeItem(edge);
        delete edge;
    }
   foreach(const auto &label , weightLabels) {
        scene->removeItem(label);
        delete label;
    }
    edgeItems.clear();
    weightLabels.clear();

    auto nodes = network->getAllNodes();
    for (const auto &node : nodes) {
        if (node->getNodeType() == NodeType::INPUT) continue;

        auto inputNodes = node->getInputNodes();
        Matrix weights = node->getWeightMatrix();

        for (qsizetype i = 0; i < inputNodes.size(); ++i) {
            int fromId = inputNodes[i];
            int toId = node->getId();

            // 获取权重值
            double weight = 0.0;
            if (static_cast<size_t>(i) < weights.getCols() && weights.getRows() > 0) {
                weight = weights(0, i); // 简化显示第一个输出神经元的权重
            }

            drawEdge(fromId, toId, weight);
        }
    }
}


void NetworkVisualization::drawEdge(int fromId, int toId, double weight) {
    if (!nodePositions.contains(fromId) || !nodePositions.contains(toId)) return;

    QPointF fromPos = nodePositions[fromId];
    QPointF toPos = nodePositions[toId];

    // 1. 绘制边
    QGraphicsLineItem* edge = scene->addLine(fromPos.x(), fromPos.y(), toPos.x(), toPos.y());
    double thickness = 2.5;
    QColor color = weight < 0 ? Qt::black : Qt::red;
    edge->setPen(QPen(color, thickness));
    edge->setZValue(0);
    edgeItems.append(edge);

    // 2. 计算边的几何属性
    double dx = toPos.x() - fromPos.x();
    double dy = toPos.y() - fromPos.y();
    double length = qSqrt(dx*dx + dy*dy);
    if (length < 1e-6) { dx=1; dy=0; length=1; } // 处理极小边

    QPointF midPoint = (fromPos + toPos) / 2; // 边中点

    // 3. 生成两个垂直方向（向上优先）
    QPointF normalUp(-dy, dx);   // 逆时针旋转90度
    QPointF normalDown(dy, -dx); // 顺时针旋转90度
    normalUp /= length;          // 归一化
    normalDown /= length;

    // 4. 动态偏移（基础+随机+长度适配）
    double baseOffset = 25;
    double lengthOffset = qMin(length / 5, 30.0); // 短边缩小偏移

    double finalOffset = baseOffset + lengthOffset ;

    // 5. 计算两个候选位置，选更靠上的（y 更小）
    QPointF posUp = midPoint + normalUp * finalOffset;
    QPointF posDown = midPoint + normalDown * finalOffset;
    QPointF labelPos = (posUp.y() < posDown.y()) ? posUp : posDown;

    // 6. 权重标签样式
    QGraphicsTextItem* weightLabel = scene->addText(QString::number(weight, 'f', 2));
    weightLabel->setPos(labelPos.x() - weightLabel->boundingRect().width()/2,
                       labelPos.y() - weightLabel->boundingRect().height()/2+40);
    weightLabel->setDefaultTextColor(weight < 0 ? Qt::black : Qt::red);
    weightLabel->setFont(QFont("Consolas", 9, QFont::Bold));
    weightLabel->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
    weightLabel->setZValue(2);
    weightLabels.append(weightLabel);
}
QPointF NetworkVisualization::calculateNodePosition(int nodeId, int layerIndex, int nodeInLayer, int totalInLayer, int maxNodes)
{
    double x = 100 + layerIndex * 250;

    // 垂直间距计算，考虑最大节点数以确保布局美观
    double verticalSpacing = 100.0;
    double y = 100 + nodeInLayer * verticalSpacing;

    return QPointF(x, y);
}

void NetworkVisualization::updateNodeValues(const QMap<int, Matrix>& nodeValueMap)
{
    // 重置所有流动点
    foreach (const auto &point , flowPoints) {
        point->setVisible(false);
    }

    // 更新节点值显示
    for (auto it = nodeValueMap.begin(); it != nodeValueMap.end(); ++it) {
        int nodeId = it.key();
        const Matrix& value = it.value();

        if (nodeValues.contains(nodeId)) {
            double val = value.getRows() > 0 ? value(0, 0) : 0.0;
            nodeValues[nodeId]->setPlainText(QString::number(val, 'f', 3));
        }

        updateNodeAppearance(nodeId, value);
    }
}

void NetworkVisualization::updateNodeAppearance(int nodeId, const Matrix& value)
{
    if (!nodeItems.contains(nodeId)) return;

    double val = value.getRows() > 0 ? value(0, 0) : 0.0;
}

void NetworkVisualization::clearVisualization()
{
    scene->clear();
    nodeItems.clear();
    nodeLabels.clear();
    nodeValues.clear();
    edgeItems.clear();
    weightLabels.clear();
    flowPoints.clear();
    flowPointEdges.clear();
    layerPositions.clear();
    layerTitles.clear();
    nodePositions.clear();
}

// NeuralNetworkWidget 实现
NeuralNetworkWidget::NeuralNetworkWidget(QWidget* parent)
    : QWidget(parent)
{
    setupUI();
    setupChart();
    setupConnections();
     updateUI();
    // 创建默认网络组件
    graph = std::make_shared<MLPGraphImpl>();
    network = std::make_shared<MLPNetworkImpl>();
    // 为了结果可重复，固定随机种子
    std::srand(42);

    // 生成示例数据
generateSampleData();
}

void NeuralNetworkWidget::setupChart()
{
    lossChart = new QChart();
    lossChart->setTitle("训练损失曲线");
    lossChart->setAnimationOptions(QChart::SeriesAnimations);

    lossSeries = new QLineSeries();
    lossSeries->setName("损失");
    lossChart->addSeries(lossSeries);

    axisX = new QValueAxis();
    axisX->setTitleText("轮数");
    axisX->setLabelFormat("%d");
    lossChart->addAxis(axisX, Qt::AlignBottom);
    lossSeries->attachAxis(axisX);

    axisY = new QValueAxis();
    axisY->setTitleText("损失值");
    axisY->setLabelFormat("%.4f");
    lossChart->addAxis(axisY, Qt::AlignLeft);
    lossSeries->attachAxis(axisY);

    lossChartView->setChart(lossChart);
}
void NeuralNetworkWidget::setupUI()
{
    QHBoxLayout* mainLayout = new QHBoxLayout(this);

    // 左侧控制面板
    QWidget* controlPanel = new QWidget();
    controlPanel->setMaximumWidth(300);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);

    // 网络配置组
    QGroupBox* networkGroup = new QGroupBox("网络配置");
    QGridLayout* networkLayout = new QGridLayout(networkGroup);

    networkLayout->addWidget(new QLabel("输入层神经元:"), 0, 0);
    inputLayerSpin = new QSpinBox();
    inputLayerSpin->setRange(1, 100);
    inputLayerSpin->setValue(2);
    networkLayout->addWidget(inputLayerSpin, 0, 1);

    networkLayout->addWidget(new QLabel("隐藏层数:"), 1, 0);
    hiddenLayersSpin = new QSpinBox();
    hiddenLayersSpin->setRange(1, 10);
    hiddenLayersSpin->setValue(1);
    networkLayout->addWidget(hiddenLayersSpin, 1, 1);

    networkLayout->addWidget(new QLabel("隐藏层神经元:"), 2, 0);
    hiddenNeuronsSpin = new QSpinBox();
    hiddenNeuronsSpin->setRange(1, 100);
    hiddenNeuronsSpin->setValue(4);
    networkLayout->addWidget(hiddenNeuronsSpin, 2, 1);

    networkLayout->addWidget(new QLabel("输出层神经元:"), 3, 0);
    outputLayerSpin = new QSpinBox();
    outputLayerSpin->setRange(1, 100);
    outputLayerSpin->setValue(1);
    networkLayout->addWidget(outputLayerSpin, 3, 1);

    networkLayout->addWidget(new QLabel("激活函数:"), 4, 0);
    activationCombo = new QComboBox();
    activationCombo->addItems({"SIGMOID", "RELU", "TANH", "LINEAR"});
    networkLayout->addWidget(activationCombo, 4, 1);

    controlLayout->addWidget(networkGroup);

    // 训练配置组
    QGroupBox* trainingGroup = new QGroupBox("训练配置");
    QGridLayout* trainingLayout = new QGridLayout(trainingGroup);

    trainingLayout->addWidget(new QLabel("学习率:"), 0, 0);
    learningRateSpin = new QDoubleSpinBox();
    learningRateSpin->setRange(0.001, 1.0);
    learningRateSpin->setValue(0.1);
    learningRateSpin->setDecimals(3);
    trainingLayout->addWidget(learningRateSpin, 0, 1);

    trainingLayout->addWidget(new QLabel("训练轮数:"), 1, 0);
    epochsSpin = new QSpinBox();
    epochsSpin->setRange(1, 50000);
    epochsSpin->setValue(1000);
    trainingLayout->addWidget(epochsSpin, 1, 1);

    controlLayout->addWidget(trainingGroup);

    // 控制按钮
    createNetworkBtn = new QPushButton("创建网络");
    loadDataBtn = new QPushButton("加载数据");
    startTrainingBtn = new QPushButton("开始训练");
    stopTrainingBtn = new QPushButton("停止训练");
    testNetworkBtn = new QPushButton("测试网络");

    controlLayout->addWidget(createNetworkBtn);
    controlLayout->addWidget(loadDataBtn);
    controlLayout->addWidget(startTrainingBtn);
    controlLayout->addWidget(stopTrainingBtn);
    controlLayout->addWidget(testNetworkBtn);

    // 进度条
    progressBar = new QProgressBar();
    controlLayout->addWidget(progressBar);

    // 日志显示
    logTextEdit = new QTextEdit();
    controlLayout->addWidget(logTextEdit);

    controlLayout->addStretch();

    // 右侧显示区域
    QSplitter* rightSplitter = new QSplitter(Qt::Vertical);

   QSplitter* rightSplitter2 = new QSplitter(Qt::Vertical);
    // 网络可视化
    networkView = new NetworkVisualization();
    rightSplitter->addWidget(networkView);

    rightSplitter2->setStretchFactor(0, 2);
    rightSplitter2->setStretchFactor(1, 1);
    // 损失曲线
    lossChartView = new QChartView();

    rightSplitter2->addWidget(lossChartView);


    mainLayout->addWidget(controlPanel);
    mainLayout->addWidget(rightSplitter);

    prediction = new QTextEdit();
    rightSplitter2->addWidget(prediction);
    mainLayout->addWidget(rightSplitter2);
    updateUI();
}
void NeuralNetworkWidget::setupConnections()
{
    connect(createNetworkBtn, &QPushButton::clicked, this, &NeuralNetworkWidget::createNetwork);
    connect(loadDataBtn, &QPushButton::clicked, this, &NeuralNetworkWidget::loadTrainingData);
    connect(startTrainingBtn, &QPushButton::clicked, this, &NeuralNetworkWidget::startTraining);
    connect(stopTrainingBtn, &QPushButton::clicked, this, &NeuralNetworkWidget::toggleTraining);
    connect(testNetworkBtn, &QPushButton::clicked, this, &NeuralNetworkWidget::testNetwork);
}


void NeuralNetworkWidget::createNetwork()
{
    try {
        // 清空现有网络
        graph->clear();

        int inputSize = inputLayerSpin->value();
        int hiddenLayers = hiddenLayersSpin->value();
        int hiddenSize = hiddenNeuronsSpin->value();
        int outputSize = outputLayerSpin->value();

        // 创建输入层节点
        QList<int> lastLayerNodeIds;
        for (int i = 0; i < inputSize; ++i) {
            auto inputNode = std::make_shared<GraphNodeImpl>(
                graph->getNextNodeId(), NodeType::INPUT, 0, 1);
            graph->addNode(inputNode);
            inputNode->setLayerId(0);
            lastLayerNodeIds.append(inputNode->getId());
        }

        // 创建隐藏层节点
        for (int layer = 0; layer < hiddenLayers; ++layer) {
            QList<int> currentLayerNodeIds;
            for (int i = 0; i < hiddenSize; ++i) {
                auto hiddenNode = std::make_shared<GraphNodeImpl>(
                    graph->getNextNodeId(), NodeType::HIDDEN,
                    layer == 0 ? inputSize : hiddenSize, 1
                  );
                hiddenNode->setLayerId(layer+1);
                graph->addNode(hiddenNode);
                currentLayerNodeIds.append(hiddenNode->getId());

                // 连接到前一层的所有节点
                for (int lastId : lastLayerNodeIds) {
                    graph->addEdge(lastId, hiddenNode->getId());
                }
            }
            lastLayerNodeIds = currentLayerNodeIds;
        }

        // 创建输出层节点
        for (int i = 0; i < outputSize; ++i) {
            auto outputNode = std::make_shared<GraphNodeImpl>(
                graph->getNextNodeId(), NodeType::OUTPUT, hiddenSize, 1
               );
            outputNode->setLayerId(hiddenLayers+1);
            graph->addNode(outputNode);

            // 连接到最后一个隐藏层的所有节点
            for (int lastId : lastLayerNodeIds) {
                graph->addEdge(lastId, outputNode->getId());
            }
        }

        // 验证并调整维度
        graph->validateAndAdjustDimensions();

        // 设置网络
        network->setGraph(graph);

        // 设置激活函数
        QString activation = activationCombo->currentText();
        ActivationType type = ActivationType::SIGMOID;
        if (activation == "RELU") type = ActivationType::RELU;
        else if (activation == "TANH") type = ActivationType::TANH;
        else if (activation == "LINEAR") type = ActivationType::LINEAR;

        network->setActivationType(type);

        // 更新可视化
        networkView->setNetwork(graph);

        logMessage(QString("网络创建成功: %1输入 -> %2隐藏层(%3神经元) -> %4输出")
                   .arg(inputSize).arg(hiddenLayers).arg(hiddenSize).arg(outputSize));

        updateUI();

    } catch (const std::exception& e) {
        QMessageBox::warning(this, "错误", QString("创建网络失败: %1").arg(e.what()));
    }
}

void NeuralNetworkWidget::toggleTraining()
{
    if (isrunning) {
        // 如果训练正在运行，停止训练
        stopTraining();
        stopTrainingBtn->setText("继续训练");
    } else {
        // 如果训练已停止，继续训练
        continueTraining();
        stopTrainingBtn->setText("停止训练");
    }
updateUI();
}

void NeuralNetworkWidget::continueTraining()
{
    shouldstop=false;
    isrunning=true;
   trainingTimer->start(10);
}


void NeuralNetworkWidget::startTraining() {
    shouldstop=false;
    isrunning=true;
    completed=false;
  network->setTrainingData(trainingInputs, trainingOutputs);
    lossSeries->clear();
    epochs = epochsSpin->value();   // 显式赋值
    learningRate = learningRateSpin->value(); // 显式赋值
    progressBar->setRange(0, epochs);
    progressBar->setValue(0);
    currentEpoch = 0;
    trainingTimer = new QTimer(this);  // 初始化计时器
       currentEpoch = 0;

      connect(trainingTimer, &QTimer::timeout, [this]() mutable {
           if (currentEpoch >= this->epochs) {
               trainingTimer->stop();
               trainingTimer->deleteLater();
               isrunning=false;
               completed=true;
               onTrainingCompleted();
               return;
           }

           if(this->shouldstop) {
               trainingTimer->stop();
               logMessage("训练已停止");
               updateUI();
           }
           try {
               network->train(1, learningRateSpin->value());
               double loss = network->getLoss();

               currentEpoch++;
                   std::cout << "Epoch " << currentEpoch << ", 训练损失: " << network->getLoss() << std::endl;

               onTrainingProgress(currentEpoch, loss);
               if (!trainingInputs.empty()) {
                   network->predict(trainingInputs[1]);
                   QMap<int, Matrix> nodeVals = network->getNodeValues();
                   onNodeValuesUpdated(nodeVals);
               }
               if (network->getLoss()< 0.01) {
                   logMessage(QString("Early stopping at epoch %1 with loss: %2").arg(currentEpoch).arg(network->getLoss()));
                   trainingTimer->stop();
                   trainingTimer->deleteLater();
                   isrunning=false;
                   completed=true;
                   onTrainingCompleted();
                   return;
               }

           } catch (const std::exception& e) {
               trainingTimer->stop();
               trainingTimer->deleteLater();
               QMessageBox::critical(this, "训练错误", QString("训练过程中发生错误: %1").arg(e.what()));
           }

       });
    trainingTimer->start(30);
    logMessage("开始训练...");
 updateUI();
  stopTrainingBtn->setText("停止训练");
}

void NeuralNetworkWidget::stopTraining()
{
    shouldstop=true;
    isrunning=false;
}

void NeuralNetworkWidget::onTrainingProgress(int epoch, double loss)
{
    progressBar->setValue(epoch);

    // 更新损失曲线
    lossSeries->append(epoch, loss);

    // 自动调整Y轴范围
    if (lossSeries->count() > 1) {
        QList<QPointF> points = lossSeries->points();
        double minY = points[0].y();
        double maxY = points[0].y();

        for (const QPointF& point : points) {
            minY = qMin(minY, point.y());
            maxY = qMax(maxY, point.y());
        }

        double margin = (maxY - minY) * 0.1;
        axisY->setRange(qMax(0.0, minY - margin), maxY + margin);
        axisX->setRange(1, epochsSpin->value());
    }

    // 每10轮记录一次日志
    if (epoch % 10 == 0 || epoch == 1) {
        logMessage(QString("Epoch %1: Loss = %2").arg(epoch).arg(loss, 0, 'f', 6));
    }
}

void NeuralNetworkWidget::onTrainingCompleted()
{
    progressBar->setValue(epochsSpin->value());
    logMessage("训练完成!");
    updateUI();
}

void NeuralNetworkWidget::onNodeValuesUpdated(QMap<int, Matrix> values)
{
    networkView->updateNodeValues(values);
    networkView->updateWeights();
}

//加载csv文件
bool NeuralNetworkWidget::LoadIrisCSV(const std::string& filepath,
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

void NeuralNetworkWidget::loadTrainingData()
{
    // 为简化，这里生成XOR问题的训练数据
    QString fileName = QFileDialog::getOpenFileName(this, "选择训练数据文件", "", "CSV Files (*.csv)");

    if (fileName.isEmpty()) {
        // 如果没有选择文件，使用默认的XOR数据
        generateSampleData();
        logMessage("使用默认XOR训练数据");
    } else {
        // TODO: 实现CSV文件读取
        std::vector<Matrix> allInputs, allOutputs;
        if (!LoadIrisCSV(fileName.toStdString(), allInputs, allOutputs))
            {
            std::cerr << " 加载 iris.data 文件失败！" << std::endl;
            return;
            }
        // 2. 划分训练集和测试集（7:3比例）
        isload=true;
        const double trainRatio = 0.7;  // 训练集占比
        const int totalSamples = allInputs.size();
        const int trainSize = static_cast<int>(totalSamples * trainRatio);
trainingInputs.clear();
trainingOutputs.clear();
        // 生成随机索引并打乱（确保每个类别均匀分布）
        std::vector<int> indices(totalSamples);
        for (int i = 0; i < totalSamples; ++i) indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());

        // 分离训练集和测试集

        for (int i = 0; i < totalSamples; ++i) {
            if (i < trainSize) {
                trainingInputs.push_back(allInputs[indices[i]]);
                trainingOutputs.push_back(allOutputs[indices[i]]);
                 std::cout << "界面输入" << allInputs[indices[i]]<< "和 " << allOutputs[indices[i]]<< std::endl;
            } else {
                predictInputs.push_back(allInputs[indices[i]]);
                predictOutputs.push_back(allOutputs[indices[i]]);
            }
        }

        // 3. 对训练集进行归一化（注意：用训练集的min/max归一化测试集，避免数据泄露）
        int inputNeurons = trainingInputs[0].getRows();  // 4个特征
        std::vector<double> minVals(inputNeurons, std::numeric_limits<double>::max());
        std::vector<double> maxVals(inputNeurons, std::numeric_limits<double>::lowest());

        // 计算训练集的min和max
        for (const auto& input : trainingInputs) {
            for (int i = 0; i < inputNeurons; ++i) {
                double val = input(i, 0);
                if (val < minVals[i]) minVals[i] = val;
                if (val > maxVals[i]) maxVals[i] = val;
            }
        }

        // 归一化训练集
        for (auto& input : trainingInputs) {
            for (int i = 0; i < inputNeurons; ++i) {
                double range = maxVals[i] - minVals[i];
                input(i, 0) = (range < 1e-9) ? 0.5 : (input(i, 0) - minVals[i]) / range;
            }
        }

        // 用训练集的min/max归一化测试集（关键：避免测试集信息泄露到训练中）
        for (auto& input : predictInputs) {
            for (int i = 0; i < inputNeurons; ++i) {
                double range = maxVals[i] - minVals[i];
                input(i, 0) = (range < 1e-9) ? 0.5 : (input(i, 0) - minVals[i]) / range;
            }
        }

        // 直接输出 trainInputs 所有内容
        std::cout << "=== trainInputs 数据详情 ===" << std::endl;
        for (size_t sample = 0; sample < trainingInputs.size(); ++sample) {
            const Matrix& mat = trainingInputs[sample];
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
  logMessage("加载文件成功");
    }

    updateUI();
}

// 在需要添加文本的地方调用此函数，实现不换行添加
void appendTextWithoutNewline(QTextEdit* edit, const QString& text) {
    // 保存当前光标位置
    QTextCursor cursor = edit->textCursor();

    // 移动到文本末尾
    cursor.movePosition(QTextCursor::End);

    // 插入文本（不添加换行符）
    cursor.insertText(text);

    // 恢复光标位置
    edit->setTextCursor(cursor);
}
void NeuralNetworkWidget::testNetwork()
{
    if ( !network  || !graph->isValidNetwork()||!graph) {
        QMessageBox::warning(this, "警告", "请先创建并训练网络!");
        return;
    }

    // 使用第一个训练样本进行测试
    if (!trainingInputs.empty()) {

        auto printPredictions = [&](const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, const std::string& name) {
        prediction->append(QString("\n==== ") + QString::fromStdString(name) + QString("预测结果 ====\n"));
            for (size_t i = 0; i < inputs.size(); ++i) {  // 遍历所有样本
                Matrix pred = network->predict(inputs[i]);
                prediction->append(QString("样本 %1 预测: [ ").arg(i));
                // 打印预测值（保留4位小数）
                for (size_t j = 0; j < pred.getRows(); ++j) {
                   appendTextWithoutNewline(prediction,QString::number(pred(j, 0), 'f', 4) + " ");
                }
               appendTextWithoutNewline(prediction,"]，实际: [ ");
                // 打印实际值
                for (size_t j = 0; j < outputs[i].getRows(); ++j) {
                appendTextWithoutNewline(prediction,QString::number(outputs[i](j, 0)) + " ");
                }
              appendTextWithoutNewline (prediction,"]" + QString("\n"));
            }
        };

        // 输出训练集和测试集的预测结果（可根据需要注释其中一个）
        printPredictions(trainingInputs, trainingOutputs, "训练集");
        if(!predictInputs.empty())
        printPredictions(predictInputs, predictOutputs, "测试集");

        // 7. 评估：分别计算训练集和测试集准确率
        auto calculateAccuracy = [&](const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, const std::string& name) {
            int correct = 0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                Matrix pred = network->predict(inputs[i]);
                int predictedClass = 0;
                double maxVal = pred(0, 0);
                for (int j = 1; j < pred.getRows(); ++j) {
                    if (pred(j, 0) > maxVal) {
                        maxVal = pred(j, 0);
                        predictedClass = j;
                    }
                }
                int actualClass = 0;
                for (int j = 0; j < outputs[i].getRows(); ++j) {
                    if (outputs[i](j, 0) == 1.0) {
                        actualClass = j;
                        break;
                    }
                }
                if (predictedClass == actualClass) correct++;
            }
            double acc = static_cast<double>(correct) / inputs.size() * 100.0;
          prediction->append(QString(" %1准确率: %2%").arg(QString::fromStdString(name)).arg(acc) + QString("\n"));
            return acc;
        };

        // 输出评估结果
       if(isload)
        {
        calculateAccuracy(trainingInputs, trainingOutputs, "训练集");  // 反映模型拟合能力
        calculateAccuracy(predictInputs, predictOutputs, "测试集");  // 反映泛化能力
    }
        // 更新可视化
        QMap<int, Matrix> nodeVals = network->getNodeValues();
        networkView->updateNodeValues(nodeVals);
    }
}

void NeuralNetworkWidget::updateUI()
{
    bool hasNetwork = graph && 1;
    bool hasData = !trainingInputs.empty();


    createNetworkBtn->setEnabled(!isrunning);
    loadDataBtn->setEnabled(!isrunning);
    startTrainingBtn->setEnabled(hasNetwork && hasData && !isrunning);

    testNetworkBtn->setEnabled(hasNetwork && !isrunning);
    stopTrainingBtn->setEnabled(!completed);
    inputLayerSpin->setEnabled(!isrunning);
    hiddenLayersSpin->setEnabled(!isrunning);
    hiddenNeuronsSpin->setEnabled(!isrunning);
    outputLayerSpin->setEnabled(!isrunning);
    activationCombo->setEnabled(!isrunning);
    learningRateSpin->setEnabled(!isrunning);
    epochsSpin->setEnabled(!isrunning);
}

void NeuralNetworkWidget::logMessage(const QString& message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    logTextEdit->append(QString("[%1] %2").arg(timestamp, message));
}

void NeuralNetworkWidget::generateSampleData()
{
    // 生成XOR问题数据
    trainingInputs.clear();
    trainingOutputs.clear();

    // XOR真值表
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> outputs = {0, 1, 1, 0};

    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix input = Matrix::fromVector(inputs[i], true);
        Matrix output = Matrix::fromVector({outputs[i]}, true);

        trainingInputs.push_back(input);
        trainingOutputs.push_back(output);
    }

    logMessage(QString("生成了 %1 个XOR训练样本").arg(trainingInputs.size()));
}


