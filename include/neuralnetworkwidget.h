// neuralnetworkwidget.h - 最终完整版本
#ifndef NEURALNETWORKWIDGET_H
#define NEURALNETWORKWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QSplitter>
#include <QGroupBox>
#include <QTimer>
#include <QThread>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsTextItem>
#include <QGraphicsRectItem>
#include <QGraphicsDropShadowEffect>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QParallelAnimationGroup>
#include <QEasingCurve>
#include <QMouseEvent>
#include <QPainter>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QValueAxis>
#include <QtCharts/QLineSeries>
#include <QMap>
#include <memory>

#include "mlpnetwork.h"
#include "mlpgraph.h"
#include "graphnode.h"

QT_USE_NAMESPACE

class NetworkVisualization : public QGraphicsView
{
    Q_OBJECT

public:
    explicit NetworkVisualization(QWidget* parent = nullptr);
    ~NetworkVisualization();

    void setNetwork(std::shared_ptr<MLPGraph> graph);

    void updateNodeValues(const QMap<int, Matrix>& nodeValues);

    void updateWeights();

    void setNodeLayer(int nodeId, int layer);
private:
    // 图形场景和网络
    QGraphicsScene* scene;
    std::shared_ptr<MLPGraph> network;

    // 图形元素映射
    QMap<int, QGraphicsEllipseItem*> nodeItems;      // 节点圆圈
    QMap<int, QGraphicsTextItem*> nodeLabels;        // 节点标签
    QMap<int, QGraphicsTextItem*> nodeValues;        // 节点激活值
    QList<QGraphicsLineItem*> edgeItems;             // 连接线
    QList<QGraphicsTextItem*> weightLabels;          // 权重标签
    QMap<NodeType, QPointF> layerPositions;
    // 布局和位置信息
    QMap<int, QPointF> nodePositions;                // 节点位置
    QMap<int, QGraphicsTextItem*> layerTitles;       // 层标题
    QMap<int, int> nodeLayerMap;                     // 节点ID到层索引的映射
    // 动画相关
    QList<QGraphicsEllipseItem*> flowPoints;         // 流动点
    QMap<QGraphicsEllipseItem*, QGraphicsLineItem*> flowPointEdges;

    // 私有方法
    void drawNetwork();
    void clearVisualization();
    QPointF calculateNodePosition(int nodeId, int layerIndex, int nodeInLayer,
                                 int totalInLayer, int maxNodes);
    void drawEdge(int fromId, int toId, double weight = 0.0);
    void updateNodeAppearance(int nodeId, const Matrix& value);

    QString getLayerName(NodeType type);
};

class NeuralNetworkWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NeuralNetworkWidget(QWidget* parent = nullptr);
    bool isrunning=false;
    bool shouldstop=false;
    bool completed=true;
    bool isload=false;
private slots:
    // 网络操作
    void createNetwork();
    void loadTrainingData();
    void testNetwork();
    void toggleTraining();
    // 训练控制
    void startTraining();
    void stopTraining();
    void continueTraining();
    void onTrainingProgress(int epoch, double loss);
    void onTrainingCompleted();
    void onNodeValuesUpdated(QMap<int, Matrix> values);

private:
    // === UI控件 ===
    int epochs ;
    double learningRate ;
    QTimer* trainingTimer;
    int currentEpoch;
    // 网络配置控件
    QSpinBox* inputLayerSpin;          // 输入层神经元数
    QSpinBox* hiddenLayersSpin;        // 隐藏层数
    QSpinBox* hiddenNeuronsSpin;       // 隐藏层神经元数
    QSpinBox* outputLayerSpin;         // 输出层神经元数
    QComboBox* activationCombo;        // 激活函数选择

    // 训练配置控件
    QDoubleSpinBox* learningRateSpin;  // 学习率
    QSpinBox* epochsSpin;              // 训练轮数

    // 控制按钮
    QPushButton* createNetworkBtn;     // 创建网络
    QPushButton* loadDataBtn;          // 加载数据
    QPushButton* startTrainingBtn;     // 开始训练
    QPushButton* stopTrainingBtn;      // 停止训练
    QPushButton* testNetworkBtn;       // 测试网络

    // 状态显示控件
    QProgressBar* progressBar;         // 训练进度条
    QTextEdit* logTextEdit;            // 日志显示
    QTextEdit* prediction;

    // 图表显示控件
    QChart* lossChart;                 // 损失图表
    QChartView* lossChartView;         // 图表视图
    QLineSeries* lossSeries;           // 损失数据序列
    QValueAxis* axisX;                 // X轴（轮数）
    QValueAxis* axisY;                 // Y轴（损失值）

    // 网络可视化控件
    NetworkVisualization* networkView; // 网络可视化组件

    // 神经网络相关
    std::shared_ptr<MLPGraph> graph;           // 网络拓扑结构
    std::shared_ptr<MLPNetworkImpl> network;   // 网络实现

    // 训练数据
    std::vector<Matrix> trainingInputs;        // 训练输入数据
    std::vector<Matrix> trainingOutputs;       // 训练输出数据
    std::vector<Matrix> predictInputs;        // 预测输入数据
    std::vector<Matrix> predictOutputs;       // 预测输出数据
    std::vector<QPointF> layerPositions;
    // === 私有方法 ===
    void setupUI();

    void setupChart();

    void setupConnections();

    void updateUI();

    void logMessage(const QString& message);
    bool LoadIrisCSV(const std::string& filepath,std::vector<Matrix>& inputs,std::vector<Matrix>& outputs);

    void generateSampleData();
};

#endif // NEURALNETWORKWIDGET_H
