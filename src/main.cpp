#include "mainwindow.h"
#include"neuralnetworkwidget.h"
#include <QApplication>

#include <modeltest.h>

int main(int argc, char* argv[]) {
    Q_UNUSED(argc);
    Q_UNUSED(argv);

    //设置固定随机种子以获得可重复的结果
    srand(42);

 // ModelTest* model = new ModelTest(1000, 0.1, 2, {15}, 1, ActivationType::SIGMOID);
//    //model->SimpleDataTest();
//    //model->ComparisonTest();
 //model->TestCsv();
    QApplication app(argc, argv);
    NeuralNetworkWidget window;
     window.setWindowTitle("神经网络可视化训练平台");
     window.resize(2000, 800);
     window.show();
     return app.exec();

}
