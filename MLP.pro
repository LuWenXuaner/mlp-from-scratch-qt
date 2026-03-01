QT       += core gui
QT += core widgets
QT += charts
T += core gui widgets
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    activationfunction.cpp \
    graphnode.cpp \
    main.cpp \
    mainwindow.cpp \
    matrix.cpp \
    mlpgraph.cpp \
    mlpnetwork.cpp \
    modeltest.cpp \
    neuralnetworkwidget.cpp

HEADERS += \
    activationfunction.h \
    graphnode.h \
    mainwindow.h \
    matrix.h \
    mlpgraph.h \
    mlpnetwork.h \
    modeltest.h \
    neuralnetworkwidget.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
