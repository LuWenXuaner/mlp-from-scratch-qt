# MLP Neural Network From Scratch (Qt GUI)

![C++](https://img.shields.io/badge/Language-C++11-00599C?logo=c%2B%2B) ![Qt](https://img.shields.io/badge/Framework-Qt-41CD52?logo=qt) ![ML](https://img.shields.io/badge/Topic-Machine%20Learning%20%26%20Data%20Structure-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project is a **complete Multi-Layer Perceptron (MLP) neural network implemented entirely from scratch** using C++ and Qt framework, completed as a course project for Data Structure Comprehensive Training. No third-party deep learning or linear algebra libraries (such as TensorFlow, PyTorch, Eigen) are used in the core computing part.

The project uses a graph structure to represent the MLP network, implements forward propagation and backpropagation with the help of topological sorting (Kahn algorithm), and includes a full set of modules: custom matrix operation library, activation function implementation, network structure management, data preprocessing, model training & validation, and Qt interactive GUI. The model is validated on the classic Iris flower classification dataset, achieving an accuracy of over 97% on both training and test sets.

This project fully demonstrates proficiency in C++ object-oriented programming, data structure design, underlying mathematical principles of neural networks, and Qt GUI development, making it a high-value portfolio project for technical resumes.

## Key Features

### Core Computing Module

- **Custom Matrix Library**: Implemented a complete matrix class from scratch, including matrix creation, access, addition, subtraction, multiplication, transposition, random initialization, reshaping, Hadamard product, dot product, and other basic operations, which is the computing foundation of the neural network.
- **Activation Functions**: Implemented 5 common activation functions and their derivatives: Sigmoid, ReLU, Tanh, Linear, Softmax, supporting both scalar and matrix calculations.
- **Complete Training Pipeline**: Implemented the full training loop: forward propagation -> loss calculation -> backpropagation -> weight update, with configurable epochs, learning rate, and network structure.

### Network Structure Design

- **Graph-based MLP Representation**: Used directed graph to represent the neural network, with neurons as graph nodes and connections between layers as edges.
- **Topological Sorting**: Implemented Kahn algorithm to get the topological order of nodes, which ensures the correct execution order of forward and backward propagation.
- **Cycle Detection & Dimension Validation**: Built-in cycle detection for the network graph, and automatic dimension adjustment for nodes to ensure the validity of the network structure.
- **Modular Node Design**: Designed a complete graph node interface and implementation, including node attribute management, weight & bias matrix management, Xavier initialization, and connection relationship maintenance.

### Data Processing & Model Validation

- **Dataset Support**: Implemented CSV file loading, data preprocessing, training/test set splitting, and data normalization functions.
- Built-in Test Cases:
  - Iris flower classification dataset (150 samples, 4 features, 3 categories)
  - Linear inseparable problems (XOR, AND, XNOR logic gates)
  - Manual gradient calculation comparison test for simple MLP, to verify the correctness of backpropagation implementation
- **Training Metrics**: Real-time calculation of loss value, training accuracy, and test accuracy, with early stopping mechanism.

### Interactive GUI

- Qt-based visual interface for real-time parameter adjustment and training process monitoring.
- Visualization of network structure, training loss curve, and prediction results.

## Tech Stack

- **Core Language**: C++11
- **GUI Framework**: Qt 5/6 (Qt Widgets)
- **Core Algorithms**: Kahn Topological Sorting, Forward Propagation, Backpropagation (Chain Rule), Gradient Descent Optimizer
- **Core Concepts**: Object-Oriented Programming (OOP), Directed Graph, Matrix Operations, Nonlinear Transformation, Loss Function Optimization
- **Data Structures Used**: Adjacency List, 2D Matrix, Dynamic Array (std::vector), Smart Pointer
- **Validation Dataset**: UCI Iris Flower Classification Dataset

## Project Structure

*Fully consistent with the actual project file structure*

plaintext

```
mlp-from-scratch-qt/
├── src/                          # All C++ source code implementation (.cpp)
│   ├── activationfunction.cpp    # Activation functions and derivatives implementation
│   ├── graphnode.cpp             # Graph node base class and implementation
│   ├── main.cpp                  # Program entry
│   ├── mainwindow.cpp            # Qt main window logic
│   ├── matrix.cpp                # Custom matrix operation library implementation
│   ├── mlpgraph.cpp              # MLP graph structure management
│   ├── mlpnetwork.cpp            # MLP network core training & inference logic
│   ├── modeltest.cpp             # Model test and validation code
│   └── neuralnetworkwidget.cpp   # Qt neural network visualization widget
├── include/                      # All header files (.h)
│   ├── activationfunction.h
│   ├── graphnode.h
│   ├── mainwindow.h
│   ├── matrix.h
│   ├── mlpgraph.h
│   ├── mlpnetwork.h
│   ├── modeltest.h
│   └── neuralnetworkwidget.h
├── ui/                           # Qt Designer UI files
│   └── mainwindow.ui
├── data/                         # Dataset files
│   ├── iris.csv
│   └── iris.data
├── MLP.pro                       # Qt QMake project configuration file
├── .gitignore                    # Git ignore file for build artifacts
└── README.md                     # Project documentation
```

## Core Implementation Details

### 1. Matrix Class

The core computing base of the entire project, implements more than 40 member functions, including:

- Basic matrix construction, copy, access, and modification
- Matrix arithmetic operations: addition, subtraction, scalar multiplication, matrix multiplication
- Matrix operations: transposition, reshaping, Hadamard product, sum, mean, L2 norm
- Matrix initialization: zero matrix, one matrix, identity matrix, random matrix, Xavier initialization
- Vector conversion and dot product calculation

### 2. Graph Node & MLP Graph

- **GraphNode Class**: Abstract base class for neurons, defines the general interface of nodes, including node ID, type, input/output node management, weight & bias matrix management, and dimension setting.
- **MLPGraph Class**: Manages the entire neural network graph structure, implements node/edge addition and deletion, Kahn algorithm for topological sorting, cycle detection, and node dimension automatic validation & adjustment.

### 3. Activation Function

Implements 5 common activation functions and their derivatives, supports both scalar and matrix input:

- Sigmoid & its derivative
- Tanh & its derivative
- ReLU & its derivative
- Linear activation
- Softmax & its derivative (for multi-classification output layer)

### 4. MLP Network Core

- **Forward Propagation**: Calculates the output value of each node in topological order, completes linear transformation and nonlinear activation.
- **Backpropagation**: Calculates the gradient of each weight and bias to the loss function using the chain rule, in reverse topological order.
- **Weight Update**: Updates weights and biases using gradient descent algorithm with configurable learning rate.
- **Loss Calculation**: Implements mean squared error (MSE) loss function for regression and classification tasks.

### 5. Model Test & Validation

- **Iris Dataset Test**: Built a 4-40-3 network structure (4 input neurons, 40 hidden neurons, 3 output neurons), 7:3 train-test split, 30000 epochs, learning rate 0.1. Achieved **97.14% accuracy on training set** and **97.78% accuracy on test set**.
- **Gradient Validation**: Built a simple 1-2-1 MLP, compared the code running results with manual calculation results, the error is within the acceptable range, verifying the correctness of the backpropagation implementation.
- **Ablation Experiment**: Tested the impact of different hidden layer neuron numbers and network depths on model performance, analyzed the gradient vanishing problem in deep networks.

## How to Build & Run

### Prerequisites

- Qt 5.15 / Qt 6.0 or above (with Qt Widgets component installed)
- C++ compiler supporting C++11 (MSVC, GCC, MinGW)
- Qt Creator (recommended for build and run)

### Build Steps

1. Clone or download this repository
2. Open `MLP.pro` in Qt Creator
3. Configure the project with a compatible Qt kit
4. Click the **Run** button to build and launch the program directly

### Usage

- For Iris dataset classification test: call `TestCsv()` function in the main function
- For simple logic gate (XOR/AND) test: call `SimpleDataTest()` function
- For gradient correctness validation: call `ComparisonTest()` function
- You can adjust network structure, training epochs, learning rate, activation function type and other parameters in the code

## Project Highlights

1. Designed and implemented a high-performance matrix operation library from scratch using C++, without relying on any third-party linear algebra libraries, which is the core computing foundation of the neural network.
2. Built a complete MLP neural network based on directed graph structure, implemented Kahn algorithm for topological sorting to ensure the correct execution order of forward and backward propagation.
3. Manually derived and implemented the backpropagation algorithm based on the chain rule, completed the full training pipeline of the neural network, and verified the correctness of the algorithm through manual calculation and actual dataset training.
4. Implemented a complete data processing module, including CSV dataset loading, preprocessing, train-test split, and normalization, validated the model on the classic Iris dataset, achieved an accuracy of over 97%.
5. Strictly followed object-oriented design principles, with highly modular code structure, clear hierarchy, strong scalability and maintainability.
6. Developed an interactive Qt GUI for real-time parameter adjustment and training process visualization, improving the usability and demonstration effect of the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
