#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>

class Matrix
{
private:
    size_t rows_;//矩阵行数
    size_t cols_;//矩阵列数
    std::vector<std::vector<double>> matrix;//二维矩阵
public:
    //无参构造函数
    Matrix();
    //给定行数、列数、初始值构造（默认是0）
    Matrix(size_t rows, size_t cols, double val = 0.0);
    //初始化列表构造{{1, 2, 3}, {4, 5, 6}} 2x3矩阵
    Matrix(std::initializer_list<std::initializer_list<double>> init);
    //向量构造矩阵
    Matrix(const std::vector<std::vector<double>>& other);
    //矩阵深拷贝
    Matrix(const Matrix& other);

    //赋值运算符重载
    Matrix& operator=(const Matrix& other);

    //获得矩阵的行数
    size_t getRows() const{return rows_;}
    //获得矩阵的列数
    size_t getCols() const{return cols_;}
    //判断矩阵是否为空
    bool isEmpty() const{return rows_ == 0 || cols_ == 0;}

    //括号访问矩阵（可读写）a(1, 2)
    double& operator()(size_t row, size_t col);
    //括号访问矩阵（只读）a(1, 2)
    const double& operator()(size_t row, size_t col) const;
    //获取指定行所有元素（可读写） a[1]
    std::vector<double>& operator[](size_t row);
    //获取指定行所有元素（只读） a[1]
    const std::vector<double>& operator[](size_t row) const;

    //两矩阵加法
    Matrix operator+(const Matrix& other) const;
    //两矩阵减法
    Matrix operator-(const Matrix& other) const;
    //两矩阵相乘 mxn nxs = mxs
    Matrix operator*(const Matrix& other) const;
    //数与矩阵相乘
    Matrix operator*(double val) const;
    //复合加法
    Matrix& operator+=(const Matrix& other);
    //复合减法
    Matrix& operator-=(const Matrix& other);
    //复合乘法（数与矩阵相乘）
    Matrix& operator*=(double val);

    //矩阵转置
    Matrix transpose() const;
    //哈达玛积（矩阵对应位置元素相乘）
    Matrix hadamard(const Matrix& other) const;
    //将矩阵中所有值设为指定值
    void fill(double val);
    //将矩阵中所有值设为min到max随机值
    void randomize(double min = -1.0, double max = 1.0);
    //Xavier初始化
    void xavierInit(size_t fanin, size_t fanout);

    //矩阵所有元素和
    double sum() const;
    //矩阵所有元素平均值
    double mean() const;
    //矩阵所有元素L2范数
    double norm() const;

    //判断矩阵是否为向量
    bool isVector() const{return rows_ == 1 || cols_ == 1;}
    //判断矩阵是否为行向量
    bool isRowVector() const{return rows_ == 1;}
    //判断矩阵是否为列向量
    bool isColVector() const{return cols_ == 1;}
    //获取向量长度
    size_t length() const;
    //两个向量点积
    double dot(const Matrix& other) const;

    //矩阵大小重塑 1x6可以变为2x3
    void reShape(size_t newRows, size_t newCols);
    //获取矩阵某一行数据
    Matrix getRow(size_t row) const;
    //获取矩阵某一列数据
    Matrix getCol(size_t col) const;
    //将矩阵某一行所有元素全部设为指定值
    void setRow(size_t row, const Matrix& other);
    //将矩阵某一列所有元素全部设为指定值
    void setCol(size_t col, const Matrix& other);

    //打印矩阵到控制台便于调试
    void printMatrix(const std::string& name = "") const;

    //生成指定大小0矩阵
    static Matrix zeros(size_t rows, size_t cols);
    //生成指定大小全1矩阵
    static Matrix ones(size_t rows, size_t cols);
    //生成指定大小单位矩阵（主对角线元素全为1）
    static Matrix identity(size_t size);
    //生成指定大小值在一定范围内的随机矩阵
    static Matrix random(size_t rows, size_t cols, double min = -1.0, double max = 1.0);
    //将一维向量转化为矩阵（true表示生成nx1列矩阵，false表示生成1xn行矩阵）
    static Matrix fromVector(const std::vector<double>& vec, bool isColumn = true);

    //矩阵转化为一维向量（按行展开）
    std::vector<double> toVector() const;
};

Matrix operator*(double val, const Matrix& other);
std::ostream& operator<<(std::ostream& os, const Matrix& other);

#endif // MATRIX_H
