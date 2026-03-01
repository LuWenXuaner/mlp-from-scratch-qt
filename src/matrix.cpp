#include "matrix.h"
#include <iomanip>
#include <random>

//无参构造函数
Matrix::Matrix():rows_(0), cols_(0){}

//指定大小和值构造
Matrix::Matrix(size_t rows, size_t cols, double val):
    rows_(rows),
    cols_(cols),
    matrix(rows, std::vector<double>(cols, val)){}

//列表构造
Matrix::Matrix(std::initializer_list<std::initializer_list<double> > init)
{
    rows_ = init.size();
    if(rows_ == 0) {
        cols_ = 0;
        return;
    }

    cols_ = init.begin()->size();
    matrix.reserve(rows_);

    for(const auto& row : init) {
        if(row.size() != cols_) {
            throw std::invalid_argument("传入的列表每行大小不一致！");
        }

        matrix.emplace_back(row);
    }
}

//向量构造矩阵
Matrix::Matrix(const std::vector<std::vector<double>> &other):
    rows_(other.size()), matrix(other)
{
    cols_ = (rows_ > 0) ? other.begin()->size() : 0;
}

//拷贝构造
Matrix::Matrix(const Matrix &other):
    rows_(other.rows_),
    cols_(other.cols_),
    matrix(other.matrix){}

//重载赋值运算符
Matrix& Matrix::operator=(const Matrix &other)
{
    if(this != &other) {
        matrix = other.matrix;
        rows_ = other.rows_;
        cols_ = other.cols_;
    }
    return *this;
}

//括号访问矩阵（可读写）a(1, 2)
double &Matrix::operator()(size_t row, size_t col)
{
    if(row >= rows_ || col >= cols_) {
        throw std::out_of_range("索引超出范围");
    }

    return matrix[row][col];
}

//括号访问矩阵（只读）a(1, 2)
const double &Matrix::operator()(size_t row, size_t col) const
{
    if(row >= rows_ || col >= cols_) {
        throw std::out_of_range("索引超出范围");
    }

    return matrix[row][col];
}

//获取指定行所有元素（可读写） a[1]
std::vector<double> &Matrix::operator[](size_t row)
{
    if(row >= rows_) {
        throw std::out_of_range("索引超出范围");
    }

    return matrix[row];
}

//获取指定行所有元素（只读） a[1]
const std::vector<double> &Matrix::operator[](size_t row) const
{
    if(row >= rows_) {
        throw std::out_of_range("索引超出范围");
    }

    return matrix[row];
}

//两矩阵加法
Matrix Matrix::operator+(const Matrix &other) const
{
    if(rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("两矩阵维度不同，无法相加！");
    }

    Matrix result(rows_, cols_);

    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            result(i, j) = matrix[i][j] + other(i, j);
        }
    }
    return result;
}

//两矩阵减法
Matrix Matrix::operator-(const Matrix &other) const
{
    if(rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("两矩阵维度不同，无法相减！");
    }

    Matrix result(rows_, cols_);

    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            result(i, j) = matrix[i][j] - other(i, j);
        }
    }
    return result;
}

//两矩阵相乘 mxn nxs = mxs
Matrix Matrix::operator*(const Matrix &other) const
{
    if(cols_ != other.rows_) {
        throw std::invalid_argument("两矩阵维度不匹配，无法相乘！");
    }

    Matrix result(rows_, other.cols_);

    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < other.cols_;j++) {
            result(i, j) = 0.0;
            for(size_t k = 0;k < cols_;k++) {
                result(i, j) = result(i, j) + matrix[i][k] * other(k, j);
            }
        }
    }
    return result;
}

//数与矩阵相乘
Matrix Matrix::operator*(double val) const
{
    Matrix result(rows_, cols_);

    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            result(i, j) = matrix[i][j] * val;
        }
    }
    return result;
}

//复合加法
Matrix &Matrix::operator+=(const Matrix &other)
{
    *this = *this + other;
    return *this;
}

//复合减法
Matrix &Matrix::operator-=(const Matrix &other)
{
    *this = *this - other;
    return *this;
}

//复合乘法（数与矩阵相乘）
Matrix &Matrix::operator*=(double val)
{
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            matrix[i][j] *= val;
        }
    }
    return *this;
}

//矩阵转置
Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);
    for(size_t i = 0;i < cols_;i++) {
        for(size_t j = 0;j < rows_;j++) {
            result(i, j) = matrix[j][i];
        }
    }
    return result;
}

//哈达玛积（矩阵对应位置元素相乘）
Matrix Matrix::hadamard(const Matrix &other) const
{
    if(rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("两矩阵维度不匹配，无法相乘！");
    }

    Matrix result(rows_, cols_);
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            result(i, j) = matrix[i][j] * other(i, j);
        }
    }
    return result;
}

//将矩阵中所有值设为指定值
void Matrix::fill(double val)
{
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            matrix[i][j] = val;
        }
    }
}

//将矩阵中所有值设为min到max随机值
void Matrix::randomize(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(min, max);

    // 或者为了可重复性，使用固定种子：
    // static std::mt19937 gen(12345); // 固定种子

    for(size_t i = 0; i < rows_; i++) {
        for(size_t j = 0; j < cols_; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

//Xavier初始化
void Matrix::xavierInit(size_t fanin, size_t fanout)
{
    double t = std::sqrt(6.0 / (fanin + fanout));
    randomize(-t, t);
}

//矩阵所有元素和
double Matrix::sum() const
{
    double s = 0.0;
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            s += matrix[i][j];
        }
    }
    return s;
}

//矩阵所有元素平均值
double Matrix::mean() const
{
    if(rows_ == 0 || cols_ == 0)
        return 0.0;

    double t = sum();
    return t / (rows_ * cols_);
}

//矩阵所有元素L2范数
double Matrix::norm() const
{
    double t = 0.0;
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            t += matrix[i][j] * matrix[i][j];
        }
    }
    return std::sqrt(t);
}

//获取向量长度
size_t Matrix::length() const
{
    if(!isVector()) {
        throw std::invalid_argument("矩阵不是一个向量！");
    }
    return std::max(rows_, cols_);
}

//两个向量点积
double Matrix::dot(const Matrix &other) const
{
    if(!isVector() || !other.isVector() || length() != other.length()) {
        throw std::invalid_argument("向量无法进行点积！");
    }

    double t = 0.0;
    for(size_t i = 0;i < length();i++) {
        double a = isRowVector() ? matrix[0][i] : matrix[i][0];
        double b = other.isRowVector() ? other.matrix[0][i] : other.matrix[i][0];
        t += a * b;
    }
    return t;
}

//矩阵大小重塑 1x6可以变为2x3
void Matrix::reShape(size_t newRows, size_t newCols)
{
    if(newRows * newCols != rows_ * cols_) {
        throw std::invalid_argument("变化后的矩阵元素数量必须和之前相同！");
    }

    std::vector<double> temp;

    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            temp.push_back(matrix[i][j]);
        }
    }

    rows_ = newRows;
    cols_ = newCols;
    matrix.assign(rows_, std::vector<double>(cols_));

    size_t idx = 0;
    for(size_t i = 0;i < rows_;i++) {
        for(size_t j = 0;j < cols_;j++) {
            matrix[i][j] = temp[idx++];
        }
    }
}

//获取矩阵某一行数据
Matrix Matrix::getRow(size_t row) const
{
    if(row >= rows_) {
        throw std::out_of_range("索引超出范围！");
    }

    return Matrix::fromVector(matrix[row], false);
}

//获取矩阵某一列数据
Matrix Matrix::getCol(size_t col) const
{
    if(col >= cols_) {
        throw std::out_of_range("索引超出范围！");
    }

    std::vector<double> temp;

    for(size_t i = 0;i < rows_;i++) {
        temp.push_back(matrix[i][col]);
    }

    return Matrix::fromVector(temp, true);
}

//将矩阵某一行所有元素全部设为指定值
void Matrix::setRow(size_t row, const Matrix &other)
{
    if(row >= rows_) {
        throw std::out_of_range("索引超出范围！");
    }

    if(!other.isRowVector() || other.cols_ != cols_) {
        throw std::invalid_argument("维度不匹配！");
    }

    for(size_t i = 0;i < cols_;i++) {
        matrix[row][i] = other.matrix[0][i];
    }
}

//将矩阵某一列所有元素全部设为指定值
void Matrix::setCol(size_t col, const Matrix &other)
{
    if(col >= cols_) {
        throw std::out_of_range("索引超出范围！");
    }

    if(!other.isColVector() || other.rows_ != rows_) {
        throw std::invalid_argument("维度不匹配！");
    }

    for(size_t i = 0;i < rows_;i++) {
        matrix[i][col] = other.matrix[i][0];
    }
}

//打印矩阵到控制台便于调试
void Matrix::printMatrix(const std::string &name) const
{
    if(!name.empty()) {
        std::cout << name << " (" << rows_ << "x" << cols_ << ")" << std::endl;
    }

    for(size_t i = 0;i < rows_;i++) {
        std::cout << "[";
        for(size_t j = 0;j < cols_;j++) {
            std::cout << std::setw(8) << std::setprecision(4) << matrix[i][j];
            if(j < cols_ - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

//生成指定大小0矩阵
Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 0.0);
}

//生成指定大小全1矩阵
Matrix Matrix::ones(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 1.0);
}

//生成指定大小单位矩阵（主对角线元素全为1）
Matrix Matrix::identity(size_t size)
{
    Matrix result(size, size, 0.0);
    for(size_t i = 0;i < size;i++) {
        result[i][i] = 1.0;
    }
    return result;
}

//生成指定大小值在一定范围内的随机矩阵
Matrix Matrix::random(size_t rows, size_t cols, double min, double max)
{
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

//将一维向量转化为矩阵（true表示生成nx1列矩阵，false表示生成1xn行矩阵）
Matrix Matrix::fromVector(const std::vector<double> &vec, bool isColumn)
{
    size_t n = vec.size();
    if(isColumn) {
        Matrix temp(n, 1);
        for(size_t i = 0;i < n;i++) {
            temp[i][0] = vec[i];
        }
        return temp;
    } else {
        Matrix temp(1, n);
        for(size_t i = 0;i < n;i++) {
            temp[0][i] = vec[i];
        }
        return temp;
    }
}

//一维矩阵转化为一维向量（按行展开）
std::vector<double> Matrix::toVector() const
{
    if(!isVector()) {
        throw std::invalid_argument("矩阵不是行或列向量");
    }

    std::vector<double> result;

    if(isRowVector()) {
        result = matrix[0];
    } else {
        for(size_t i = 0;i < rows_;i++) {
            result.push_back(matrix[i][0]);
        }
    }

    return result;
}

Matrix operator*(double val, const Matrix& other) {
    return other *val;
}

std::ostream& operator<<(std::ostream& os, const Matrix& other) {
    for(size_t i = 0;i < other.getRows();i++) {
        os << "[";
        for(size_t j = 0;j < other.getCols();j++) {
            os << std::setw(8) << std::setprecision(4) << other[i][j];
            if(j < other.getCols() - 1)
                os << ", ";
        }
        os << "]";
        os << std::endl;
    }
    return os;
}
