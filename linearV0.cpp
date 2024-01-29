#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include <random>
#include "gemm.h"

int* randomInit(int size, int minValue, int maxValue) {
    std::random_device rd;                          // 随机设备，用于产生随机种子
    std::mt19937 gen(rd());                         // 梅森旋转算法，用于生成随机数
    std::uniform_int_distribution<int> dist(minValue, maxValue);  // 均匀分布

    int* arr = new int[size];                        // 动态分配 int* 数组空间

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);                           // 生成随机整数并赋值给数组元素
    }

    return arr;
}

double* randomInit(int size, double minValue, double maxValue) {
    std::random_device rd;                                   // 随机设备，用于产生随机种子
    std::mt19937 gen(rd());                                  // 梅森旋转算法，用于生成随机数
    std::uniform_real_distribution<double> dist(minValue, maxValue);  // 均匀实数分布

    double* arr = new double[size];                          // 动态分配 double* 数组空间

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);                                   // 生成随机实数并赋值给数组元素
    }

    return arr;
}


double* add(double* arr1, double* arr2, int size) {
    double* result = new double[size];
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }
    return result;
}

double* subtract(double* arr1, double* arr2, int size) {
    double* result = new double[size];
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
    }
    return result;
}

double* num_mul(double k, double* arr2, int size) {
    double* result = new double[size];
    for (int i = 0; i < size; i++) {
        result[i] = k * arr2[i];
    }
    return result;
}

double* transpose(double* input, int rows, int cols){
    double* transposedMatrix = new double[cols * rows];   // 创建用于存储转置矩阵的一维数组

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposedMatrix[j * rows + i] = input[i * cols + j];   // 将矩阵中的元素按转置规则赋值给转置矩阵
        }
    }
    return transposedMatrix;
}

double* gemm_v1(double* A, double* B, int M, int N, int K){
    double* C   = (double*)malloc( sizeof(double) * M * N );
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    return C;
}

class LinearLayer {
private:
    double *weights, *bias;
    int _inputSize, _outputSize;

public:
    LinearLayer(int inputSize, int outputSize) {
        // 初始化权重和偏置
        weights = randomInit(outputSize * inputSize, 0.0, 10.0);
        bias = randomInit(outputSize, 0.0, 10.0);
        _inputSize = inputSize;
        _outputSize = outputSize;
    }

    double* forward(double* input) {
        // 执行前向传播
        return add(gemm_v1(weights, input, _outputSize, 1, _inputSize), bias, _outputSize);
    }

    double* backward(double* input, double* gradOutput, double learningRate) {
        // 执行后向传播，并更新权重和偏置
        double* gradInput = gemm_v1(transpose(weights, _outputSize, _inputSize), gradOutput, _inputSize, 1, _outputSize);
        weights = subtract(weights, gemm_v1(num_mul(learningRate, gradOutput, _outputSize), transpose(input, _inputSize, 1), _outputSize, _inputSize, 1), _outputSize * _inputSize);
        bias = subtract(bias, num_mul(learningRate, gradOutput, _outputSize), _outputSize);
        return gradInput;
    }
};

int main() {
    double* input = randomInit(1024, 0.0, 10.0);
    auto start = std::chrono::high_resolution_clock::now();
    LinearLayer linear(1024, 1024);

    // 前向传播
    double* output = linear.forward(input);
    // std::cout << "Forward output: " << output << std::endl;

    // 后向传播
    double* gradOutput = randomInit(1024, 0.0, 10.0);
    double* gradInput = linear.backward(input, gradOutput, 0.01);
    // std::cout << "Backward gradInput: " << gradInput << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Compution time: " << duration << " ms" << std::endl;
    delete[] input;
    delete[] output;
    delete[] gradOutput;
    delete[] gradInput;
    return 0;
}