#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#define LEN 4096

std::vector<std::vector<double>> gemm(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
    int M = A.size(), K = A[0].size(), N = B[0].size();
    std::vector<std::vector<double>> C;
    for (int i = 0; i < M; i++){
        std::vector<double> C_row;
        for (int j = 0; j < N; j++){
            double cij=0;
            for (int k = 0; k < K; k++){
                cij += A[i][k] * B[k][j];
            }
            C_row.push_back(cij);
        }
        C.push_back(C_row);
    }
    return C;
}


std::vector<double> gemv(std::vector<std::vector<double>> A, std::vector<double> B){
    int M = A.size(), K = A[0].size();
    std::vector<double> C;
    for (int i = 0; i < M; i++){
        double ci = 0;
        for (int k = 0; k < K; k++){
            ci += A[i][k] * B[k];
        }
        C.push_back(ci);
    }
    return C;
}

std::vector<std::vector<double>> gevv(std::vector<double> A, std::vector<double> B){
    int M = A.size(), N = B.size();
    std::vector<std::vector<double>> C;
    for (int i = 0; i < M; i++){
        std::vector<double> C_row;
        for(int j = 0; j < N; j++){
            C_row.push_back(A[i]*B[j]);
        }
        C.push_back(C_row);
    }
    return C;
}


std::vector<double> add(std::vector<double> A, std::vector<double> B){
    int n = A.size();
    std::vector<double> C;
    for(int i=0;i<n;i++){
        C.push_back(A[i]+B[i]);
    }
    return C;
}

std::vector<double> sub(std::vector<double> A, std::vector<double> B){
    int n = A.size();
    std::vector<double> C;
    for(int i=0;i<n;i++){
        C.push_back(A[i]-B[i]);
    }
    return C;
}


std::vector<double> dv(double k, std::vector<double> A){
    int n = A.size();
    std::vector<double> C;
    for(int i=0;i<n;i++){
        C.push_back(A[i]*k);
    }
    return C;
}

std::vector<std::vector<double>> dm(double k, std::vector<std::vector<double>> A){
    int rows = A.size(), cols = A[0].size();
    std::vector<std::vector<double>> C;
    for (int i = 0; i < rows; i++) {
        std::vector<double> row_i;
        for (int j = 0; j < cols; j++) {
            row_i.push_back(A[i][j] * k);   // 将矩阵中的元素按转置规则赋值给转置矩阵
        }
        C.push_back(row_i);
    }
    return C;
}

std::vector<std::vector<double>> subMatrix(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
    int rows = A.size(), cols = A[0].size();
    std::vector<std::vector<double>> C;
    for (int i = 0; i < rows; i++) {
        std::vector<double> row_i;
        for (int j = 0; j < cols; j++) {
            row_i.push_back(A[i][j] + B[i][j]);   // 将矩阵中的元素按转置规则赋值给转置矩阵
        }
        C.push_back(row_i);
    }
    return C;
}

std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>> matrix) {
    std::vector<std::vector<double>> transposedMatrix;  // 创建用于存储转置矩阵的 double** 数组
    int rows = matrix.size(), cols = matrix[0].size();
    for (int i = 0; i < cols; i++) {
        std::vector<double> row_i;
        for (int j = 0; j < rows; j++) {
            row_i.push_back(matrix[j][i]);   // 将矩阵中的元素按转置规则赋值给转置矩阵
        }
        transposedMatrix.push_back(row_i);
    }

    return transposedMatrix;
}


class LinearLayer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

public:
    LinearLayer() {
        // 初始化权重和偏置
        for(int i=0;i<LEN;i++){
            std::vector<double> w;
            for(int j=0;j<LEN;j++){
                w.push_back(1.0);
            }
            weights.push_back(w);
        }
        for(int i=0;i<LEN;i++)bias.push_back(1.0);
    }

    std::vector<double> forward(std::vector<double> input) {
        // 执行前向传播
        return add(gemv(weights, input), bias);
    }

    std::vector<double> backward(std::vector<double> input, std::vector<double> gradOutput, double learningRate) {
        // 执行后向传播，并更新权重和偏置
        std::vector<double> gradInput = gemv(transposeMatrix(weights), gradOutput);
        weights = subMatrix(weights, dm(learningRate, gevv(gradOutput, input)));
        bias = sub(bias, dv(learningRate, gradOutput));
        return gradInput;
    }
};

int main(){
    std::vector<double> input;
    for(int i=0;i<LEN;i++)input.push_back(1.0);
    LinearLayer linear;
    auto start = std::chrono::high_resolution_clock::now();
    // 前向传播
    std::vector<double> output = linear.forward(input);
    // std::cout << "Forward output: " << output << std::endl;

    // 后向传播
    std::vector<double> gradOutput;
    for(int i=0;i<LEN;i++)gradOutput.push_back(1.0);
    std::vector<double> gradInput = linear.backward(input, gradOutput, 0.01);
    // std::cout << "Backward gradInput: " << gradInput << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Compution time: " << duration << " ms" << std::endl;


    return 0;
}

