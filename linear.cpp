#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>


class LinearLayer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;

public:
    LinearLayer(int inputSize, int outputSize) {
        // 初始化权重和偏置
        weights = Eigen::MatrixXd::Random(outputSize, inputSize);
        bias = Eigen::VectorXd::Random(outputSize);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& input) {
        // 执行前向传播
        return weights * input + bias;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& input, const Eigen::VectorXd& gradOutput, double learningRate) {
        // 执行后向传播，并更新权重和偏置
        Eigen::VectorXd gradInput = weights.transpose() * gradOutput;
        weights -= learningRate * gradOutput * input.transpose();
        bias -= learningRate * gradOutput;
        return gradInput;
    }
};

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;i++){
        LinearLayer linear(1024, 1024);

        Eigen::VectorXd input = Eigen::VectorXd::Random(1024);

        // 前向传播
        Eigen::VectorXd output = linear.forward(input);
        // std::cout << "Forward output: " << output << std::endl;

        // 后向传播
        Eigen::VectorXd gradOutput = Eigen::VectorXd::Random(1024);
        Eigen::VectorXd gradInput = linear.backward(input, gradOutput, 0.01);
        // std::cout << "Backward gradInput: " << gradInput << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Compution time: " << duration << " ms" << std::endl;

    return 0;
}