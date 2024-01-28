
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
int main(){
    int *A, *B, *C;
    int M=1024, N=1024, K=1024;
    A   = (int*)malloc( sizeof(int) * M * K );
    B   = (int*)malloc( sizeof(int) * K * N );
    C   = (int*)malloc( sizeof(int) * M * N );

    for (int i = 0; i < M * K; i++)A[i] = 1;
    for (int i = 0; i < K * N; i++)B[i] = 1;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "代码执行时间: " << duration << " 毫秒" << std::endl;


    return 0;
}