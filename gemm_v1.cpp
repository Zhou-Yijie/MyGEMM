
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "gemm.h"

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


int main(){
    int *A, *B, *C;
    int M=1024, N=1024, K=1024;
    A   = (int*)malloc( sizeof(int) * M * K );
    B   = (int*)malloc( sizeof(int) * K * N );

    for (int i = 0; i < M * K; i++)A[i] = i;
    for (int i = 0; i < K * N; i++)B[i] = i;

    auto start = std::chrono::high_resolution_clock::now();
    C = gemm_v1(A, B, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Compution time: " << duration << " ms" << std::endl;
    free(A);
    free(B);
    free(C);



    return 0;
}