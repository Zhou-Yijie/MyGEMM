#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

int main(){
    int *A, *B, *C;
    int M=1024, N=1024, K=1024;
    A   = (int*)malloc( sizeof(int) * M * K );
    B   = (int*)malloc( sizeof(int) * K * N );
    C   = (int*)malloc( sizeof(int) * M * N );

    for (int i = 0; i < M * K; i++)A[i] = 1;
    for (int i = 0; i < K * N; i++)B[i] = 1;


    __m256i A_m256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A));
    __m256i B_m256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
    __m256 A_m256 = _mm256_cvtepi32_ps(A_m256);
    __m256 B_m256 = _mm256_cvtepi32_ps(B_m256);

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < N; j++){
        for (int i = 0; i < M; i++){
            __m256 result = _mm256_dp_ps(A_m256, B_m256, 0xFF);
            float sum = _mm256_cvtss_f32(result);
            C[i * N + j] = int(sum);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "代码执行时间: " << duration << " 毫秒" << std::endl;


    return 0;
}