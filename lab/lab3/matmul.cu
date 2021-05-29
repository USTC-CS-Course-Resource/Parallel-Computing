#include <iostream>
#include <cassert>
#include <chrono>

using namespace std;

// M x K and K x N

constexpr long M = 128;
constexpr long K = 128;
constexpr long N = 128;

void MatmulOnCPU(double* mat1, double* mat2, double* result) { 
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j){
        double sum = 0;
        for (int k = 0; k < K; ++k){
            sum += mat1[i * K + k] * mat2[k * N + j];
        }
        result[i * N + j] = sum;
    }
}

__global__ void MatmulKernel(double* mat1, double* mat2, double* result);

void MatmulOnGPU(double* mat1, double* mat2, double* result) {
    double *mat1_cuda, *mat2_cuda, *result_cuda;
    dim3 dimBlock(1, 1);
    dim3 dimGrid(M, N);

    cudaMalloc(&mat1_cuda, M * K * sizeof(double));
    cudaMemcpy(mat1_cuda, mat1, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&mat2_cuda, K * N * sizeof(double));
    cudaMemcpy(mat2_cuda, mat2, K * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&result_cuda, M * N * sizeof(double));

    MatmulKernel <<<dimGrid, dimBlock>>> (mat1_cuda, mat2_cuda, result_cuda);
    
    cudaMemcpy(result, result_cuda, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(mat1_cuda);
    cudaFree(mat2_cuda);
    cudaFree(result_cuda);
}

__global__ void MatmulKernel(double* mat1_cuda, double* mat2_cuda, double* result_cuda) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    double sum = 0;

    for (int k = 0; k < K; k++) {
        sum += mat1_cuda[i * K + k] * mat2_cuda[k * N + j];
    }
    result_cuda[i * N + j] = sum;
}

int main() {
    auto mat1 = new double[M * K] {0};
    auto mat2 = new double[K * N] {0};
    for (int i = 0; i < M * K; i++) {
        mat1[i] = i;
    }
    for (int i = 0; i < K * N; i++) {
        mat2[i] = i;
    }
    auto PCPU = new double[M * N] {0};
    auto PGPU = new double[M * N] {0};

    chrono::system_clock::time_point begin, end;
    begin = chrono::system_clock::now();
    MatmulOnCPU(mat1, mat2, PCPU);
    end = chrono::system_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    
    begin = chrono::system_clock::now();
    MatmulOnGPU(mat1, mat2, PGPU);
    end = chrono::system_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    

#ifdef DEBUG
    printf("/\n");
    for (int i = 0; i < M; i++) {
        printf("|\t");
        for (int j = 0; j < K; j++) {
            printf("%.2lf\t", mat1[i * K + j]);
        }
        printf("\t|\n");
    }
    printf("\\\n");
    printf("/\n");
    for (int i = 0; i < K; i++) {
        printf("|\t");
        for (int j = 0; j < N; j++) {
            printf("%.2lf\t", mat2[i * N + j]);
        }
        printf("\t|\n");
    }
    printf("\\\n");
    printf("/\n");
    for (int i = 0; i < M; i++) {
        printf("|\t");
        for (int j = 0; j < N; j++) {
            printf("%.2lf\t", PCPU[i * N + j]);
        }
        printf("\t|\n");
    }
    printf("\\\n");
    printf("/\n");
    for (int i = 0; i < M; i++) {
        printf("|\t");
        for (int j = 0; j < N; j++) {
            printf("%.2lf\t", PGPU[i * N + j]);
        }
        printf("\t|\n");
    }
    printf("\\\n");
#endif
    bool correct = true;
    for (long i = 0; i < M * N; i++) {
        if (abs(PCPU[i] - PGPU[i]) > 1e-4) {
            correct = false;
            printf("at [%d, %d], %lf -- %lf -- %lf\n", i / N, i % N, PCPU[i], PGPU[i], PCPU[i] - PGPU[i]);
            break;
        }
    }

    printf("=====================Summary=======================\n");
    printf("mat(%dx%d) x mat(%dx%d)\n", M, K, K, N);
    if (correct) {
        printf("\033[1;32mThe result is correct!\033[0m\n");
    }
    else {
        printf("\033[1;31mThe result is wrong!\033[0m\n");
    }
    printf("cpu:\t %lld us\n", cpu_duration);
    printf("gpu:\t %lld us\n", gpu_duration);
    printf("speedup:\t %lf\n", cpu_duration / (double)gpu_duration);
    printf("===================================================\n");
}