#include <iostream>
#include <cassert>
#include <chrono>

using namespace std;

constexpr long WIDTH = 1024;
constexpr long TILE_WIDTH = 16;

void MatmulOnCPU(double* M, double* N, double* P) { 
for (int i = 0; i < WIDTH; ++i)
    for (int j = 0; j < WIDTH; ++j){
        double sum = 0;
        for (int k = 0; k < WIDTH; ++k){
            double a = M[i * WIDTH + k];
            double b = N[k * WIDTH + j];
            sum += a * b;
        }
        P[i * WIDTH + j] = sum;
    }
}

__global__ void MatrixMulKernel(double *Md, double *Nd, double *Pd);

void MatmulOnGPU(double* M, double* N, double* P) {
    constexpr long size = WIDTH * WIDTH;
    double *Md, *Nd, *Pd;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);

    cudaMalloc(&Md, size * sizeof(double));
    cudaMemcpy(Md, M, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&Nd, size * sizeof(double));
    cudaMemcpy(Nd, N, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&Pd, size * sizeof(double));

    MatrixMulKernel <<<dimGrid, dimBlock>>> (Md, Nd, Pd);
    cudaMemcpy(P, Pd, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

__global__ void MatrixMulKernel(double *Md, double *Nd, double *Pd) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    double p = 0;

    for (int m = 0; m < WIDTH / TILE_WIDTH; m++) {
        // get the start position of sub-matrix
        auto subMd = Md + by * TILE_WIDTH * WIDTH + m * TILE_WIDTH;
        auto subNd = Nd + m * TILE_WIDTH * WIDTH + bx * TILE_WIDTH;

        __shared__ double subMds[TILE_WIDTH][TILE_WIDTH];
        __shared__ double subNds[TILE_WIDTH][TILE_WIDTH];

        // each thread load an element from global memory to shared memory
        subMds[ty][tx] = subMd[ty * WIDTH + tx];
        subNds[ty][tx] = subNd[ty * WIDTH + tx];
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            p += subMds[ty][k] * subNds[k][tx];
        }

        __syncthreads();
    }

    Pd[(by * TILE_WIDTH + ty) * WIDTH + (bx * TILE_WIDTH + tx)] = p;
}

int main() {
    assert(WIDTH % TILE_WIDTH == 0);

    constexpr long size = WIDTH * WIDTH;
    double *M = new double[size];
    double *N = new double[size];
    for (int i = 0; i < size; i++) {
        M[i] = i;
        N[i] = i;
    }
    double *PCPU = new double[size];
    double *PGPU = new double[size];

    chrono::system_clock::time_point begin, end;
    begin = chrono::system_clock::now();
    MatmulOnCPU(M, N, PCPU);
    end = chrono::system_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    
    begin = chrono::system_clock::now();
    MatmulOnGPU(M, N, PGPU);
    end = chrono::system_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    

#ifdef DEBUG
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%.2lf\t", PCPU[i * WIDTH + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%.2lf\t", PGPU[i * WIDTH + j]);
        }
        printf("\n");
    }
#endif
    bool correct = true;
    for (long i = 0; i < size; i++) {
        if (abs(PCPU[i] - PGPU[i]) > 1e-4) {
            correct = false;
            printf("at i = %ld, %lf -- %lf -- %lf\n", i, PCPU[i], PGPU[i], PCPU[i] - PGPU[i]);
            // break;
        }
    }

    printf("=====================Summary=======================\n");
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