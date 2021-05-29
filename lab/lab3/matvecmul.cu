#include <iostream>
#include <cassert>
#include <chrono>

using namespace std;

constexpr long WIDTH = 8192;
constexpr long TILE_WIDTH = 16;

void matvecmulOnCPU(double* mat, double* vec, double* P) { 
    for (int i = 0; i < WIDTH; ++i) {
        double sum = 0;
        for (int j = 0; j < WIDTH; ++j){
            sum += mat[i * WIDTH + j] * vec[j];
        }
        P[i] = sum;
    }
}

__global__ void matvecmulKernel(double *matd, double *vecd, double *Pd);

void matvecmulOnGPU(double* mat, double* vec, double* P) {
    constexpr long size = WIDTH * WIDTH;
    double *matd, *vecd, *Pd;
    dim3 dimBlock(1, TILE_WIDTH);
    dim3 dimGrid(1, WIDTH / TILE_WIDTH);

    cudaMalloc(&matd, size * sizeof(double));
    cudaMemcpy(matd, mat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&vecd, WIDTH * sizeof(double));
    cudaMemcpy(vecd, vec, WIDTH * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&Pd, WIDTH * sizeof(double));

    matvecmulKernel <<<dimGrid, dimBlock>>> (matd, vecd, Pd);
    cudaMemcpy(P, Pd, WIDTH * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(matd);
    cudaFree(vecd);
    cudaFree(Pd);
}

__global__ void matvecmulKernel(double *matd, double *vecd, double *Pd) {
    int by = blockIdx.y;
    int ty = threadIdx.y;
    
    double p = 0;

    for (int m = 0; m < WIDTH / TILE_WIDTH; m++) {
        // get the start position of sub-matrix
        auto submatd = matd + by * TILE_WIDTH * WIDTH + m * TILE_WIDTH;
        auto subvecd = vecd + m * TILE_WIDTH;

        // __shared__ double submatds[TILE_WIDTH][TILE_WIDTH];
        __shared__ double subvecds[TILE_WIDTH];

        // each thread load an element from global memory to shared memory
        subvecds[ty] = subvecd[ty];
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            // p += submatds[ty][k] * subvecds[k];
            p += submatd[ty * WIDTH + k] * subvecds[k];
        }

        __syncthreads();
    }
    
    // printf("(%d %d %d %d) to %d: %lf\n", bx, by, tx, ty, by * TILE_WIDTH + ty, p);

    Pd[by * TILE_WIDTH + ty] = p;

}

int main() {
    assert(WIDTH % TILE_WIDTH == 0);

    constexpr long size = WIDTH * WIDTH;
    double *mat = new double[size];
    double *vec = new double[WIDTH];
    for (int i = 0; i < size; i++) {
        mat[i] = i;
    }
    for (int i = 0; i < WIDTH; i++) {
        vec[i] = i;
    }
    double *PCPU = new double[WIDTH];
    double *PGPU = new double[WIDTH];

    chrono::system_clock::time_point begin, end;
    begin = chrono::system_clock::now();
    matvecmulOnCPU(mat, vec, PCPU);
    end = chrono::system_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    
    begin = chrono::system_clock::now();
    matvecmulOnGPU(mat, vec, PGPU);
    end = chrono::system_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    

#ifdef DEBUG
    for (int i = 0; i < WIDTH; i++) {
        printf("%.2lf\t", PCPU[i]);
    }
    for (int i = 0; i < WIDTH; i++) {
        printf("%.2lf\t", PGPU[i]);
    }
#endif
    bool correct = true;
    for (long i = 0; i < WIDTH; i++) {
        if (abs(PCPU[i] - PGPU[i]) > 1e-4) {
            correct = false;
            printf("at i = %ld, %lf -- %lf -- %lf\n", i, PCPU[i], PGPU[i], PCPU[i] - PGPU[i]);
            // break;
        }
    }

    printf("=====================Summary=======================\n");
    printf("mat size: %ld x %ld\n", WIDTH, WIDTH);
    printf("vec size: %ld x %ld\n", 1, WIDTH);
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