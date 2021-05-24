#include <iostream>

__global__ void MatrixMulKernel(double *Md, double *Nd, double *Pd, long width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    double p = 0;

    for (int k = 0; k < width; k++) {
        auto m = Md[ty * width + k];
        auto n = Nd[k * width + tx];
        p += m * n;
    }

    Pd[ty * width + tx] = p;
}

int main() {
    long width = 128;
    long size = width * width * sizeof(double);
    double *M = new double[size] {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    };
    double *N = new double[size] {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    };
    double *P = new double[size];
    double *Md, *Nd, *Pd;
    dim3 dimBlock(width, width);
    dim3 dimGrid(1, 1);

    cudaMalloc(&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMalloc(&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    cudaMalloc(&Pd, size);

    MatrixMulKernel <<<dimGrid, dimBlock>>> (Md, Nd, Pd, width);
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < width; i++) {
    //     for (int j = 0; j < width; j++) {
    //         printf("%.2lf\t", P[i * width + j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}