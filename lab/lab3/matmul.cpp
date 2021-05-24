#include <iostream>
#include <cassert>
#include <chrono>
constexpr long WIDTH = 2048;
constexpr long TILE_WIDTH = 16;

void MatmulOnCPU(double* M, double* N, double* P);
void MatmulOnGPU(double* M, double* N, double* P);

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


    auto begin = std::chrono::system_clock::now();
    MatmulOnCPU(M, N, PGPU);
    auto end = std::chrono::system_clock::now();

    printf("cpu:\t %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

#ifdef DEBUG
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%.2lf\t", PCPU[i * WIDTH + j]);
        }
        printf("\n");
    }
#endif

}

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
