#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>

// kernels
__global__ void matMulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

template <int TILE>
__global__ void matMulShared(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

__global__ void matMulReg(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.f;
        int k = 0, limit = N / 4 * 4;
        for (; k < limit; k += 4) {
            sum += A[row*N + k+0] * B[(k+0)*N + col]
                 + A[row*N + k+1] * B[(k+1)*N + col]
                 + A[row*N + k+2] * B[(k+2)*N + col]
                 + A[row*N + k+3] * B[(k+3)*N + col];
        }
        for (; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

enum PartitionType { ROW=0, COL=1, BLOCK=2 };
enum ModeType { NAIVE=0, SHARED=1, REG=2 };

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: %s <N> <tile> <partition> <mode>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]), TILE = atoi(argv[2]);
    int part = atoi(argv[3]), mode = atoi(argv[4]);
    size_t bytes = N*(size_t)N*sizeof(float);
    float *hA = (float*)malloc(bytes), *hB=(float*)malloc(bytes), *hC=(float*)malloc(bytes);
    for (int i=0;i<N*N;i++){hA[i]=rand()/(float)RAND_MAX;hB[i]=rand()/(float)RAND_MAX;}
    float *dA,*dB,*dC;
    cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes); cudaMalloc(&dC,bytes);
    cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice);

    dim3 block((mode==SHARED)?TILE: TILE, (mode==SHARED)?TILE: TILE);
    dim3 grid;
    switch(part){ case ROW: grid=dim3((N+TILE-1)/TILE,1);break;
                 case COL: grid=dim3(1,(N+TILE-1)/TILE);break;
                 default:  grid=dim3((N+TILE-1)/TILE,(N+TILE-1)/TILE);}

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    switch(mode){
        case NAIVE: matMulNaive<<<grid,block>>>(dA,dB,dC,N); break;
        case SHARED:
            if (TILE==8) matMulShared<8><<<grid,block>>>(dA,dB,dC,N);
            else if (TILE==16) matMulShared<16><<<grid,block>>>(dA,dB,dC,N);
            else if (TILE==32) matMulShared<32><<<grid,block>>>(dA,dB,dC,N);
            break;
        case REG:   matMulReg<<<grid,block>>>(dA,dB,dC,N); break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms,start,stop);
    printf("%d,%d,%d,%d,%.3f\n", N, TILE, part, mode, ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
