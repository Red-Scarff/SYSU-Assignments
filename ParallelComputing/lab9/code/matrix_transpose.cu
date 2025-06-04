#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// 任务1: CUDA Hello World
__global__ void helloWorldKernel() {
    int blockId = blockIdx.x;
    int threadId_x = threadIdx.x;
    int threadId_y = threadIdx.y;
    
    printf("Hello World from Thread (%d, %d) in Block %d!\n", 
           threadId_x, threadId_y, blockId);
}

// 任务2: 矩阵转置kernels

// 1. 全局内存版本
__global__ void transposeGlobal(float* input, float* output, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

// 2. 共享内存版本
__global__ void transposeShared(float* input, float* output, int n) {
    __shared__ float tile[32][32];
    
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    
    int col = blockIdx_x * blockDim.x + threadIdx.x;
    int row = blockIdx_y * blockDim.y + threadIdx.y;
    
    // 读取到共享内存
    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    
    __syncthreads();
    
    // 转置后的坐标
    int new_col = blockIdx_y * blockDim.y + threadIdx.x;
    int new_row = blockIdx_x * blockDim.x + threadIdx.y;
    
    if (new_row < n && new_col < n) {
        output[new_col * n + new_row] = tile[threadIdx.x][threadIdx.y];
    }
}

// 3. 优化共享内存版本 (避免bank conflicts)
__global__ void transposeSharedOptimized(float* input, float* output, int n) {
    __shared__ float tile[32][33]; // 增加一列避免bank conflicts
    
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    
    int col = blockIdx_x * blockDim.x + threadIdx.x;
    int row = blockIdx_y * blockDim.y + threadIdx.y;
    
    // 读取到共享内存
    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    
    __syncthreads();
    
    // 转置后的坐标
    int new_col = blockIdx_y * blockDim.y + threadIdx.x;
    int new_row = blockIdx_x * blockDim.x + threadIdx.y;
    
    if (new_row < n && new_col < n) {
        output[new_col * n + new_row] = tile[threadIdx.x][threadIdx.y];
    }
}

// 辅助函数
void initializeMatrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verifyTranspose(float* original, float* transposed, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(original[i * n + j] - transposed[j * n + i]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(float* matrix, int n, const char* name) {
    printf("%s:\n", name);
    int printSize = (n > 8) ? 8 : n; // 只打印前8x8部分
    for (int i = 0; i < printSize; i++) {
        for (int j = 0; j < printSize; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        if (n > 8) printf("...");
        printf("\n");
    }
    if (n > 8) printf("...\n");
    printf("\n");
}

float runTranspose(void (*kernelFunc)(float*, float*, int), 
                   float* d_input, float* d_output, int n, 
                   int blockSize, const char* kernelName) {
    
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    
    // 预热
    kernelFunc<<<gridDim, blockDim>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) { // 多次运行取平均
        kernelFunc<<<gridDim, blockDim>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 10.0f; // 返回平均时间
}

int main(int argc, char* argv[]) {
    printf("=== CUDA 矩阵转置实验 ===\n");
    
    // 任务1: Hello World
    if (argc >= 4) {
        int n = atoi(argv[1]);
        int m = atoi(argv[2]);
        int k = atoi(argv[3]);
        
        if (n >= 1 && n <= 32 && m >= 1 && m <= 32 && k >= 1 && k <= 32) {
            printf("\n=== 任务1: CUDA Hello World ===\n");
            printf("Hello World from the host!\n");
            
            dim3 blockDim(m, k);
            helloWorldKernel<<<n, blockDim>>>();
            cudaDeviceSynchronize();
        }
    }
    
    // 任务2: 矩阵转置性能测试
    printf("\n=== 任务2: 矩阵转置性能测试 ===\n");
    
    int matrixSizes[] = {512, 1024, 2048};
    int blockSizes[] = {8, 16, 32};
    
    // 先显示矩阵示例
    int n = 512;
    float* h_input = (float*)malloc(n * n * sizeof(float));
    float* h_output = (float*)malloc(n * n * sizeof(float));
    
    srand(time(NULL));
    initializeMatrix(h_input, n);
    printMatrix(h_input, n, "原始矩阵A (前8x8部分)");
    
    // 执行转置验证
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * n * sizeof(float));
    cudaMalloc(&d_output, n * n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    transposeGlobal<<<gridDim, blockDim>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = verifyTranspose(h_input, h_output, n);
    printf("转置正确性验证: %s\n", correct ? "通过" : "失败");
    printMatrix(h_output, n, "转置矩阵A^T (前8x8部分)");
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    printf("矩阵转置性能对比 (时间单位: ms)\n");
    printf("矩阵规模(N) | 线程块大小 | 全局内存版本 | 共享内存版本 | 优化共享内存版本\n");
    printf("-----------|------------|--------------|--------------|------------------\n");
    
    for (int sizeIdx = 0; sizeIdx < 3; sizeIdx++) {
        int n = matrixSizes[sizeIdx];
        
        // 分配主机内存
        float* h_input = (float*)malloc(n * n * sizeof(float));
        float* h_output = (float*)malloc(n * n * sizeof(float));
        
        // 初始化矩阵
        srand(time(NULL));
        initializeMatrix(h_input, n);
        
        // 分配设备内存
        float *d_input, *d_output;
        cudaMalloc(&d_input, n * n * sizeof(float));
        cudaMalloc(&d_output, n * n * sizeof(float));
        
        // 复制数据到设备
        cudaMemcpy(d_input, h_input, n * n * sizeof(float), cudaMemcpyHostToDevice);
        
        for (int blockIdx = 0; blockIdx < 3; blockIdx++) {
            int blockSize = blockSizes[blockIdx];
            
            printf("%-10d | %-10dx%-2d |", n, blockSize, blockSize);
            
            // 测试全局内存版本
            float time_global = runTranspose(transposeGlobal, d_input, d_output, n, blockSize, "Global");
            printf(" %-12.3f |", time_global);
            
            // 测试共享内存版本
            float time_shared = runTranspose(transposeShared, d_input, d_output, n, blockSize, "Shared");
            printf(" %-12.3f |", time_shared);
            
            // 测试优化共享内存版本
            float time_optimized = runTranspose(transposeSharedOptimized, d_input, d_output, n, blockSize, "Optimized");
            printf(" %-16.3f\n", time_optimized);
        }
        printf("-----------|------------|--------------|--------------|------------------\n");
        
        // 清理内存
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
    }
    
    return 0;
}