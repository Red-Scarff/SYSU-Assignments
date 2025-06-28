#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define PADDING 1

// im2col转换核函数
__global__ void im2colKernel(const float* data_im, float* data_col, int H, int W,
                             int outH, int outW, int stride) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= outH * outW) return;
    
    int h_out = col_idx / outW;
    int w_out = col_idx % outW;
    int h_start = h_out * stride - PADDING;
    int w_start = w_out * stride - PADDING;
    
    for (int ch = 0; ch < CHANNELS; ++ch) {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int h_im = h_start + kh;
                int w_im = w_start + kw;
                int col_index = ((ch * KERNEL_SIZE + kh) * KERNEL_SIZE + kw) * (outH * outW) + col_idx;
                
                if (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) {
                    int im_index = (ch * H + h_im) * W + w_im;
                    data_col[col_index] = data_im[im_index];
                } else {
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}

// 矩阵乘法核函数（基于提供的GEMM代码）
__global__ void matMulShared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int t = 0; t < (K + 15) / 16; ++t) {
        int aCol = t * 16 + threadIdx.x;
        int bRow = t * 16 + threadIdx.y;
        
        if (row < M && aCol < K) As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else As[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (bRow < K && col < N) Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_size> <stride>\n";
        return 1;
    }
    
    int H = atoi(argv[1]);
    int W = H;
    int stride = atoi(argv[2]);
    
    // 计算输出尺寸
    int outH = (H + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    int outW = (W + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    
    // 分配主机内存
    size_t inputSize = CHANNELS * H * W * sizeof(float);
    size_t kernelSize = CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    size_t outputSize = outH * outW * sizeof(float);
    size_t colSize = CHANNELS * KERNEL_SIZE * KERNEL_SIZE * outH * outW * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_kernel = (float*)malloc(kernelSize);
    float *h_output = (float*)malloc(outputSize);
    
    // 初始化数据
    for (int i = 0; i < CHANNELS * H * W; ++i) h_input[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i) h_kernel[i] = rand() / (float)RAND_MAX;
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output, *d_col;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_col, colSize);
    
    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    
    // 计时开始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Step 1: im2col转换
    dim3 block_im2col(256);
    dim3 grid_im2col((outH * outW + 255) / 256);
    im2colKernel<<<grid_im2col, block_im2col>>>(d_input, d_col, H, W, outH, outW, stride);
    
    // Step 2: GEMM计算 (1x(3*3*3) * (3*3*3)x(outH*outW) = 1x(outH*outW)
    dim3 block_gemm(16, 16);
    dim3 grid_gemm((outH * outW + 15) / 16, 1);
    matMulShared<<<grid_gemm, block_gemm>>>(d_kernel, d_col, d_output, 
                                           1, outH * outW, CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    
    // 计时结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // 输出结果
    std::cout << "im2col+GEMM计算时间: " << milliseconds << " ms\n";
    
    // 验证小尺寸输出
    std::cout << "前10个输出值:\n";
    for (int i = 0; i < 10 && i < outH * outW; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";
    
    // 释放资源
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_col);
    
    return 0;
}