#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define PADDING 1

// CUDA核函数：直接卷积计算
__global__ void directConvolution(const float* input, const float* kernel, float* output, 
                                  int H, int W, int stride) {
    // 计算输出尺寸
    int outH = (H + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    int outW = (W + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < outH && col < outW) {
        float sum = 0.0f;
        int outIdx = row * outW + col;
        
        // 计算输入起始位置
        int h_start = row * stride - PADDING;
        int w_start = col * stride - PADDING;
        
        for (int ch = 0; ch < CHANNELS; ++ch) {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    int h = h_start + kh;
                    int w = w_start + kw;
                    
                    // 检查边界
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        int inputIdx = (ch * H + h) * W + w;
                        int kernelIdx = (ch * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        output[outIdx] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_size> <stride>\n";
        return 1;
    }
    
    int H = atoi(argv[1]);
    int W = H; // 假设为正方形输入
    int stride = atoi(argv[2]);
    
    // 计算输出尺寸
    int outH = (H + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    int outW = (W + 2 * PADDING - KERNEL_SIZE) / stride + 1;
    
    // 分配主机内存
    size_t inputSize = CHANNELS * H * W * sizeof(float);
    size_t kernelSize = CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    size_t outputSize = outH * outW * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_kernel = (float*)malloc(kernelSize);
    float *h_output = (float*)malloc(outputSize);
    
    // 初始化输入和卷积核（随机值）
    for (int i = 0; i < CHANNELS * H * W; ++i) h_input[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i) h_kernel[i] = rand() / (float)RAND_MAX;
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);
    
    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    
    // 设置CUDA核函数配置
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    
    // 创建CUDA事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 执行核函数
    cudaEventRecord(start);
    directConvolution<<<grid, block>>>(d_input, d_kernel, d_output, H, W, stride);
    cudaEventRecord(stop);
    
    // 同步并计算时间
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // 输出结果
    std::cout << "卷积计算时间: " << milliseconds << " ms\n";
    
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
    
    return 0;
}