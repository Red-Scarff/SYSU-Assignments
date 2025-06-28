#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define PADDING 1

void checkCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl;
        exit(1);
    }
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
    
    float *h_input = (float*)malloc(inputSize);
    float *h_kernel = (float*)malloc(kernelSize);
    float *h_output = (float*)malloc(outputSize);
    
    // 初始化数据
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
    
    // 创建cuDNN句柄
    cudnnHandle_t cudnn;
    checkCudnnError(cudnnCreate(&cudnn));
    
    // 创建张量描述符
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
    checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
    
    // 设置输入输出张量 (NCHW格式)
    checkCudnnError(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, CHANNELS, H, W));
    checkCudnnError(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, 1, outH, outW));
    
    // 创建卷积核描述符
    cudnnFilterDescriptor_t kernelDesc;
    checkCudnnError(cudnnCreateFilterDescriptor(&kernelDesc));
    checkCudnnError(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              1, CHANNELS, KERNEL_SIZE, KERNEL_SIZE));
    
    // 创建卷积描述符
    cudnnConvolutionDescriptor_t convDesc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCudnnError(cudnnSetConvolution2dDescriptor(convDesc, 
        PADDING, PADDING, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
    // 寻找最优卷积算法
    cudnnConvolutionFwdAlgo_t algo;
    int algoCount;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    checkCudnnError(cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, kernelDesc,
        convDesc, outputDesc, 1, &algoCount, &algoPerf));
    algo = algoPerf.algo;
    
    // 分配工作空间
    size_t workspaceSize;
    checkCudnnError(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc,
        convDesc, outputDesc, algo, &workspaceSize));
    void *d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);
    
    // 执行卷积
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    float alpha = 1.0f, beta = 0.0f;
    checkCudnnError(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input,
        kernelDesc, d_kernel, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_output));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // 输出结果
    std::cout << "cuDNN计算时间: " << milliseconds << " ms\n";
    
    // 打印部分输出
    std::cout << "前10个输出值:\n";
    for (int i = 0; i < 10 && i < outH * outW; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";
    
    // 清理资源
    cudaFree(d_workspace);
    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(kernelDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return 0;
}