#!/bin/bash

# 编译所有程序
echo "编译程序..."
nvcc task1_direct_conv.cu -o task1
nvcc task2_im2col_gemm.cu -o task2
nvcc task3_cudnn.cu -o task3 -lcudnn

# 测试配置
input_sizes=(32 64 128 256 512 1024 2048 4096)
strides=(1 2 3)
kernels=3

# 运行测试
for size in "${input_sizes[@]}"; do
    for stride in "${strides[@]}"; do
        echo ""
        echo "========================================"
        echo "测试配置: 输入尺寸=$size, 步长=$stride"
        echo "========================================"
        
        # 任务1
        echo "任务1 - 直接卷积:"
        ./task1 $size $stride
        
        # 任务2
        echo "任务2 - im2col+GEMM:"
        ./task2 $size $stride
        
        # 任务3
        echo "任务3 - cuDNN卷积:"
        ./task3 $size $stride
        
        echo ""
    done
done

echo "所有测试完成!"