#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parallel_for.h"



// 矩阵乘法参数结构体
struct Args {
    double *A, *B, *C;
    int n, k;
};

// 矩阵乘法核心函数（按行计算）
void matmul_func(int i, void* args) {
    struct Args* data = (struct Args*)args;
    for (int j = 0; j < data->k; ++j) {
        double sum = 0.0;
        for (int l = 0; l < data->n; ++l) {
            sum += data->A[i * data->n + l] * data->B[l * data->k + j];
        }
        data->C[i * data->k + j] = sum;
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc < 6) {
        printf("Usage: %s <m> <n> <k> <threads> <sched_mode> [chunk_size]\n", argv[0]);
        printf("  sched_mode: 0 (static) or 1 (dynamic)\n");
        return 1;
    }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    int threads = atoi(argv[4]);
    ScheduleMode sched_mode = atoi(argv[5]) ? SCHED_DYNAMIC : SCHED_STATIC;
    int chunk_size = (argc > 6) ? atoi(argv[6]) : 1;

    // 分配矩阵内存
    struct Args args = {
        .A = malloc(m * n * sizeof(double)),
        .B = malloc(n * k * sizeof(double)),
        .C = calloc(m * k, sizeof(double)),
        .n = n,
        .k = k
    };
    if (!args.A || !args.B || !args.C) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // 初始化随机矩阵
    srand(42);
    for (int i = 0; i < m * n; ++i) args.A[i] = rand() % 100;
    for (int i = 0; i < n * k; ++i) args.B[i] = rand() % 100;

    // 执行并行矩阵乘法
    clock_t start = clock();
    parallel_for(0, m, 1, matmul_func, &args, threads, sched_mode, chunk_size);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f s\n", elapsed);

    // 释放内存
    free(args.A);
    free(args.B);
    free(args.C);
    return 0;
}