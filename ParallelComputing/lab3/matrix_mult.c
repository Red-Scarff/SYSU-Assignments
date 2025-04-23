#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

// 全局变量
int m, n, k, num_threads;
int **A, **B, **C;

// 结构体：线程参数
typedef struct {
    int tid, row_start, row_end;
} thread_arg_t;

// 线程例程：计算指定行范围
void* multiply_rows(void* arg) {
    thread_arg_t *t = (thread_arg_t*)arg;
    for (int i = t->row_start; i < t->row_end; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i][j] = 0;
            for (int p = 0; p < n; ++p) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return NULL;
}

int** alloc_matrix(int rows, int cols) {
    int **M = malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; ++i)
        M[i] = malloc(cols * sizeof(int));
    return M;
}

void free_matrix(int **M, int rows) {
    for (int i = 0; i < rows; ++i) free(M[i]);
    free(M);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <m> <n> <k> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    num_threads = atoi(argv[4]);

    // 固定随机数种子
    srand(42);

    // 分配并初始化矩阵
    A = alloc_matrix(m, n);
    B = alloc_matrix(n, k);
    C = alloc_matrix(m, k);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        A[i][j] = rand() % 10;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < k; ++j)
        B[i][j] = rand() % 10;

    // 创建线程并计时
    pthread_t threads[num_threads];
    thread_arg_t args[num_threads];
    int rows_per_thread = m / num_threads;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int t = 0; t < num_threads; ++t) {
        args[t].tid = t;
        args[t].row_start = t * rows_per_thread;
        args[t].row_end = (t == num_threads - 1) ? m : args[t].row_start + rows_per_thread;
        pthread_create(&threads[t], NULL, multiply_rows, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t)
        pthread_join(threads[t], NULL);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Matrix multiplication time: %.6f seconds\n", elapsed);

    // 输出部分结果（矩阵前 5×5 块，若维度小于5则输出实际大小）
    int lim_r = m < 5 ? m : 5;
    int lim_c = k < 5 ? k : 5;
    printf("\nA (first %d×%d):\n", lim_r, n<5?n:5);
    for (int i = 0; i < lim_r; ++i) {
        for (int j = 0; j < (n<5?n:5); ++j)
            printf("%3d ", A[i][j]);
        if (n > 5) printf("...");
        printf("\n");
    }
    printf("\nB (first %d×%d):\n", (n<5?n:5), lim_c);
    for (int i = 0; i < (n<5?n:5); ++i) {
        for (int j = 0; j < lim_c; ++j)
            printf("%3d ", B[i][j]);
        if (k > 5) printf("...");
        printf("\n");
    }
    printf("\nC (first %d×%d):\n", lim_r, lim_c);
    for (int i = 0; i < lim_r; ++i) {
        for (int j = 0; j < lim_c; ++j)
            printf("%6d ", C[i][j]);
        if (k > 5) printf("...");
        printf("\n");
    }

    free_matrix(A, m);
    free_matrix(B, n);
    free_matrix(C, m);
    return EXIT_SUCCESS;
}
