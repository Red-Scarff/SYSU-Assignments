#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <stddef.h>  // for offsetof

#define SEED 42

// 自定义结构体，用于聚合每个进程的局部计算时间及进程号
typedef struct {
    int rank;
    double comp_time;
} PerfData;

// 初始化矩阵：随机生成[0,10)之间的浮点数
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0;
    }
}

// 部分打印矩阵：打印前2行、前2列（若矩阵尺寸小于2则全部打印）
void print_submatrix(float *matrix, int rows, int cols) {
    int r_print = (rows < 2 ? rows : 2);
    int c_print = (cols < 2 ? cols : 2);
    for (int i = 0; i < r_print; i++) {
        for (int j = 0; j < c_print; j++) {
            printf("%8.2f", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 检查参数个数：m n k
    if (argc != 4) {
        if (rank == 0)
            printf("Usage: mpirun -np <processes> %s m n k\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    
    // 检查参数合法性
    if (m <= 0 || n <= 0 || k <= 0) {
        if (rank == 0)
            printf("Error: m, n, k must be positive integers.\n");
        MPI_Finalize();
        return 1;
    }
    
    // 检查m是否能被进程数整除
    if (m % size != 0) {
        if (rank == 0)
            printf("Error: m (%d) must be divisible by the number of processes (%d).\n", m, size);
        MPI_Finalize();
        return 1;
    }
    
    int rows_per_proc = m / size;
    
    // 声明矩阵指针
    float *A = NULL;
    float *B = NULL;
    float *C = NULL;
    double global_start, global_end;
    
    // 每个进程分配B矩阵
    srand(SEED);
    B = (float*)malloc(n * k * sizeof(float));
    if (B == NULL) {
        fprintf(stderr, "Failed to allocate memory for B (rank %d)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 根进程初始化A和C
    if (rank == 0) {
        A = (float*)malloc(m * n * sizeof(float));
        C = (float*)malloc(m * k * sizeof(float));
        if (A == NULL || C == NULL) {
            fprintf(stderr, "Failed to allocate memory for A or C\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        initialize_matrix(A, m, n);
        initialize_matrix(B, n, k);
        global_start = MPI_Wtime();
    }
    
    // 广播矩阵B到所有进程
    MPI_Bcast(B, n * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // 分发A的行到各进程
    float *local_A = (float*)malloc(rows_per_proc * n * sizeof(float));
    if (local_A == NULL) {
        fprintf(stderr, "Failed to allocate local_A (rank %d)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Scatter(A, rows_per_proc * n, MPI_FLOAT,
                local_A, rows_per_proc * n, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // 各进程计算局部结果
    float *local_C = (float*)malloc(rows_per_proc * k * sizeof(float));
    if (local_C == NULL) {
        fprintf(stderr, "Failed to allocate local_C (rank %d)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double local_start = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0;
            for (int l = 0; l < n; l++) {
                sum += local_A[i * n + l] * B[l * k + j];
            }
            local_C[i * k + j] = sum;
        }
    }
    double local_end = MPI_Wtime();
    double local_comp_time = local_end - local_start;
    
    // 收集各局部结果到根进程
    MPI_Gather(local_C, rows_per_proc * k, MPI_FLOAT,
               C, rows_per_proc * k, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    
    // 创建自定义MPI数据类型收集性能数据
    PerfData my_perf;
    my_perf.rank = rank;
    my_perf.comp_time = local_comp_time;
    
    MPI_Datatype mpi_perf_type;
    int block_lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    
    displacements[0] = offsetof(PerfData, rank);
    displacements[1] = offsetof(PerfData, comp_time);
    
    MPI_Type_create_struct(2, block_lengths, displacements, types, &mpi_perf_type);
    MPI_Type_commit(&mpi_perf_type);
    
    PerfData *all_perf = NULL;
    if (rank == 0) {
        all_perf = (PerfData*)malloc(size * sizeof(PerfData));
        if (all_perf == NULL) {
            fprintf(stderr, "Failed to allocate all_perf\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Gather(&my_perf, 1, mpi_perf_type,
               all_perf, 1, mpi_perf_type,
               0, MPI_COMM_WORLD);
    
    // 根进程输出结果和性能数据
    if (rank == 0) {
        global_end = MPI_Wtime();
        
        printf("Matrix A (partial):\n");
        print_submatrix(A, m, n);
        
        printf("\nMatrix B (partial):\n");
        print_submatrix(B, n, k);
        
        printf("\nMatrix C (partial):\n");
        print_submatrix(C, m, k);
        
        printf("\nGlobal time elapsed: %.6f seconds\n", global_end - global_start);
        printf("\nPer-process performance:\n");
        for (int i = 0; i < size; i++) {
            printf("Rank %d: Local computation time: %.6f seconds\n", all_perf[i].rank, all_perf[i].comp_time);
        }
        
        free(A);
        free(C);
        free(all_perf);
    }
    
    // 释放资源
    free(local_A);
    free(local_C);
    free(B);
    MPI_Type_free(&mpi_perf_type);
    
    MPI_Finalize();
    return 0;
}