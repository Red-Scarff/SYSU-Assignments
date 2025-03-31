#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define SEED 42

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows*cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0;
    }
}

void print_submatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%8.2f", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == 0)
            printf("Usage: mpirun -np <processes> %s m n k\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    
    float *A = NULL, *B = NULL, *C = NULL;
    double start_time, end_time;
    
    if (rank == 0) {
        srand(SEED);
        A = (float*)malloc(m * n * sizeof(float));
        B = (float*)malloc(n * k * sizeof(float));
        C = (float*)malloc(m * k * sizeof(float));
        
        initialize_matrix(A, m, n);
        initialize_matrix(B, n, k);
        
        start_time = MPI_Wtime();
    }

    // 广播矩阵B给所有进程
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Send(B, n*k, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    } else {
        B = (float*)malloc(n * k * sizeof(float));
        MPI_Recv(B, n*k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 任务分配
    int rows_per_proc = m / size;
    float *local_A = (float*)malloc(rows_per_proc * n * sizeof(float));
    float *local_C = (float*)malloc(rows_per_proc * k * sizeof(float));

    // 主进程分发任务
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Send(A + p*rows_per_proc*n, rows_per_proc*n, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
        }
        memcpy(local_A, A, rows_per_proc*n*sizeof(float));
    } else {
        MPI_Recv(local_A, rows_per_proc*n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 本地计算
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0;
            for (int l = 0; l < n; l++) {
                sum += local_A[i*n + l] * B[l*k + j];
            }
            local_C[i*k + j] = sum;
        }
    }

    // 收集结果
    if (rank == 0) {
        memcpy(C, local_C, rows_per_proc*k*sizeof(float));
        for (int p = 1; p < size; p++) {
            MPI_Recv(C + p*rows_per_proc*k, rows_per_proc*k, MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        end_time = MPI_Wtime();
    } else {
        MPI_Send(local_C, rows_per_proc*k, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }

    // 输出结果
    if (rank == 0) {
        printf("Matrix A (partial):\n");
        print_submatrix(A, m, n);
        
        printf("\nMatrix B (partial):\n");
        print_submatrix(B, n, k);
        
        printf("\nMatrix C (partial):\n");
        print_submatrix(C, m, k);
        
        printf("\nTime elapsed: %.6f seconds\n", end_time - start_time);
        
        free(A);
        free(B);
        free(C);
    }
    
    free(local_A);
    free(local_C);
    if (rank != 0) free(B);
    
    MPI_Finalize();
    return 0;
}