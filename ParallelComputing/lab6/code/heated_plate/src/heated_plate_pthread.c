#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include "parallel_for.h"

#define ROWS 500
#define COLS 500

// 结构体定义
typedef struct {
    double (*w)[COLS];
} BoundaryArgs;

typedef struct {
    double *mean;
    pthread_mutex_t *mutex;
    double (*w)[COLS];
    int rows, cols;
} MeanArgs;

typedef struct {
    double (*w)[COLS];
    int cols;
    double mean;
} InitInteriorArgs;

typedef struct {
    double (*u)[COLS];
    double (*w)[COLS];
    int cols;
} SaveArgs;

typedef struct {
    double (*w)[COLS];
    double (*u)[COLS];
    int cols;
} UpdateArgs;

typedef struct {
    double (*w)[COLS];
    double (*u)[COLS];
    int rows, cols;
    pthread_mutex_t *mutex;
    double *diff;
} DiffArgs;

// 回调函数实现
void set_left(int i, void *args) { ((BoundaryArgs*)args)->w[i][0] = 100.0; }
void set_right(int i, void *args) { ((BoundaryArgs*)args)->w[i][COLS-1] = 100.0; }
void set_bottom(int j, void *args) { ((BoundaryArgs*)args)->w[ROWS-1][j] = 100.0; }
void set_top(int j, void *args) { ((BoundaryArgs*)args)->w[0][j] = 0.0; }

void sum_loop1(int i, void *args) {
    MeanArgs *data = (MeanArgs*)args;
    double contrib = data->w[i][0] + data->w[i][data->cols-1];
    pthread_mutex_lock(data->mutex);
    *(data->mean) += contrib;
    pthread_mutex_unlock(data->mutex);
}

void sum_loop2(int j, void *args) {
    MeanArgs *data = (MeanArgs*)args;
    double contrib = data->w[0][j] + data->w[data->rows-1][j];
    pthread_mutex_lock(data->mutex);
    *(data->mean) += contrib;
    pthread_mutex_unlock(data->mutex);
}

void init_interior(int i, void *args) {
    InitInteriorArgs *data = (InitInteriorArgs*)args;
    for (int j = 1; j < data->cols-1; j++) data->w[i][j] = data->mean;
}

void save_u(int i, void *args) {
    SaveArgs *data = (SaveArgs*)args;
    for (int j = 0; j < data->cols; j++) data->u[i][j] = data->w[i][j];
}

void update_w(int i, void *args) {
    UpdateArgs *data = (UpdateArgs*)args;
    for (int j = 1; j < data->cols-1; j++)
        data->w[i][j] = (data->u[i-1][j] + data->u[i+1][j] + data->u[i][j-1] + data->u[i][j+1]) / 4.0;
}

void compute_diff(int i, void *args) {
    DiffArgs *data = (DiffArgs*)args;
    double my_diff = 0.0;
    for (int j = 1; j < data->cols-1; j++) {
        double current_diff = fabs(data->w[i][j] - data->u[i][j]);
        if (current_diff > my_diff) my_diff = current_diff;
    }
    pthread_mutex_lock(data->mutex);
    if (my_diff > *(data->diff)) *(data->diff) = my_diff;
    pthread_mutex_unlock(data->mutex);
}

// 时间函数替代
double get_wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    int num_threads = 1, chunk_size = 1;
    ScheduleMode sched_mode = SCHED_STATIC;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0) sched_mode = (strcmp(argv[++i], "dynamic") == 0) ? SCHED_DYNAMIC : SCHED_STATIC;
        else if (strcmp(argv[i], "-c") == 0) chunk_size = atoi(argv[++i]);
    }

    double epsilon = 0.001, diff = epsilon, mean = 0.0, wtime;
    int iterations = 0, iterations_print = 1;
    double w[ROWS][COLS], u[ROWS][COLS];
    pthread_mutex_t mean_mutex = PTHREAD_MUTEX_INITIALIZER, diff_mutex = PTHREAD_MUTEX_INITIALIZER;

    // 初始化边界
    BoundaryArgs args = {w};
    parallel_for(1, ROWS-1, 1, set_left, &args, num_threads, sched_mode, chunk_size);
    parallel_for(1, ROWS-1, 1, set_right, &args, num_threads, sched_mode, chunk_size);
    parallel_for(0, COLS, 1, set_bottom, &args, num_threads, sched_mode, chunk_size);
    parallel_for(0, COLS, 1, set_top, &args, num_threads, sched_mode, chunk_size);

    // 计算平均值
    MeanArgs mean_args = {&mean, &mean_mutex, w, ROWS, COLS};
    parallel_for(1, ROWS-1, 1, sum_loop1, &mean_args, num_threads, sched_mode, chunk_size);
    parallel_for(0, COLS, 1, sum_loop2, &mean_args, num_threads, sched_mode, chunk_size);
    mean /= (2*ROWS + 2*COLS -4);

    // 初始化内部点
    InitInteriorArgs init_args = {w, COLS, mean};
    parallel_for(1, ROWS-1, 1, init_interior, &init_args, num_threads, sched_mode, chunk_size);

    // 主循环
    wtime = get_wtime();
    while (diff >= epsilon) {
        SaveArgs save_args = {u, w, COLS};
        parallel_for(0, ROWS, 1, save_u, &save_args, num_threads, sched_mode, chunk_size);

        UpdateArgs update_args = {w, u, COLS};
        parallel_for(1, ROWS-1, 1, update_w, &update_args, num_threads, sched_mode, chunk_size);

        diff = 0.0;
        DiffArgs diff_args = {w, u, ROWS, COLS, &diff_mutex, &diff};
        parallel_for(1, ROWS-1, 1, compute_diff, &diff_args, num_threads, sched_mode, chunk_size);

        iterations++;
        if (iterations == iterations_print) {
            printf("%8d %f\n", iterations, diff);
            iterations_print *= 2;
        }
    }
    wtime = get_wtime() - wtime;

    printf("Wallclock time: %.2fs\n", wtime);
    pthread_mutex_destroy(&mean_mutex);
    pthread_mutex_destroy(&diff_mutex);
    return 0;
}