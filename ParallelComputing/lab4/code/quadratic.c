#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

// 全局变量
double a, b, c;
double delta;
double sqrt_delta;
double x1, x2;
int error = 0;

// 同步变量
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_delta = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_sqrt = PTHREAD_COND_INITIALIZER;
int delta_ready = 0;
int sqrt_ready = 0;

// 阶段标识
typedef enum {
    STAGE_DELTA,
    STAGE_SQRT,
    STAGE_ROOTS,
    NUM_STAGES
} ComputationStage;

// 线程池结构
typedef struct {
    pthread_t thread;
    ComputationStage stage;
} Worker;

// 计算判别式Δ
void *compute_delta(void *arg) {
    pthread_mutex_lock(&mutex);
    delta = b * b - 4 * a * c;
    delta_ready = 1;
    pthread_cond_broadcast(&cond_delta);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

// 计算平方根√Δ
void *compute_sqrt_delta(void *arg) {
    pthread_mutex_lock(&mutex);
    while (!delta_ready) pthread_cond_wait(&cond_delta, &mutex);
    if (delta < 0) {
        error = 1;
        pthread_cond_broadcast(&cond_sqrt);
        pthread_mutex_unlock(&mutex);
        return NULL;
    }
    sqrt_delta = sqrt(delta);
    sqrt_ready = 1;
    pthread_cond_broadcast(&cond_sqrt);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

// 并行计算根
void *compute_roots(void *arg) {
    pthread_mutex_lock(&mutex);
    while (!sqrt_ready && !error) pthread_cond_wait(&cond_sqrt, &mutex);
    if (!error) {
        x1 = (-b + sqrt_delta) / (2 * a);
        x2 = (-b - sqrt_delta) / (2 * a);
    }
    pthread_mutex_unlock(&mutex);
    return NULL;
}

// 运行指定阶段
void run_stage(ComputationStage stage, int num_workers) {
    Worker workers[num_workers];
    void *(*task)(void*) = NULL;

    switch(stage) {
        case STAGE_DELTA:
            task = compute_delta;
            break;
        case STAGE_SQRT:
            task = compute_sqrt_delta;
            break;
        case STAGE_ROOTS:
            task = compute_roots;
            break;
        default:
            return;
    }

    for (int i = 0; i < num_workers; i++) {
        pthread_create(&workers[i].thread, NULL, task, NULL);
    }
    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i].thread, NULL);
    }
}

// 优化后的测试函数
void run_test(int num_threads) {
    error = 0;
    delta_ready = 0;
    sqrt_ready = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 阶段式并行
    run_stage(STAGE_DELTA, 1);      // 第1阶段：1线程计算Δ
    run_stage(STAGE_SQRT, 1);       // 第2阶段：1线程计算√Δ
    run_stage(STAGE_ROOTS, num_threads); // 第3阶段：N线程计算根

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (error) {
        printf("[%2d threads] No real roots. Time: %9.6f s\n", num_threads, elapsed);
    } else {
        printf("[%2d threads] x1 = %7.3f, x2 = %7.3f. Time: %9.6f s\n", 
               num_threads, x1, x2, elapsed);
    }
}

int main() {
    printf("Enter coefficients a, b, c: ");
    scanf("%lf %lf %lf", &a, &b, &c);

    int thread_configs[] = {1, 2, 4, 8};
    for (int i = 0; i < sizeof(thread_configs)/sizeof(int); i++) {
        run_test(thread_configs[i]);
    }
    return 0;
}