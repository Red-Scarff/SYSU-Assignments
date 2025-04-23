#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

long long *A;
long long total_sum = 0;
int num_threads;
long long n;

// 互斥锁保护全局累加
pthread_mutex_t sum_mutex;

// 线程参数
typedef struct {
    int tid;
    long long start, end;
} thread_arg_t;

void* partial_sum(void* arg) {
    thread_arg_t *t = (thread_arg_t*)arg;
    long long local = 0;
    for (long long i = t->start; i < t->end; ++i)
        local += A[i];
    pthread_mutex_lock(&sum_mutex);
    total_sum += local;
    pthread_mutex_unlock(&sum_mutex);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <n> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoll(argv[1]);
    num_threads = atoi(argv[2]);

    // 固定随机数种子
    srand(42);

    // 分配并初始化数组
    A = malloc(sizeof(long long) * n);
    for (long long i = 0; i < n; ++i)
        A[i] = rand() % 100 + 1;

    pthread_t threads[num_threads];
    thread_arg_t args[num_threads];
    pthread_mutex_init(&sum_mutex, NULL);

    long long chunk = n / num_threads;
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    for (int t = 0; t < num_threads; ++t) {
        args[t].tid   = t;
        args[t].start = t * chunk;
        args[t].end   = (t == num_threads - 1) ? n : args[t].start + chunk;
        pthread_create(&threads[t], NULL, partial_sum, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t)
        pthread_join(threads[t], NULL);

    gettimeofday(&tv_end, NULL);
    double elapsed = (tv_end.tv_sec - tv_start.tv_sec)
                   + (tv_end.tv_usec - tv_start.tv_usec) / 1e6;

    // 输出部分数组元素以供验证
    long long lim = n < 10 ? n : 10;
    printf("Array A (first %lld elements):\n", lim);
    for (long long i = 0; i < lim; ++i)
        printf("%3lld ", A[i]);
    if (n > 10) printf("...\n");
    else printf("\n");

    printf("Total sum = %lld\n", total_sum);
    printf("Summation time: %.6f seconds\n", elapsed);

    pthread_mutex_destroy(&sum_mutex);
    free(A);
    return EXIT_SUCCESS;
}
