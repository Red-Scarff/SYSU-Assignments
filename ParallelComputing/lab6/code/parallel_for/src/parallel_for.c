#include <pthread.h>
#include <stdlib.h>
#include "parallel_for.h"

// 静态调度线程参数
struct static_thread_args {
    int start, inc, iter_start, iter_end;
    Functor functor;
    void *arg;
};

// 动态调度共享数据
struct dynamic_shared_data {
    int start, inc, total_iters, chunk_size;
    Functor functor;
    void *arg;
    pthread_mutex_t mutex;
    int current_iter;
};

// 静态调度工作函数
static void* static_thread_work(void *args) {
    struct static_thread_args *targs = (struct static_thread_args *)args;
    for (int iter = targs->iter_start; iter < targs->iter_end; iter++) {
        int idx = targs->start + iter * targs->inc;
        targs->functor(idx, targs->arg);
    }
    free(args);
    return NULL;
}

// 动态调度工作函数
static void* dynamic_thread_work(void *args) {
    struct dynamic_shared_data *shared = (struct dynamic_shared_data *)args;
    while (1) {
        pthread_mutex_lock(&shared->mutex);
        int local_iter = shared->current_iter;
        if (local_iter >= shared->total_iters) {
            pthread_mutex_unlock(&shared->mutex);
            break;
        }
        shared->current_iter += shared->chunk_size;
        pthread_mutex_unlock(&shared->mutex);

        int iter_end = local_iter + shared->chunk_size;
        if (iter_end > shared->total_iters) iter_end = shared->total_iters;

        for (int iter = local_iter; iter < iter_end; iter++) {
            int idx = shared->start + iter * shared->inc;
            shared->functor(idx, shared->arg);
        }
    }
    return NULL;
}

// 主并行函数
void parallel_for(int start, int end, int inc, Functor functor, void *arg,
                  int num_threads, ScheduleMode sched_mode, int chunk_size) {
    if (num_threads <= 0) num_threads = 1;
    if (inc == 0) return;
    if ((start >= end && inc > 0) || (start <= end && inc < 0)) return;

    int total_iters;
    if (inc > 0) total_iters = (end - start + inc - 1) / inc;
    else total_iters = (start - end - inc - 1) / (-inc);

    if (total_iters <= 0) return;
    num_threads = (num_threads > total_iters) ? total_iters : num_threads;

    if (sched_mode == SCHED_STATIC) {
        pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
        int basic_chunk = total_iters / num_threads, remaining = total_iters % num_threads;

        for (int tid = 0; tid < num_threads; tid++) {
            int iter_start = basic_chunk * tid + (tid < remaining ? tid : remaining);
            int iter_end = iter_start + basic_chunk + (tid < remaining ? 1 : 0);

            struct static_thread_args *targs = malloc(sizeof(struct static_thread_args));
            *targs = (struct static_thread_args){
                .start = start, .inc = inc, .iter_start = iter_start, .iter_end = iter_end,
                .functor = functor, .arg = arg
            };
            pthread_create(&threads[tid], NULL, static_thread_work, targs);
        }

        for (int tid = 0; tid < num_threads; tid++) pthread_join(threads[tid], NULL);
        free(threads);
    } else if (sched_mode == SCHED_DYNAMIC) {
        if (chunk_size <= 0) chunk_size = 1;
        pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
        struct dynamic_shared_data shared = {
            .start = start, .inc = inc, .total_iters = total_iters, .chunk_size = chunk_size,
            .functor = functor, .arg = arg, .mutex = PTHREAD_MUTEX_INITIALIZER, .current_iter = 0
        };

        for (int tid = 0; tid < num_threads; tid++)
            pthread_create(&threads[tid], NULL, dynamic_thread_work, &shared);
        for (int tid = 0; tid < num_threads; tid++) pthread_join(threads[tid], NULL);
        free(threads);
        pthread_mutex_destroy(&shared.mutex);
    }
}