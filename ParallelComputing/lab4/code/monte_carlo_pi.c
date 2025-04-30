#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

typedef struct {
    int points;
    unsigned int seed;
    int hits;
} MonteCarloArgs;

void *monte_carlo(void *arg) {
    MonteCarloArgs *args = (MonteCarloArgs *)arg;
    args->hits = 0;
    for (int i = 0; i < args->points; i++) {
        double x = (double)rand_r(&args->seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&args->seed) / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) {
            args->hits++;
        }
    }
    return NULL;
}

void run_monte_carlo(int n, int num_threads) {
    int points_per_thread = n / num_threads;
    int remainder = n % num_threads;

    pthread_t threads[num_threads];
    MonteCarloArgs args[num_threads];
    unsigned int base_seed = time(NULL);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        args[i].points = points_per_thread + (i < remainder ? 1 : 0);
        args[i].seed = base_seed + i;
        pthread_create(&threads[i], NULL, monte_carlo, &args[i]);
    }

    int total_hits = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_hits += args[i].hits;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double pi_estimate = 4.0 * total_hits / n;
    printf("[%2d threads] n = %d, pi ≈ %.6f, time = %9.6f s\n", num_threads, n, pi_estimate, elapsed);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);

    int thread_configs[] = {1, 2, 4, 8};
    for (int i = 0; i < sizeof(thread_configs)/sizeof(int); i++) {
        run_monte_carlo(n, thread_configs[i]);
    }
    return 0;
}