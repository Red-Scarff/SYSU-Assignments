#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

typedef void (*Functor)(int, void*);  // 修改为void返回类型

typedef enum {
    SCHED_STATIC,
    SCHED_DYNAMIC
} ScheduleMode;

void parallel_for(int start, int end, int inc, Functor functor, void *arg, 
                  int num_threads, ScheduleMode sched_mode, int chunk_size);

#endif