CC = gcc
CFLAGS = -O3 -I../parallel_for/include
LDFLAGS_OPENMP = -fopenmp -lm
LDFLAGS_PTHREAD = -L../parallel_for -lparallel_for -pthread -Wl,-rpath=../parallel_for -lm

# 默认编译两个版本
all: heated_plate_openmp heated_plate_pthread

# OpenMP 版本
heated_plate_openmp: src/heated_plate_openmp.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS_OPENMP)

# Pthreads 版本（依赖 libparallel_for.so）
heated_plate_pthread: src/heated_plate_pthread.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS_PTHREAD)

clean:
	rm -f heated_plate_openmp heated_plate_pthread