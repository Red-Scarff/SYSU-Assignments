CC = gcc
CFLAGS = -I../parallel_for/include -L../parallel_for -lparallel_for -pthread -Wl,-rpath=../parallel_for

matrix_mul: matrix_mul_pthread.c
	$(CC) $< -o $@ $(CFLAGS)

clean:
	rm -f matrix_mul