#!/bin/bash
echo "N,tile,partition,mode,time_ms"
for N in 512 1024 2048; do
  for TILE in 8 16 32; do
    for PART in 0 1 2; do
      for MODE in 0 1 2; do
        ./matrix_mul $N $TILE $PART $MODE
      done
    done
  done
done
