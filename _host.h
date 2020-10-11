#pragma once
#include <time.h>
#include "main.h"


// Run vector-sum on CPU.
//
// a: output vector (x + y)
// x: input vector 1
// y: input vector 2
// N: vector size (a, x, y)
float run_host(float* a, float* x, float* y, int N) {
  clock_t begin = clock();
  vector_sum(a, x, y, N);
  clock_t end = clock();

  float duration = (float) (end - begin) / CLOCKS_PER_SEC;
  return duration * 1000;
}
