#pragma once
#include "support.h"


// Calculate vector sum on CPU
void vector_sum(float* a, float* x, float* y, int N) {
  for (int i = 0; i < N; i++)
    a[i] = x[i] + y[i];
}


// Prints the output of a run.
void printrun(float exp, float *ans, int N, float duration) {
  float act = SUM_ARRAY(ans, N);
  printf("Execution time: %3.1f ms\n", duration);
  printf("Result %s valid!\n", exp == act? "is" : "is not");
  printf("\n");
}
