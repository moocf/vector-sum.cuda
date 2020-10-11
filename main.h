#pragma once
#include "support.h"


// Calculate vector sum on CPU
void vector_sum(float* a, float* x, float* y, int N) {
  for (int i = 0; i < N; i++)
    a[i] = x[i] + y[i];
}

// Validate if vector sum is correct.
bool validate_sum(int *a, int *x, int *y, int N) {
  for (int i=0; i<N; i++)
    if (a[i] != x[i] + y[i]) return 0;
  return 1;
}

// Prints the output of a run.
void printrun(float exp, float *ans, int N, float duration) {
  float act = SUM_ARRAY(ans, N);
  printf("Execution time: %3.1f ms\n", duration);
  if (exp != act) printf("Result is invalid!\n");
  printf("\n");
}
