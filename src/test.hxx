#pragma once
#include <cstdio>
#include "_main.hxx"

using std::printf;


// Calculate vector sum on CPU
void vectorSum(float* a, float* x, float* y, int N) {
  for (int i = 0; i < N; i++)
    a[i] = x[i] + y[i];
}

// Validate if vector sum is correct.
bool validateSum(int *a, int *x, int *y, int N) {
  for (int i=0; i<N; i++)
    if (a[i] != x[i] + y[i]) return 0;
  return 1;
}

// Prints the output of a run.
void printRun(float exp, float *ans, int N, float duration) {
  float act = sum(ans, N);
  printf("Execution time: %3.1f ms\n", duration);
  if (exp != act) printf("Result is invalid!\n");
  printf("\n");
}
