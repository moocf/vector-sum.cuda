#pragma once


// Calculate vector sum on CPU
void vector_sum(float* a, float* x, float* y, int N) {
  for (int i = 0; i < N; i++)
    a[i] = x[i] + y[i];
}
