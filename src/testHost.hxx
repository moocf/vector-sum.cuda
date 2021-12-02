#pragma once
#include <ctime>
#include "test.hxx"

using std::clock_t;
using std::clock;


// Run vector-sum on CPU.
//
// a: output vector (x + y)
// x: input vector 1
// y: input vector 2
// N: vector size (a, x, y)
float testHost(float* a, float* x, float* y, int N) {
  clock_t begin = clock();
  vectorSum(a, x, y, N);
  clock_t end = clock();

  float duration = (float) (end - begin) / CLOCKS_PER_SEC;
  return duration * 1000;
}
