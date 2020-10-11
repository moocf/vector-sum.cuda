#include <stdio.h>
#include "main.h"
#include "_host.h"
#include "_simple.h"


// 1. Allocate space for 3 vectors A, X, Y (of length 1000000).
// 2. Define vectors X and Y (A = X + Y will be computed).
// 3. Calculate expected value for varifying results.
// 4. Run vector sum on CPU.
// 5. Run vector sume on GPU, simple.
// 6. Free vectors A, X, Y.
int main() {
  int N = 1000000;                 // 1
  size_t N1 = N * sizeof(int);     // 1

  float *a = (float*) malloc(N1);  // 1
  float *x = (float*) malloc(N1);  // 1
  float *y = (float*) malloc(N1);  // 1
  for (int i=0; i<N; i++) { // 2
    x[i] = i * 0.2f;        // 2
    y[i] = i * -0.1f;       // 2
  }

  vector_sum(a, x, y, N);      // 3
  float exp = SUM_ARRAY(a, N); // 3

  printf("CPU vector-sum ...\n");            // 4
  printrun(exp, a, N, run_host(a, x, y, N)); // 4
  
  printf("GPU vector-sum, simple ...\n");      // 5
  printrun(exp, a, N, run_simple(a, x, y, N)); // 5

  free(y); // 6
  free(x); // 6
  free(a); // 6
  return 0;
}
